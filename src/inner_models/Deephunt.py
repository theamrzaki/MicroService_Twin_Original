import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# PyTorch Geometric imports
from torch_geometric.nn import GraphConv
from torch_geometric.data import Batch

import numpy as np
from torch_geometric.utils import add_self_loops

class DeepHuntWrapper(nn.Module):
    def __init__(self, adj, 
                 raw_metric, raw_logs, raw_traces, 
                 feature_metric, feature_logs, feature_traces,
                 hidden_feats=64, out_feats=32, dropout=0.3, num_layers=2):
        super().__init__()
        
        # 1. Linear Projection Layers per modality
        self.metric_proj = nn.Linear(raw_metric, feature_metric)
        self.log_proj    = nn.Linear(raw_logs, feature_logs)
        self.trace_proj   = nn.Linear(raw_traces, feature_traces)

        # 2. Derive dimensions for the GNN Autoencoder
        in_feats = feature_metric + feature_logs + feature_traces
        self.num_nodes = adj.shape[0] if not torch.is_tensor(adj) else adj.size(0)

        # 3. Instantiate base DeepHunt Autoencoder (GraphSAGE or GATAE)
        self.model = GraphSAGE(
            in_feats=in_feats, 
            hidden_feats=hidden_feats, 
            out_feats=out_feats, 
            dropout=dropout, 
            num_layers=num_layers
        )

        # 4. Adjacency Matrix to PyG edge_index conversion
        if torch.is_tensor(adj):
            adj_np = adj.detach().cpu().numpy()
        else:
            adj_np = adj

        src, dst = np.nonzero(adj_np)
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        
        # Add self-loops to prevent isolated node messaging failures
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        # Register PyG buffers for GPU handling
        self.register_buffer('adj_matrix_fixed', adj if torch.is_tensor(adj) else torch.tensor(adj))
        self.register_buffer('edge_index', edge_index)

        # 5. Output Reconstruction projection back to raw feature space
        total_raw_dim = raw_metric + raw_logs + raw_traces
        self.out = nn.Linear(in_feats, total_raw_dim)

    def forward(self, metrics, logs, traces):
        """
        metrics: (B, T, N, raw_metric)
        logs:    (B, T, N, raw_logs)
        traces:  (B, T, N, N, raw_traces)
        """
        batch_size, series_len, num_nodes, _ = metrics.shape

        # Feature Projection & Fusion across modalities
        m = self.metric_proj(metrics)
        L = self.log_proj(logs)
        t = self.trace_proj(traces).mean(dim=3)  # Pool matrix across trace dest dimension N
        features = torch.cat([m, L, t], dim=-1)   # (B, T, N, in_feats)

        # Reshape to 2D matrix for PyG Graph Convolutions: (B * T * N, in_feats)
        x = features.view(-1, features.size(-1))

        # Dynamically batch PyG edge_index across (B * T) graph snapshots
        total_graphs = batch_size * series_len
        edge_indices = [self.edge_index + i * self.num_nodes for i in range(total_graphs)]
        batched_edge_index = torch.cat(edge_indices, dim=1).to(x.device)

        # Pass through the PyG DeepHunt Graph Autoencoder
        x_hat = self.model(batched_edge_index, x)

        # Reshape back to sequence structure: (B, T, N, in_feats)
        x_hat = x_hat.view(batch_size, series_len, num_nodes, -1)

        # Project back to original feature dimensions
        rec = F.relu(self.out(x_hat))
        return rec
    
# -------------------- Neural Network Architecture -------------------------
# MeanSage - Encoder
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout, num_layers):
        super(GraphSAGEEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_feats = hidden_feats if num_layers > 1 else out_feats
        
        # Explicit input_conv matching your original structure
        self.input_conv = GraphConv(in_feats, hidden_feats)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_feats, hidden_feats))
        if num_layers > 1:
            self.convs.append(GraphConv(hidden_feats, out_feats))

    def forward(self, edge_index, features):
        # Layer 1 (input_conv)
        h = F.leaky_relu(self.input_conv(features, edge_index))
        h = self.dropout(h)
        
        # Remaining layers in self.convs
        for conv in self.convs:
            h = F.leaky_relu(conv(h, edge_index))
            h = self.dropout(h)
        return h

    def transform(self, edge_index, features):
        h = F.leaky_relu(self.input_conv(features, edge_index))
        for conv in self.convs:
            h = F.leaky_relu(conv(h, edge_index))
        return h

# MeanSage - Decoder
class GraphSAGEDecoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout, num_layers):
        super(GraphSAGEDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_feats = hidden_feats if num_layers > 1 else out_feats
        
        # 1. First layer is assigned ONLY to self.input_conv
        self.input_conv = GraphConv(in_feats, hidden_feats)
        
        # 2. self.convs holds only the remaining layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_feats, hidden_feats))
            
        if num_layers > 1:
            self.convs.append(GraphConv(hidden_feats, out_feats))

    def forward(self, edge_index, features):
        h = F.leaky_relu(self.input_conv(features, edge_index))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(h, edge_index))
            h = self.dropout(h)
        return h

    def transform(self, edge_index, features):
        h = F.leaky_relu(self.input_conv(features, edge_index))
        for conv in self.convs:
            h = F.leaky_relu(conv(h, edge_index))
        return h

# GraphSAGE - AE
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.0, mask_rate=0.0, num_layers=2, norm='none'):
        super(GraphSAGE, self).__init__()
        self.encoder = GraphSAGEEncoder(in_feats, hidden_feats, out_feats, dropout, num_layers)
        self.decoder = GraphSAGEDecoder(out_feats, hidden_feats, in_feats, dropout, num_layers)
        self.mask_rate = mask_rate
    
    def forward(self, edge_index, features):
        z = self.encoder(edge_index, features)
        x_hat = self.decoder(edge_index, z)
        return x_hat
    
    def transform(self, edge_index, features):
        z = self.encoder.transform(edge_index, features)
        x_hat = self.decoder.transform(edge_index, z)
        return x_hat

# Linear - Encoder/Decoder
class LinearCoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0):
        super(LinearCoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, out_feats)

    def forward(self, z):
        h = F.leaky_relu(self.fc1(z))
        h = self.dropout(h)
        x_hat = self.fc2(h)
        return x_hat
    
    def transform(self, z):
        h = F.leaky_relu(self.fc1(z))
        x_hat = self.fc2(h)
        return x_hat

# Linear - AE
class MLPAE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(MLPAE, self).__init__()
        self.encoder = LinearCoder(in_feats, hidden_feats, out_feats)
        self.decoder = LinearCoder(out_feats, hidden_feats, in_feats)

    def forward(self, edge_index, features):
        z = self.encoder(features)
        x_hat = self.decoder(z)
        return x_hat
    
    def transform(self, edge_index, features):
        z = self.encoder.transform(features)
        x_hat = self.decoder.transform(z)
        return x_hat

# GAT - Encoder/Decoder
class GATCoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, dropout):
        super(GATCoder, self).__init__()
        # Concatenation is enabled here for multi-head representation
        self.conv1 = GATv2Conv(in_feats, hidden_feats, heads=num_heads, concat=True, dropout=dropout)
        # Final output layer projects multi-head hidden dim back to out_feats
        self.conv2 = GATv2Conv(hidden_feats * num_heads, out_feats, heads=1, concat=False, dropout=dropout)

    def forward(self, edge_index, feat):
        h = F.elu(self.conv1(feat, edge_index))
        h = self.conv2(h, edge_index)
        return h

# GAT - AE
class GATAE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GATAE, self).__init__()
        self.encoder = GATCoder(in_feats, hidden_feats, out_feats, num_heads=3, dropout=0)
        self.decoder = GATCoder(out_feats, hidden_feats, in_feats, num_heads=3, dropout=0)

    def forward(self, edge_index, features):
        z = self.encoder(edge_index, features)
        x_hat = self.decoder(edge_index, z)
        return x_hat
# end Neural Network Architecture -----------------------------------------


# -------------------- Collate Function -------------------------

# Creating a DataLoader for gae pre_training.
# Sample Format: (timestamp, topo, node_feats)
def collate(samples):
    timestamps, graphs, feats = map(list, zip(*samples))
    batched_graph = Batch.from_data_list(graphs)
    return timestamps, batched_graph, torch.cat(feats, dim=0)

def create_dataloader(samples, batch_size, shuffle=True):
    dataset = list(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloader

def collate_fd(samples):
    graphs, feats, labels = map(list, zip(*samples))
    batched_graph = Batch.from_data_list(graphs)
    feats = torch.stack([torch.tensor(step) for series in feats for step in series])
    feats = feats.view(-1, feats.shape[-3]//len(labels), feats.shape[-2], feats.shape[-1]).permute(0,2,1,3)
    shape = feats.shape
    feats = feats.reshape(-1, shape[-2], shape[-1])
    return batched_graph, feats, torch.tensor(np.array([lb.cpu().detach().numpy() for lb in labels])).view(-1)

def create_dataloader_fd(fd_samples, labels, batch_size, shuffle=False):
    dataset = [[fd_samples[i][-1][1], [step[2] for step in fd_samples[i]], labels[i]] for i in range(len(labels))]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fd)
    return dataloader
# end Collate Function  ------------------------------------------


# -------------------- Loss Function -------------------------

# Loss_version_0: torch.nn.MSELoss()

# Loss_version_1: Calculate the MSE loss separately for different modalities of data and then sum them up.
# for gae pre_training.
class ModalLoss(nn.Module):
    def __init__(self, feat_span):
        super(ModalLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.feat_span = feat_span

    def forward(self, inputs, targets):
        h = 0
        for start, end in self.feat_span:
            loss = self.mse(inputs[:, start: end + 1], targets[:, start: end + 1])
            h += loss
        return h
    
    def compute(self, inputs, targets):
        h = 0
        for start, end in self.feat_span:
            loss = self.mse(inputs[start: end + 1], targets[start: end + 1])
            h += loss
        return h

# Loss_version_2
# for feedback.
class RankingLoss(nn.Module):
    def __init__(self, rank_range):
        super(RankingLoss, self).__init__()
        self.rank_range = rank_range
        self.weights = torch.Tensor([(1/i) for i in range(1, rank_range+1)]) ** 1
    
    def forward(self, predicted_scores, labels):
        loss = []
        for start in range(0, len(labels), self.rank_range):
            end = start + self.rank_range
            tmp_s = predicted_scores[start:end]
            tmp_l = labels[start:end]
            tmp_s = (tmp_s - torch.min(tmp_s))/(torch.max(tmp_s) - torch.min(tmp_s))
            tmp_s = (torch.sort(tmp_s, descending=True).values - torch.mean(tmp_s[np.where(tmp_l==1)])) * self.weights
            loss.append(torch.sum(tmp_s[tmp_s>0]))
        return torch.mean(torch.stack(loss))
# end Loss Function  ------------------------------------------
