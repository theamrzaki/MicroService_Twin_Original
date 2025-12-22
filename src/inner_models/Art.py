import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class ARTWrapper(nn.Module):
    def __init__(self, adj, 
                 raw_metric, raw_logs, raw_traces, 
                 feature_metric, feature_logs, feature_traces):
        super().__init__()
        self.metric_proj = nn.Linear(raw_metric, feature_metric)
        self.log_proj    = nn.Linear(raw_logs, feature_logs)
        self.trace_proj = nn.Linear(raw_traces, feature_traces)

        tf_in_dim = adj.size(0) # num_services
        num_heads = 2
        gnn_in_dim = feature_metric + feature_logs + feature_traces
        gnn_hidden_dim = 64
        gnn_out_dim = 32
        gru_hidden_dim = 32
        dropout = 0.3
        tf_layers = 1
        gnn_layers = 2
        gru_layers = 1
        self.model = AutoRegressor(tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout, tf_layers, gnn_layers, gru_layers)
        def normalize_adj(adj):
            adj = adj.float()                 # 
            deg = adj.sum(dim=1)              # (N,)
            deg_inv = deg.pow(-1)              # safe now
            deg_inv[torch.isinf(deg_inv)] = 0
            D_inv = torch.diag(deg_inv)
            return D_inv @ adj

        adj = normalize_adj(adj).float()
        self.register_buffer("adj", adj)

        self.out = nn.Linear(gnn_in_dim, raw_metric + raw_logs + raw_traces)

    def forward(self, metrics, logs, traces):
        # metrics: B,T,N,raw_metric
        # logs:    B,T,N,raw_logs
        # traces:  B,T,N,N,raw_trace

        m = self.metric_proj(metrics)
        L = self.log_proj(logs)
        t = self.trace_proj(traces).mean(dim=3)
        features = torch.cat([m, L, t], dim=-1)

        rec = self.model(self.adj, features)
        rec = F.relu(self.out(rec))

        return rec





# -------------------- Neural Network Architecture -------------------------
# Transformer - Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim*2,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

    def forward(self, features):
        # (batch_size, sequence_length, hidden_size)
        # print(x.shape)
        h = F.leaky_relu(self.transformer_encoder(features))
        return h

class GraphSAGEEncoder_dynamic_graph_notused(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm):
        super(GraphSAGEEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_dim = hidden_dim if num_layers > 1 else out_dim
        self.input_conv = dgl.nn.GraphConv(in_dim, hidden_dim, norm=norm)
        self.convs = []
        for _ in range(num_layers - 2):
            self.convs.append(dgl.nn.GraphConv(hidden_dim, hidden_dim, norm=norm))
        if num_layers > 1:
            self.convs.append(dgl.nn.GraphConv(hidden_dim, out_dim, norm=norm))

    def forward(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
            h = self.dropout(h)
        return h
    
    def transform(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
        return h

class StaticGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, adj):
        """
        x   : (B*N, F)
        adj : (N, N)
        """
        N = adj.size(0)

        # x: (B*N, F) → (B, N, F)
        B = x.size(0) // N
        x = x.view(B, N, -1)

        # Linear transform
        h = self.linear(x)             # (B, N, F_out)

        # Graph aggregation per batch
        h = torch.matmul(adj, h)       # (B, N, F_out)

        # Flatten back
        return h.view(B * N, -1)
    
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        hidden_dim = hidden_dim if num_layers > 1 else out_dim

        self.input_conv = StaticGraphConv(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(StaticGraphConv(hidden_dim, hidden_dim))

        if num_layers > 1:
            self.convs.append(StaticGraphConv(hidden_dim, out_dim))

    def forward(self, adj, features):
        """
        adj      : (N, N) normalized adjacency
        features : (B*N, F)
        """
        h = F.leaky_relu(self.input_conv(features, adj))
        h = self.dropout(h)

        for conv in self.convs:
            h = F.leaky_relu(conv(h, adj))
            h = self.dropout(h)

        return h

    def transform(self, adj, features):
        h = F.leaky_relu(self.input_conv(features, adj))
        for conv in self.convs:
            h = F.leaky_relu(conv(h, adj))
        return h
    
class Extractor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout=0, tf_layers=1, gnn_layers=2, gru_layers=1):
        super(Extractor, self).__init__()
        self.TFEncoder = TransformerEncoder(tf_in_dim, num_heads, tf_layers)
        self.GRUEncoder = nn.GRU(gnn_in_dim, gru_hidden_dim, gru_layers, bias=False, batch_first=True)
        self.GraphEncoder = GraphSAGEEncoder(gru_hidden_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers, dropout)#, norm='none')
        
    def forward(self, g, features):
        bacth_size, series_len, instance_num, channel_dim = features.shape # 2,5,46,130
        h = features.permute(0,1,3,2)
        h = h.view(-1, channel_dim, instance_num)
        h = self.TFEncoder(h)
        h = h.permute(0,2,1).view(bacth_size, series_len, instance_num, channel_dim).permute(0,2,1,3).reshape(-1, series_len, channel_dim) # 92,5,130
        output, h_n = self.GRUEncoder(h)
        h = F.leaky_relu(h_n[-1]) # 92,32
        h = self.GraphEncoder(g, h) # 92, 32
        h = h.view(bacth_size, instance_num, -1) # 2,46,32
        return h
    
class Regressor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Regressor, self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
        
    def forward(self, features):
        h = F.leaky_relu(self.mlp(features))
        return h

class AutoRegressor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout=0, tf_layers=1, gnn_layers=2, gru_layers=1):
        super(AutoRegressor, self).__init__()
        self.extractor = Extractor(tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout, tf_layers, gnn_layers, gru_layers)
        self.regressor = Regressor(gru_hidden_dim, gnn_in_dim)
        
    def forward(self, g, features):
        z = self.extractor(g, features)
        h = self.regressor(z)
        #return z, h
        return h #reconstructed features
# end Neural Network Architecture -----------------------------------------

