import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Replaced DGL imports with PyG equivalents
from torch_geometric.nn import GATv2Conv, GlobalAttention

class GraphModel1(nn.Module):
    def __init__(self, in_dim, graph_hiddens=[64], device='cpu',
                 attn_head=4, activation=0.2, **kwargs):
        super().__init__()
        
        graph_hiddens = [in_dim]
        self.convs = nn.ModuleList()
        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i - 1] if i > 0 else in_dim
            dropout = kwargs.get("attn_drop", 0)
            self.convs.append(GATv2Conv(
                in_channels=in_feats, 
                out_channels=hidden, 
                heads=attn_head,
                dropout=dropout, 
                concat=True,
                negative_slope=activation,
                add_self_loops=True
            ))
            
        self.out_dim = graph_hiddens[-1]
        self.attn_head = attn_head
        self.maxpool = nn.MaxPool1d(attn_head)
        
        # REMOVED: GlobalAttention pooling (was collapsing nodes into 1 vector per graph) --> as we need to build node-level embeddings for reconstruction, we will not use global pooling here
        self.to(device)

    def forward(self, edge_index, x, batch=None):
        """
        edge_index: Graph connectivity tensor [2, num_edges]
        x: Node feature matrix [num_nodes, in_dim] (e.g., [1200, 64])
        
        Returns:
            out: Node-level embeddings [num_nodes, out_dim] (e.g., [1200, 64])
        """
        out = x
        
        # Guard: Collapse 3D sequence inputs (e.g., [1200, 10, 64] -> [1200, 64]) if passed directly
        if out.dim() == 3:
            out = out.mean(dim=1)

        for conv in self.convs:
            # 1. GATv2Conv with concat=True -> [1200, 256]
            out = conv(out, edge_index)
            
            # 2. Reshape to 3D for MaxPool -> [1200, 4, 64]
            out = out.view(-1, self.attn_head, conv.out_channels)
            
            # 3. FIXED 2: Specify .squeeze(-1) to safely collapse ONLY the pooled head dimension
            # [1200, 4, 64] -> permute -> [1200, 64, 4] -> MaxPool1d -> [1200, 64, 1] -> squeeze(-1) -> [1200, 64]
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze(-1)
            
        # FIXED 3: Directly return node embeddings [1200, 64] (no global pooling)
        return out


class SimpleAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self._glorot_init()

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights) + self.bias
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return x * attention_weights

    def _glorot_init(self):
        stdv = math.sqrt(6.0 / (self.attention_weights.size(-2) + self.attention_weights.size(-1)))
        self.attention_weights.data.uniform_(-stdv, stdv)