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
        
        # PyG GATv2Conv layers stored in a ModuleList
        self.convs = nn.ModuleList()
        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i - 1] if i > 0 else in_dim
            dropout = kwargs.get("attn_drop", 0)
            self.convs.append(GATv2Conv(
                in_channels=in_feats, 
                out_channels=hidden, 
                heads=attn_head,
                dropout=dropout, 
                negative_slope=activation,
                add_self_loops=True
            ))
            
        self.out_dim = graph_hiddens[-1]
        
        # PyG Global Attention Pooling equivalent
        gate_nn = nn.Linear(self.out_dim, 1)
        self.pooling = GlobalAttention(gate_nn=gate_nn)
        
        self.maxpool = nn.MaxPool1d(attn_head)
        self.to(device)

    def forward(self, edge_index, x, batch=None):
        """
        edge_index: Graph connectivity tensor [2, num_edges]
        x: Node feature matrix [num_nodes, in_dim]
        batch: Graph assignment vector for batched graphs [num_nodes]. 
               If single graph, batch will default to zeros.
        """
        out = x
        for conv in self.convs:
            # PyG GATv2Conv takes (x, edge_index) and outputs [num_nodes, num_heads, out_channels]
            out = conv(out, edge_index)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
            
        if batch is None:
            batch = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
            
        return self.pooling(out, batch)


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