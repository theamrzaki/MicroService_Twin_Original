import torch as t
import sys
sys.path.append('./')
from src.inner_models.layers.GTblock import GTN
from src.inner_models.layers.GATGRU import *
import torch.nn as nn

class permute(nn.Module):
    def __init__(self):
        super(permute, self).__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)

class AnoFusionWrapper(nn.Module):
    def __init__(self,
                 num_services,
                 edge_types,
                 window_size,
                 metric_dim,
                 log_dim,
                 trace_dim,
                 out_dim):
        super().__init__()

        self.num_services = num_services
        self.out_dim = out_dim

        # ---- Serialization (paper §4.2) ----
        self.metric_proj = nn.Linear(metric_dim, 1)
        self.log_proj    = nn.Linear(log_dim, 1)
        self.trace_proj  = nn.Linear(1, 1)

        # ---- Core AnoFusion ----
        self.anofusion = Net(
            node_num=num_services,  # 12
            edge_types=2,
            window_samples_num=window_size,
            dropout=0.2
        )

        # ---- Decode  ----
        self.linear_x = nn.Linear(3, 20)

    def forward(self, graph, data_node, data_log, data_edge):
        B, T, N, _ = data_node.shape
        device = data_node.device

        # ---- 1. Project each modality to scalar ----
        data_node_proj = self.metric_proj(data_node)  # [B, T, N, 1]
        data_log_proj  = self.log_proj(data_log)      # [B, T, N, 1]
        trace_proj     = self.trace_proj(data_edge.mean(dim=3).mean(dim=-1, keepdim=True))  # [B, T, N, 1]

        # ---- 2. Concatenate channels as features ----
        X = torch.cat([data_node_proj, data_log_proj, trace_proj], dim=-1)  # [B, T, N, 3]
        # linear layer to 20
        X = self.linear_x(X)  # [B, T, N, 20]
        θ = min(self.anofusion.window_samples_num, T)

        # ---- 3. Windowing ----
        Xw = X[:, -θ:, :, :]  # [B, θ, N, 3]

        # ---- 4. Prepare adjacency ----
        A = graph.unsqueeze(0).unsqueeze(0).repeat(B, θ, 1, 1, 1)  # [B, θ, N, N, K]

        # ---- 5. Merge batch & window ----
        Bθ = B * θ
        Xw = Xw.view(Bθ, N, 20)       # [B*θ, N, 3]
        A = A.view(Bθ, N, N, -1)     # [B*θ, N, N, K]

        # ---- 6. Net forward ----
        X_pred = self.anofusion(Xw, A)  # [B*θ, N, 3]

        # ---- 8. Restore batch & window ----
        rec_window = X_pred.view(B, θ, N, -1)  # [B, θ, N, out_dim]

        # ---- 9. Pad to full sequence if needed ----
        if θ < T:
            pad = torch.zeros(B, T-θ, N, self.out_dim, device=device)
            rec = torch.cat([pad, rec_window], dim=1)
        else:
            rec = rec_window

        return rec

class Net(nn.Module):
    def __init__(self, node_num, edge_types, window_samples_num, dropout):
        super(Net, self).__init__()
        self.edge_types = edge_types
        self.num_channels = edge_types
        self.node_num = node_num
        self.window_samples_num = window_samples_num
        self.dropout = dropout
        self.GTN = GTN(edge_types=self.edge_types, num_channels=self.num_channels, num_layers=5, norm=False)
        self.GAT_GRU = GAT_GRU(self.window_samples_num, self.node_num, self.num_channels)
        self.flatten = nn.Flatten()
        self.linT = nn.Linear(self.window_samples_num, self.window_samples_num // 2)
    
        self.all = self.window_samples_num * self.node_num
        self.Dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(self.all, 64)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(64, 2)
        self._final_softmax = nn.Softmax(dim=1)
      
        
    def forward(self, X, A):
        X = self.Dropout(X)
        A = A.view((-1, self.node_num, self.node_num, self.edge_types))
        X = X.view((-1, self.node_num, X.shape[-1]))
        # GTN
        device = X.device
        A = self.GTN(A)
        # GAT and GRU
        out_T = self.GAT_GRU(X, A)
        return out_T