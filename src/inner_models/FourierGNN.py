import torch
import torch.nn as nn
import torch.nn.functional as F

class FGN(nn.Module):
    def __init__(self, pre_length, embed_size,
                 feature_size, seq_length, hidden_size, hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        #self.embeddings = nn.Parameter(torch.randn(20, self.embed_size)) #20 = metric + log 
        self.tokenEmb = nn.Linear(in_features=20, out_features=self.embed_size)#20 = metric + log 

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.to('cuda:0')

    #def tokenEmb(self, x):
    #    #x = x.unsqueeze(2)
    #    y = self.embeddings
    #    return x * y

    # FourierGNN
    def fourierGC(self, x, adj_ft, B, N, L):
        """
            x: [B, T, N, D] in complex domain (T = time/frequency steps)
            adj_ft: [B, T, N, N] complex-valued adjacency in frequency domain
        """
        # x: [B, T, N, D]
        T = (N * L) // 2 + 1
        D = self.frequency_size

        # Graph conv: x' = A x W
        def graph_conv(x_real, x_imag, W_real, W_imag, bias_real, bias_imag):
            x_real_agg = torch.einsum('btij,btjd->btid', adj_ft.real, x_real)
            x_imag_agg = torch.einsum('btij,btjd->btid', adj_ft.real, x_imag)

            out_real = F.relu(torch.einsum('btid,df->btif', x_real_agg, W_real) -
                            torch.einsum('btid,df->btif', x_imag_agg, W_imag) +
                            bias_real)

            out_imag = F.relu(torch.einsum('btid,df->btif', x_imag_agg, W_real) +
                            torch.einsum('btid,df->btif', x_real_agg, W_imag) +
                            bias_imag)
            return out_real, out_imag

        # Layer 1
        o1_real, o1_imag = graph_conv(x.real, x.imag, self.w1[0], self.w1[1], self.b1[0], self.b1[1])
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        # Layer 2
        o2_real, o2_imag = graph_conv(o1_real, o1_imag, self.w2[0], self.w2[1], self.b2[0], self.b2[1])
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        # Layer 3
        o3_real, o3_imag = graph_conv(o2_real, o2_imag, self.w3[0], self.w3[1], self.b3[0], self.b3[1])
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x

        return torch.view_as_complex(z)

    def compute_edge_features(self,node_embeds):
        # node_embeds: [B, N, D]
        B, N, D = node_embeds.shape
        source = node_embeds.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, D]
        target = node_embeds.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, D]
        edge_feats = torch.cat([source, target], dim=-1)      # [B, N, N, 2*D]
        edge_feats = edge_feats.view(B, N*N, 2*D)             # flatten edges
        return edge_feats

    def forward(self, x, adj_ft):
        #x = x.permute(0, 2, 1).contiguous()
        B, N, L, F = x.shape
        # B*N*L ==> B*NL
        #x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x) # [B,N,L, embed_size]

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=2, norm='ortho')   #[B,N,6, embed_size]
            
        #x = x.reshape(B, (N*L)//2+1, self.frequency_size)
        # Rearranging for graph conv: [B, N, T, D] → [B, T, N, D]
        x = x.permute(0, 2, 1, 3).contiguous() #[B,6,N, embed_size]
        bias = x
        adj_ft = torch.fft.rfft(adj_ft, dim=1, norm='ortho')  # assuming dim=1 is time dimension

        # FourierGNN
        x = self.fourierGC(x, adj_ft, B, N, L)

        x = x + bias

        #x = x.reshape(B, (N*L)//2+1, self.embed_size)
        # Rearrange back: [B, T, N, D] → [B, N, T, D]
        x = x.permute(0, 2, 1, 3).contiguous()

        # ifft
        x = torch.fft.irfft(x, n=L, dim=2, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)

        return x # [B, N, embed_size]

