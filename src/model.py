import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
print("1. model ")
#from torch_geometric.utils import dense_to_sparse
print("2. model ")
#from src.model_util import *
from src.inner_models.FITS import Model as FITSModel 
from src.inner_models.FITS_LPF import Model as FITSModel_LPF 
from src.inner_models.FITS_Pai import Model as FITSModel_Pai
from src.inner_models.iTransformer import Model as iTransformerModel
from src.inner_models.DLinear import Model as DLinearModel
from src.inner_models.FourierGNN import FGN
#from src.inner_models.GPT4TS import Model as GPT2Model
from src.inner_models.FreTS import Model as FreTSModel
from src.inner_models.TimesNet import Model as TimesNetModel
from src.inner_models.FEDformer import Model as FEDformerModel
from src.inner_models.FITS_Legendre import Model as FITSModel_Legendre
from src.inner_models.FITS_chebyshev import Model as FITS_chebyshev
from src.inner_models.FITS_lag import Model as FITS_lag
from src.inner_models.FITS_hermite import Model as FITS_hermite
import src.inner_models.FITS_Legendre as FITS_Legendre_operations
import src.inner_models.FITS_chebyshev as FITS_chebyshev_operations
import src.inner_models.FITS_lag as FITS_lag_operations
import src.inner_models.FITS_hermite as FITS_hermite_operations
from src.inner_models.Eadro import MainModel 	
from src.inner_models.Anofusion import AnoFusionWrapper as AnoFusion
##from src.inner_models.Art import ARTWrapper as Art_Model
#from src.inner_models.Hades import HadesWrapper as Hades_Model
from util.util import is_raspberry_pi
import numpy as np
import argparse
print("3. model ")
from numpy.polynomial import Legendre as L



class Temporal_Attention(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, trace2pod, heads_node=4, heads_edge=4, heads_log=4, dropout=0.1,
                 window_size=16, batch_size=10):
        super(Temporal_Attention, self).__init__()
        self.window_size = window_size
        self.batch_size = batch_size
        self.trace2pod = trace2pod
        
        self.attention_node = nn.MultiheadAttention(embed_dim=node_embedding_dim, num_heads=heads_node,
                                                    dropout=dropout,batch_first=True)
        self.attention_trace = nn.MultiheadAttention(embed_dim=edge_embedding_dim, num_heads=heads_edge,
                                                     dropout=dropout, batch_first=True)
        self.attention_log = nn.MultiheadAttention(embed_dim=log_embedding_dim, num_heads=heads_log,
                                                     dropout=dropout, batch_first=True)

        self.vff_node = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.vff_trace = nn.Linear(edge_embedding_dim, edge_embedding_dim)
        self.vff_log = nn.Linear(log_embedding_dim, log_embedding_dim)
        self.headff_node = nn.Linear(heads_node * window_size, window_size)
        self.headff_trace = nn.Linear(heads_edge * window_size, window_size)
        self.headff_log = nn.Linear(heads_log * window_size, window_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_node, x_trace, x_log, mask=False):
        x_node = x_node.permute(0, 2, 1, 3).reshape(-1, self.window_size, x_node.shape[-1])
        x_trace = x_trace.permute(0, 2, 1, 3).reshape(-1, self.window_size, x_trace.shape[-1])
        x_log = x_log.permute(0, 2, 1, 3).reshape(-1, self.window_size, x_log.shape[-1])

        if mask:
            mask_att = (torch.triu(torch.ones(self.window_size, self.window_size, device=x_node.device)) == 1).transpose(0, 1)
            mask_att = mask_att.float().masked_fill(mask_att == 0, float('-inf')).masked_fill(mask_att == 1, float(0.0))

            att_n = self.attention_node(x_node, x_node, x_node, attn_mask=mask_att, average_attn_weights=False)[1]
            att_t = self.attention_trace(x_trace, x_trace, x_trace, attn_mask=mask_att, average_attn_weights=False)[1]
            att_l = self.attention_log(x_log, x_log, x_log, attn_mask=mask_att, average_attn_weights=False)[1]
        else:
            att_n = self.attention_node(x_node, x_node, x_node, average_attn_weights=False)[1]
            att_t = self.attention_trace(x_trace, x_trace, x_trace, average_attn_weights=False)[1]
            att_l = self.attention_log(x_log, x_log, x_log, average_attn_weights=False)[1]

        att_n = att_n.reshape(self.batch_size, -1, att_n.shape[-3], att_n.shape[-2], att_n.shape[-1])
        att_t = att_t.reshape(self.batch_size, -1, att_t.shape[-3], att_t.shape[-2], att_t.shape[-1])
        att_l = att_l.reshape(self.batch_size, -1, att_l.shape[-3], att_l.shape[-2], att_l.shape[-1])
        device = att_n.device  # or att_t.device / att_l.device — they should all be the same

        self.trace2pod = self.trace2pod.to(device)
        att_nn = torch.matmul(att_n.permute(0, 2, 3, 4, 1), self.trace2pod.T.float()).permute(0, 4, 1, 2, 3)
        att_tn = torch.matmul(att_t.permute(0, 2, 3, 4, 1), self.trace2pod.float()).permute(0, 4, 1, 2, 3)
        att_ln = torch.matmul(att_l.permute(0, 2, 3, 4, 1), self.trace2pod.T.float()).permute(0, 4, 1, 2, 3)

        att_node = torch.concat([att_n.mean(axis=[1,2], keepdims=True), att_tn.mean(axis=[1,2], keepdims=True), att_l.mean(axis=[1,2], keepdims=True)], dim=1).mean(axis=1, keepdims=True)
        att_edge = torch.concat([att_nn.mean(axis=[1,2], keepdims=True), att_t.mean(axis=[1,2], keepdims=True), att_ln.mean(axis=[1,2], keepdims=True)], dim=1).mean(axis=1, keepdims=True)

        x_node = torch.bmm(
            self.softmax(att_node + att_n).reshape(att_n.shape[0] * att_n.shape[1], att_n.shape[2] * att_n.shape[3],
                                              att_n.shape[-1]), self.vff_node(x_node))
        x_trace = torch.bmm(
            self.softmax(att_edge + att_t).reshape(att_t.shape[0] * att_t.shape[1], att_t.shape[2] * att_t.shape[3],
                                              att_t.shape[-1]), self.vff_trace(x_trace))
        x_log = torch.bmm(
            self.softmax(att_node + att_l).reshape(att_l.shape[0] * att_l.shape[1], att_l.shape[2] * att_l.shape[3],
                                              att_l.shape[-1]), self.vff_log(x_log))
    
        x_node = self.headff_node(x_node.permute(0, 2, 1)).permute(0, 2, 1) \
            .reshape(self.batch_size, -1, self.window_size, x_node.shape[-1]).permute(0, 2, 1, 3)
        x_trace = self.headff_trace(x_trace.permute(0, 2, 1)).permute(0, 2, 1) \
            .reshape(self.batch_size, -1, self.window_size, x_trace.shape[-1]).permute(0, 2, 1, 3)
        x_log = self.headff_log(x_log.permute(0, 2, 1)).permute(0, 2, 1) \
            .reshape(self.batch_size, -1, self.window_size, x_log.shape[-1]).permute(0, 2, 1, 3)
        return x_node, x_trace, x_log


class Spatial_Attention(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, heads_n2e=4, heads_e2n=4, dropout=0.1, batch_size=10,
                 window_size=16):
        super(Spatial_Attention, self).__init__()

        self.batch_size = batch_size
        self.window_size = window_size

        self.node2node = GATv2Conv(in_channels=node_embedding_dim + log_embedding_dim,
                                   out_channels=int((node_embedding_dim + log_embedding_dim) / heads_n2e),
                                   heads=heads_n2e, dropout=dropout, edge_dim=edge_embedding_dim, add_self_loops=False)
        self.egde2node = GATv2Conv(in_channels=edge_embedding_dim, out_channels=int(edge_embedding_dim / heads_e2n),
                                   heads=heads_e2n, dropout=dropout, edge_dim=node_embedding_dim + log_embedding_dim,
                                   add_self_loops=False)

    def forward(self, x_node, x_trace, x_log, node_adj, edge_adj, edge_efea):
        node = torch.concat([x_node, x_log], dim=-1)
        node = node.reshape(-1, node.shape[-1])
        x_trace = x_trace.reshape(-1, x_trace.shape[-1])
        
        node = self.node2node(node, node_adj, x_trace)
        x_trace = self.egde2node(x_trace, edge_adj, node[edge_efea.long()])

        x_node = node[:, :x_node.shape[-1]].reshape(self.batch_size, self.window_size, -1, x_node.shape[-1])
        x_trace = x_trace.reshape(self.batch_size, self.window_size, -1, x_trace.shape[-1])
        x_log = node[:, x_node.shape[-1]:].reshape(self.batch_size, self.window_size, -1, x_log.shape[-1])
        return x_node, x_trace, x_log


class Encoder_Decoder_Attention(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, heads_node=4, heads_edge=4, heads_log=4, dropout=0.1):
        super(Encoder_Decoder_Attention, self).__init__()
        self.attention_node = nn.MultiheadAttention(
            embed_dim=node_embedding_dim, num_heads=heads_node, batch_first=True, dropout=dropout)
        self.attention_trace = nn.MultiheadAttention(
            embed_dim=edge_embedding_dim, num_heads=heads_edge, batch_first=True, dropout=dropout)
        self.attention_log = nn.MultiheadAttention(
            embed_dim=log_embedding_dim, num_heads=heads_log, batch_first=True, dropout=dropout)

    def forward(self, x_node, x_trace, x_log, z_node, z_trace, z_log):
        x_node = x_node.reshape(x_node.shape[0], -1, x_node.shape[-1])
        z_node = z_node.reshape(z_node.shape[0], -1, z_node.shape[-1])
        x_node = self.attention_node(x_node, z_node, z_node)[0]

        x_trace = x_trace.reshape(x_trace.shape[0], -1, x_trace.shape[-1])
        z_trace = z_trace.reshape(z_trace.shape[0], -1, z_trace.shape[-1])
        x_trace = self.attention_trace(x_trace, z_trace, z_trace)[0]

        x_log = x_log.reshape(x_log.shape[0], -1, x_log.shape[-1])
        z_log = z_log.reshape(z_log.shape[0], -1, z_log.shape[-1])
        x_log = self.attention_log(x_log, z_log, z_log)[0]

        return x_node, x_trace, x_log



class Encoder(nn.Module):
    def __init__(self, graph, node_embedding, edge_embedding, log_embedding, node_heads, log_heads, edge_heads, n2e_heads, e2n_heads, dropout, batch_size, window_size, num_layer, trace2pod):
        super(Encoder, self).__init__()
        self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = adj2adj(graph, batch_size, window_size, edge_embedding)
        self.L = num_layer
        self.batch_size = batch_size
        self.window_size = window_size

        self.spatial_attention = nn.ModuleList(
            [Spatial_Attention(node_embedding, edge_embedding, log_embedding,
                               heads_n2e=n2e_heads, heads_e2n=e2n_heads, dropout=dropout, batch_size=batch_size,
                               window_size=window_size) for _ in range(self.L)])
        self.sa_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        self.temporal_attention = nn.ModuleList(
            [Temporal_Attention(node_embedding, edge_embedding, log_embedding, trace2pod,
                                heads_node=node_heads, heads_edge=edge_heads, heads_log=log_heads, dropout=dropout, window_size=window_size,
                                batch_size=batch_size) for _ in range(self.L)])
        self.ta_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        self.ffn = nn.ModuleList([FFN(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])


    def forward(self, e_node, e_edge, e_log):
        device = e_edge.device  # or self.node_efea.device, whichever is appropriate
        e_edge = torch.masked_select(e_edge, self.node_efea.bool().to(device)).reshape(e_edge.shape[0], e_edge.shape[1], -1, e_edge.shape[-1])

        #e_edge = torch.masked_select(e_edge, self.node_efea.bool()) \
        #    .reshape(e_edge.shape[0], e_edge.shape[1], -1, e_edge.shape[-1])

        for i in range(self.L):
            device = e_node.device  # ensure consistency across tensors

            # Move static tensors to the same device before use
            self.node_adj = self.node_adj.to(device)
            self.edge_adj = self.edge_adj.to(device)
            self.edge_efea = self.edge_efea.to(device)
            e_node, e_edge, e_log = self.sa_add[i](e_node, e_edge, e_log, *self.spatial_attention[i](e_node, e_edge, e_log, self.node_adj, self.edge_adj, self.edge_efea))
            e_node, e_edge, e_log = self.ta_add[i](e_node, e_edge, e_log, *self.temporal_attention[i](e_node, e_edge, e_log))
            e_node, e_edge, e_log = self.ffn[i](e_node, e_edge, e_log)
        return e_node, e_edge, e_log


class Decoder(nn.Module):
    def __init__(self, graph, node_embedding, edge_embedding, log_embedding, node_heads, log_heads, edge_heads, n2e_heads, e2n_heads, dropout, batch_size, window_size, num_layer, trace2pod):
        super(Decoder, self).__init__()
        self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = adj2adj(graph, batch_size, window_size, edge_embedding)

        self.L = num_layer
        self.batch_size = batch_size
        self.window_size = window_size

        self.spatial_attention = nn.ModuleList(
            [Spatial_Attention(node_embedding, edge_embedding, log_embedding,
                              heads_n2e=n2e_heads, heads_e2n=e2n_heads, dropout=dropout, batch_size=batch_size,
                               window_size=window_size) for _ in range(self.L)])
        self.sa_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        self.temporal_attention = nn.ModuleList(
            [Temporal_Attention(node_embedding, edge_embedding, log_embedding, trace2pod,
                                heads_node=node_heads, heads_edge=edge_heads, heads_log=log_heads, dropout=dropout, window_size=window_size,
                                batch_size=batch_size) for _ in range(self.L)])
        self.ta_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        self.cross_attention = nn.ModuleList(
            [Encoder_Decoder_Attention(node_embedding, edge_embedding, log_embedding, 
                                        heads_node=node_heads, heads_edge=edge_heads, heads_log=log_heads, dropout=dropout)
                                        for _ in range(self.L)])
        self.ca_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])     
        self.ffn = nn.ModuleList([FFN(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])

    def forward(self, d_node, d_edge, d_log, z_node, z_edge, z_log):
        device = d_edge.device
        self.node_efea = self.node_efea.to(device)
        d_edge = torch.masked_select(d_edge, self.node_efea.bool()) \
            .reshape(d_edge.shape[0], d_edge.shape[1], -1, d_edge.shape[-1])
        self.node_adj = self.node_adj.to(device)
        self.edge_adj = self.edge_adj.to(device)
        self.edge_efea = self.edge_efea.to(device)
        for i in range(self.L):
            
            d_node, d_edge, d_log = self.sa_add[i](d_node, d_edge, d_log, *self.spatial_attention[i](d_node, d_edge, d_log, self.node_adj, self.edge_adj, self.edge_efea))
            d_node, d_edge, d_log = self.ta_add[i](d_node, d_edge, d_log, *self.temporal_attention[i](d_node, d_edge, d_log, mask=True))
            d_node, d_edge, d_log = self.ca_add[i](d_node, d_edge, d_log, *self.cross_attention[i](d_node, d_edge, d_log, z_node, z_edge, z_log))
            d_node, d_edge, d_log = self.ffn[i](d_node, d_edge, d_log)
        return d_node, d_edge, d_log


class Embed(nn.Module):
    def __init__(self, raw_dim, embedding_dim, max_len=1000, dim=4):
        super(Embed, self).__init__()
        self.linear = nn.Linear(raw_dim, embedding_dim)
        self.dim = dim
        pe = torch.zeros((1, max_len, embedding_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000,
                                                                                  torch.arange(0, embedding_dim, 2,
                                                                                               dtype=torch.float32) / embedding_dim)
        pe[:, :, 0::2] = torch.sin(X)
        pe[:, :, 1::2] = torch.cos(X)
        if dim == 4:
            pe = pe.unsqueeze(2)
        elif dim == 5:
            pe = pe.unsqueeze(2).unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = self.linear(X)
        if self.dim == 4:
                padding = (0, 0, 0, 0, 1, 0)
                X_new = F.pad(X, padding, "constant", 0)
                return X + Variable(self.pe[:, :X.shape[1], :, :], requires_grad=False), X_new[:, :X.shape[1], :, :] + Variable(self.pe[:, :X.shape[1], :, :], requires_grad=False)
        else:
                padding = (0, 0, 0, 0, 0, 0, 1, 0)
                X_new = F.pad(X, padding, "constant", 0)
                return X + Variable(self.pe[:, :X.shape[1], :, :, :], requires_grad=False), X_new[:, :X.shape[1], :, :, :] + Variable(self.pe[:, :X.shape[1], :, :, :], requires_grad=False)
				
def adj2adj(graph, batch_size, window_size, zdim):
    graph1 = graph.squeeze(0).squeeze(0).repeat(batch_size, window_size, 1, 1) \
        .reshape(-1, graph.shape[-2], graph.shape[-1])
    adj0, adj1, fea = [], [], []
    
    # 1. Native replacement for dense_to_sparse
    node_adj = torch.nonzero(graph1).t()
    node_efea = graph.unsqueeze(-1).repeat(1, 1, zdim)
    
    for num in range(node_adj.shape[1]):
        idx = torch.argwhere(node_adj[1] == num)
        idy = torch.argwhere(node_adj[0] == num)
        adj0.append(idx.repeat(1, idy.shape[0]).reshape(-1))
        adj1.append(idy.repeat(idx.shape[0], 1).reshape(-1))
        fea.append(torch.ones(
            idy.shape[0] * idx.shape[0], device=graph.device) * num)

    adj = torch.stack([torch.concat(adj0), torch.concat(adj1)], dim=0)
    fea = torch.concat(fea)
    
    # 2. Native PyTorch implementation to remove self-loops
    # Find positions where source index equals destination index
    non_self_loop_mask = adj[0] != adj[1]
    
    # Filter both the adjacency list and the edge features
    edge_adj = adj[:, non_self_loop_mask]
    edge_efea = fea[non_self_loop_mask]
    
    return node_adj, node_efea, edge_adj, edge_efea

def phi(x):
    return torch.nn.functional.elu(x) + 1
print("4. model ")
class LinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, Z):  
        # Z: [B*N, M, D]   (M = number of modalities)
        Q = phi(self.Wq(Z))          # [B*N, M, D]
        K = phi(self.Wk(Z))          # [B*N, M, D]
        V = self.Wv(Z)               # [B*N, M, D]

        KV = torch.einsum("bmd,bme->bde", K, V)   # Σ φ(K_j)V_j^T
        Ksum = K.sum(dim=1)                      # Σ φ(K_j)

        out = torch.einsum("bmd,bde->bme", Q, KV)
        denom = torch.einsum("bmd,bd->bm", Q, Ksum).unsqueeze(-1) + 1e-6
        out = out / denom

        return self.norm(self.out(out) + Z)
	
class MyModel(nn.Module):
	def __init__(self, graph, **args):
		super(MyModel, self).__init__()
		self.name = args['FREQ_DOMAIN']
		if is_raspberry_pi():
			self.graph = torch.tensor(graph)#.cuda()
			adj = torch.nonzero(self.graph).t()
		else:
			self.graph = torch.tensor(graph).cuda()
			adj = dense_to_sparse(self.graph)[0]
		self.label_weight = args['label_weight']
		self.multi_fits = args["MULTI_FITS"]
		#
		trace2pod = torch.nn.functional.one_hot(adj[0], num_classes=graph.shape[0]) \
			+ torch.nn.functional.one_hot(adj[1], num_classes=graph.shape[0])
		trace2pod = trace2pod / trace2pod.sum(axis=0, keepdim=True)
		trace2pod = torch.where(torch.isnan(
			trace2pod), torch.full_like(trace2pod, 0), trace2pod)
		
		self.num_classes = graph.shape[0]
		self.FREQ_DOMAIN = args['FREQ_DOMAIN']
		self.req_loss_approach = args['req_loss_approach']
		self.rec_lambda = args['rec_lambda']
		self.auxi_lambda = args['auxi_lambda']
		self.modules_attn = args['modules_attn']

		if self.FREQ_DOMAIN == "encoder_decoder":
			self.encoder = Encoder(graph=self.graph, node_embedding=args['feature_node'], edge_embedding=args['feature_edge'], log_embedding=args['feature_log'],
							node_heads=args['num_heads_node'], log_heads=args['num_heads_log'], edge_heads=args['num_heads_edge'],
							n2e_heads=args['num_heads_n2e'], e2n_heads=args['num_heads_e2n'],
							dropout=args['dropout'], batch_size=args['batch_size'], window_size=args['window'], num_layer=args['num_layer'], trace2pod=trace2pod)
			self.decoder = Decoder(graph=self.graph, node_embedding=args['feature_node'], edge_embedding=args['feature_edge'], log_embedding=args['feature_log'],
							node_heads=args['num_heads_node'], log_heads=args['num_heads_log'], edge_heads=args['num_heads_edge'],
							n2e_heads=args['num_heads_n2e'], e2n_heads=args['num_heads_e2n'],
							dropout=args['dropout'], batch_size=args['batch_size'], window_size=args['window'], num_layer=args['num_layer'], trace2pod=trace2pod)
		elif self.FREQ_DOMAIN in ["FITS_Pai","FITS_LPF","FITS","iTransformer","DLinear", "FreTS","TimesNet", "FEDformerModel","FITS_Legendre","FITS_chebyshev","FITS_lag","FITS_hermite"]:
			class Config: pass
			config = Config()

			config.win_size = args['window']  # Window size
			config.DSR = 1  # Downsampling rate
			config.cutfreq = 0  # Cut frequency for FITS, set to 0 for automatic calculation
			if config.cutfreq == 0:
				config.cutfreq = int((config.win_size / config.DSR)/2)
			assert (config.win_size / config.DSR)/2 >= config.cutfreq, 'cutfreq should be smaller than half of the window size after downsampling'

			config.seq_len = config.win_size//config.DSR
			config.pred_len = config.win_size-config.win_size//config.DSR
			config.individual	= False  
			config.num_class = self.num_classes
			config.filter_used = args['filter_used']
			config.basis_type = args['basis_type']
			self.basis_type = args['basis_type']
			config.degree = 5
			config.use_normlin = args.get('use_normlin', False)

			t = np.linspace(-1, 1, config.seq_len)
			if config.basis_type == "legendre":
				from scipy.special import legendre
				basis = np.array([legendre(i)(t) for i in range(config.degree)])

			elif config.basis_type == "chebyshev":
				from numpy.polynomial.chebyshev import chebvander
				basis = chebvander(t, config.degree - 1).T

			elif config.basis_type == "fourier":
				t = np.linspace(0, 1, config.seq_len)
				device = self.graph.device
				basis = self._build_fourier_basis(t, config.degree, device)

			elif config.basis_type == "hermite":
				from numpy.polynomial.hermite import hermvander
				basis = hermvander(t, config.degree - 1).T

			elif config.basis_type == "laguerre":
				from numpy.polynomial.laguerre import lagvander
				basis = lagvander(t, config.degree - 1).T

			basis = torch.tensor(basis, dtype=torch.float32)

			self.register_buffer("basis", basis)
			self.register_buffer("basis_T", basis.t().contiguous())
	
			if self.multi_fits=='true':
				config.enc_in = args['feature_node']
				if self.FREQ_DOMAIN == "FITS":
					self.fits_node = FITSModel(configs=config)
				elif self.FREQ_DOMAIN == "FITS_LPF":
					self.fits_node = FITSModel_LPF(configs=config)
				elif self.FREQ_DOMAIN == "FITS_Pai":
					self.fits_node = FITSModel_Pai(configs=config)
				elif self.FREQ_DOMAIN == "iTransformer":
					self.fits_node = iTransformerModel(configs=config)
				elif self.FREQ_DOMAIN == "DLinear":
					self.fits_node = DLinearModel(configs=config)
				elif self.FREQ_DOMAIN == "FreTS":
					self.fits_node = FreTSModel(configs=config)
				elif self.FREQ_DOMAIN == "TimesNet":
					self.fits_node = TimesNetModel(configs=config)
				elif self.FREQ_DOMAIN == "FEDformerModel":
					self.fits_node = FEDformerModel(configs=config)
				elif self.FREQ_DOMAIN == "FITS_Legendre":
					self.fits_node = FITSModel_Legendre(configs=config)
				elif self.FREQ_DOMAIN == "FITS_chebyshev":
					self.fits_node = FITS_chebyshev(configs=config)
				elif self.FREQ_DOMAIN == "FITS_lag":
					self.fits_node = FITS_lag(configs=config)
				elif self.FREQ_DOMAIN == "FITS_hermite":
					self.fits_node = FITS_hermite(configs=config)

				config.enc_in = args['feature_log'] 
				if self.FREQ_DOMAIN == "FITS":
					self.fits_log = FITSModel(configs=config)
				elif self.FREQ_DOMAIN == "FITS_LPF":
					self.fits_log = FITSModel_LPF(configs=config)
				elif self.FREQ_DOMAIN == "FITS_Pai":
					self.fits_log = FITSModel_Pai(configs=config)
				elif self.FREQ_DOMAIN == "iTransformer":
					self.fits_log = iTransformerModel(configs=config)
				elif self.FREQ_DOMAIN == "DLinear":	
					self.fits_log = DLinearModel(configs=config)
				elif self.FREQ_DOMAIN == "FreTS":
					self.fits_log = FreTSModel(configs=config)
				elif self.FREQ_DOMAIN == "TimesNet":
					self.fits_log = TimesNetModel(configs=config)
				elif self.FREQ_DOMAIN == "FEDformerModel":
					self.fits_log = FEDformerModel(configs=config)
				elif self.FREQ_DOMAIN == "FITS_Legendre":
					self.fits_log = FITSModel_Legendre(configs=config)
				elif self.FREQ_DOMAIN == "FITS_chebyshev":
					self.fits_log = FITS_chebyshev(configs=config)
				elif self.FREQ_DOMAIN == "FITS_lag":
					self.fits_log = FITS_lag(configs=config)
				elif self.FREQ_DOMAIN == "FITS_hermite":
					self.fits_log = FITS_hermite(configs=config)

				config.enc_in = args['feature_edge'] 
				if self.FREQ_DOMAIN == "FITS":
					self.fits_edge = FITSModel(configs=config)
				elif self.FREQ_DOMAIN == "FITS_LPF":
					self.fits_edge = FITSModel_LPF(configs=config)
				elif self.FREQ_DOMAIN == "FITS_Pai":
					self.fits_edge = FITSModel_Pai(configs=config)
				elif self.FREQ_DOMAIN == "iTransformer":
					self.fits_edge = iTransformerModel(configs=config)
				elif self.FREQ_DOMAIN == "DLinear":
					self.fits_edge = DLinearModel(configs=config)
				elif self.FREQ_DOMAIN == "FreTS":
					self.fits_edge = FreTSModel(configs=config)
				elif self.FREQ_DOMAIN == "TimesNet":
					self.fits_edge = TimesNetModel(configs=config)
				elif self.FREQ_DOMAIN == "FEDformerModel":
					self.fits_edge = FEDformerModel(configs=config)
				elif self.FREQ_DOMAIN == "FITS_Legendre":
					self.fits_edge = FITSModel_Legendre(configs=config)
				elif self.FREQ_DOMAIN == "FITS_chebyshev":
					self.fits_edge = FITS_chebyshev(configs=config)
				elif self.FREQ_DOMAIN == "FITS_lag":
					self.fits_edge = FITS_lag(configs=config)
				elif self.FREQ_DOMAIN == "FITS_hermite":
					self.fits_edge = FITS_hermite(configs=config)
			else:
				self.linear_attn = LinearAttention(dim=10)
				config.enc_in = 10
				if self.FREQ_DOMAIN == "FITS":
					self.shared_fits = FITSModel(configs=config)
				elif self.FREQ_DOMAIN == "FITS_LPF":
					self.shared_fits = FITSModel_LPF(configs=config)
				elif self.FREQ_DOMAIN == "FITS_Pai":
					self.shared_fits = FITSModel_Pai(configs=config)
				elif self.FREQ_DOMAIN == "iTransformer":
					self.shared_fits = iTransformerModel(configs=config)
				elif self.FREQ_DOMAIN == "DLinear":
					self.shared_fits = DLinearModel(configs=config)
				elif self.FREQ_DOMAIN == "FreTS":	
					self.shared_fits = FreTSModel(configs=config)
				elif self.FREQ_DOMAIN == "TimesNet":
					self.shared_fits = TimesNetModel(configs=config)
				elif self.FREQ_DOMAIN == "FEDformerModel":
					self.shared_fits = FEDformerModel(configs=config)
				elif self.FREQ_DOMAIN == "FITS_Legendre":
					#config.enc_in = config.enc_in * 3
					self.shared_fits = FITSModel_Legendre(configs=config)
				elif self.FREQ_DOMAIN == "FITS_chebyshev":
					self.shared_fits = FITS_chebyshev(configs=config)
				elif self.FREQ_DOMAIN == "FITS_lag":
					self.shared_fits = FITS_lag(configs=config)
				elif self.FREQ_DOMAIN == "FITS_hermite":
					self.shared_fits = FITS_hermite(configs=config)
				self.modality_proj = nn.ModuleDict({
					'node': nn.Linear(args['feature_node'], config.enc_in),
					'log': nn.Linear(args['feature_log'], config.enc_in),
					'edge': nn.Linear(args['feature_edge'], config.enc_in)
				})
				self.modality_proj_out = nn.ModuleDict({
					'node': nn.Linear(config.enc_in,args['feature_node']),
					'log': nn.Linear(config.enc_in,args['feature_log'] ),
					'edge': nn.Linear(config.enc_in,args['feature_edge'])
				})

			self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = adj2adj(self.graph, args['batch_size'], args['window'], args['feature_edge']) # <--- can by modified DynamicTopology (as a parameter instead of being in init)
		elif self.FREQ_DOMAIN == "FITS_LENGDRE_parallel_oth_compoenents":
			class Config: pass
			config = Config()
			config.win_size = args['window']
			config.degree = args.get('basis_degree', 5)  # Number of orthogonal components (K)
			config.basis_type = args.get('basis_type', 'legendre')
			self.degree = config.degree
			self.basis_type = args['basis_type']
			# 1. Generate the Orthogonal Bases (T -> K)
			t_steps = np.linspace(-1, 1, config.win_size)
			if config.basis_type == "legendre":
				from scipy.special import legendre
				basis = np.array([legendre(i)(t_steps) for i in range(config.degree)])
			elif config.basis_type == "chebyshev":
				from numpy.polynomial.chebyshev import chebvander
				basis = chebvander(t_steps, config.degree - 1).T
			else:
				raise ValueError(f"Basis {config.basis_type} not implemented for this parallel mode.")

			# Shapes: basis is [K, T]
			basis = torch.tensor(basis, dtype=torch.float32)
			self.register_buffer("time_basis", basis)  # [K, T]

			# 2. Define the Inner Core Spatial Model (No temporal parameters)
			# This processes a single orthogonal component slice [B, N, F]
			class SpatialReconstructionModel(nn.Module):
				def __init__(self, f_dim):
					super().__init__()
					# Simple two-layer MLP for cross-feature spatial mapping
					self.net = nn.Sequential(
						nn.Linear(f_dim, f_dim * 2),
						nn.GELU(),
						nn.Linear(f_dim * 2, f_dim)
					)
				def forward(self, x):
					return self.net(x)

			if self.multi_fits == 'true':
				self.spatial_node = SpatialReconstructionModel(args['feature_node'])
				self.spatial_log  = SpatialReconstructionModel(args['feature_log'])
				self.spatial_edge = SpatialReconstructionModel(args['feature_edge'])
			else:
				config.enc_in = 10
				self.shared_spatial = SpatialReconstructionModel(config.enc_in)
				self.modality_proj = nn.ModuleDict({
					'node': nn.Linear(args['feature_node'], config.enc_in),
					'log': nn.Linear(args['feature_log'], config.enc_in),
					'edge': nn.Linear(args['feature_edge'], config.enc_in)
				})
				self.modality_proj_out = nn.ModuleDict({
					'node': nn.Linear(config.enc_in, args['feature_node']),
					'log': nn.Linear(config.enc_in, args['feature_log']),
					'edge': nn.Linear(config.enc_in, args['feature_edge'])
				})

			self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = adj2adj(self.graph, args['batch_size'], args['window'], args['feature_edge'])
		elif self.FREQ_DOMAIN == "FourierGNN":
			self.adj_proj = nn.Linear(args['feature_edge'], 1)               # Edge → scalar weight
			class Config: pass
			config = Config()
			config.DSR = 1  # Downsampling rate
			config.pre_length = args['window']  # Window size
			config.embed_size = args['feature_node']
			config.feature_size = args['feature_node']
			config.seq_length = args['window'] // config.DSR
			config.hidden_size = 10
			config.enc_in = 10
			if self.multi_fits == 'false':
				self.shared_fgn = FGN(pre_length=config.pre_length, embed_size=config.embed_size, feature_size=config.feature_size, seq_length=config.seq_length, hidden_size=config.hidden_size)
				#self.modality_proj = nn.ModuleDict({
				#		'node': nn.Linear(args['feature_node'], config.enc_in),
				#		'log': nn.Linear(args['feature_log'], config.enc_in),
				#		'edge': nn.Linear(args['feature_edge'], config.enc_in)
				#})
				self.modality_proj_out = nn.ModuleDict({
					'node': nn.Linear(config.enc_in,args['feature_node']),
					'log': nn.Linear(config.enc_in,args['feature_log'] ),
    				'edge': nn.Linear(2*config.enc_in, args['feature_edge'])  # <-- doubled input dim
				})
			self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = adj2adj(self.graph, args['batch_size'], args['window'], args['feature_edge']) # <--- can by modified DynamicTopology (as a parameter instead of being in init)
		elif self.FREQ_DOMAIN == "GPT2":
			class Config: pass
			config = Config()

			# Core parameters from LLM4MST paper
			config.ln = True
			config.task_name = "anomaly_detection"
			config.pred_len = 10
			config.seq_len = 10
			config.patch_size = 1
			config.stride = 1
			config.d_ff = 768
			config.use_gpu = True
			config.mlp = 0
			config.gpt_layers = 6

			# Embedding parameters
			config.enc_in = 768
			config.d_model = 768
			config.embed = "timeF"
			config.freq = "h"
			config.dropout = 0.1

			# Task-specific
			config.c_out = 1  # For forecasting/anomaly detection
			config.num_class = 2  # For classification
			# Hardware
			config.use_gpu = True
			if self.multi_fits == 'false':
				self.shared_fits = GPT2Model(configs=config)
				# as it would be padded by the model, we can use nn.Identity() to be the same as FITS and FGN
				self.modality_proj = nn.ModuleDict({
						'node': nn.Linear(args['feature_node'], config.enc_in),
						'log': nn.Linear(args['feature_log'], config.enc_in),
						'edge': nn.Linear(args['feature_edge'], config.enc_in)
				})
				self.modality_proj_out = nn.ModuleDict({
					'node': nn.Linear(config.enc_in,args['feature_node']),
					'log': nn.Linear(config.enc_in,args['feature_log'] ),
					'edge': nn.Linear(config.enc_in,args['feature_edge'])
				})
			self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = adj2adj(self.graph, args['batch_size'], args['window'], args['feature_edge']) # <--- can by modified DynamicTopology (as a parameter instead of being in init)

		elif self.FREQ_DOMAIN == "Eadro":
			event_num = args['log_len']
			metric_num = args['raw_node']
			node_num = args['raw_edge']
			self.Eadro_Model = MainModel(event_num, metric_num, node_num)
		
		elif self.FREQ_DOMAIN == "AnoFusion":
			self.AnoFusion = AnoFusion(
				num_services=self.num_classes,
				#edge_types=self.graph.shape[0],
				window_size=args['window'],
				metric_dim=args['raw_node'],
				log_dim=args['log_len'],
				trace_dim=args['raw_edge'],
				out_dim=args['raw_node'] + args['raw_edge'] + args['log_len']
			)
			
		elif self.FREQ_DOMAIN == "Art":
			self.Art_Model = Art_Model(
				adj=self.graph,
				raw_metric=args['raw_node'],
				raw_logs=args['log_len'],
				raw_traces=args['raw_edge'],
				feature_metric=args['feature_node'],
				feature_logs=args['feature_log'],
				feature_traces=args['feature_edge']
			)
		
		elif self.FREQ_DOMAIN == "Hades":
			event_num = args['log_len']
			metric_num = args['raw_node']
			self.Hades_model = Hades_Model(
				raw_metric=args['raw_node'],
				raw_logs=args['log_len'],
				feature_metric=args['feature_node'],
				feature_logs=args['feature_log'],
				device = 'cuda',
				#TODO to be in the forward only
			)
			
		self.node_emb = Embed(args['raw_node'], args['feature_node'], dim=4)
		self.log_emb = Embed(args['log_len'], args['feature_log'], dim=4)
		self.egde_emb = Embed(args['raw_edge'], args['feature_edge'], dim=5)

		self.trace2pod = torch.nn.functional.one_hot(adj[0], num_classes=self.graph.shape[0]) \
			+ torch.nn.functional.one_hot(adj[1], num_classes=self.graph.shape[0])
		self.trace2pod = self.trace2pod / 2

		self.dense_node = nn.Linear(args['feature_node'], args['raw_node'])
		self.dense_log = nn.Linear(args['feature_log'], args['log_len'])
		self.dense_edge = nn.Linear(args['feature_edge'], args['raw_edge'])

		self.show = nn.Sequential(nn.Linear(args['raw_node'] + args['raw_edge'] + args['log_len'], 128),
							nn.LeakyReLU(inplace=True),
							nn.Linear(128, 2))

		#edge_exists_mask = (self.node_efea.sum(dim=-1) != 0)  # [N, N] boolean mask
		#edge_index = torch.nonzero(edge_exists_mask, as_tuple=False)  # [num_edges, 2]
		#self.register_buffer("edge_index", edge_index)
#
		#self.num_edges = edge_index.shape[0]

	def upsample_time_dim(self, tensor_4d: torch.Tensor, target_time: int) -> torch.Tensor:
		B, T_old, N, F_ = tensor_4d.shape
		tensor_3d = tensor_4d.permute(0, 2, 3, 1).reshape(B, N * F_, T_old)  # [B, C, T_old]
		tensor_upsampled = torch.nn.functional.interpolate(tensor_3d, size=target_time, mode='linear', align_corners=False)
		tensor_upsampled = tensor_upsampled.reshape(B, N, F_, target_time).permute(0, 3, 1, 2)  # [B, T_new, N, F]
		return tensor_upsampled

	# =====================================================
	# Basis Construction (Fourier), using real sines and cosines
	# =====================================================
	def _build_fourier_basis(self, t, degree,device):
		"""
		Build a Fourier basis with exactly `degree` vectors, adding 1 if degree is odd.
		"""
		# Ensure even degree for consistent sin/cos pairing
		if degree % 2 != 0:
			degree += 1  # add 1 if odd

		# Convert t to tensor if it's not already
		if not isinstance(t, torch.Tensor):
			t = torch.tensor(t, dtype=torch.float32, device=device)
		with torch.no_grad():
			# Start with constant term
			basis = [torch.ones_like(t.cpu())]
			for k in range(1, degree // 2):
				basis.append(np.sin(2 * np.pi * k * t.cpu()))
				basis.append(np.cos(2 * np.pi * k * t.cpu()))

		return torch.tensor(np.array(basis))

	def forward(self, x, evaluate=False):
		if self.FREQ_DOMAIN == "encoder_decoder":
			x_node, d_node = self.node_emb(x['data_node'])#torch.Size([50, 10, 5, 3])
			x_edge, d_edge = self.egde_emb(x['data_edge'])#torch.Size([50, 10, 5, 5, 7])
			x_log, d_log = self.log_emb(x['data_log'])#torch.Size([50, 10, 5, 256])

			z_node, z_edge, z_log = self.encoder(x_node, x_edge, x_log)
			node, edge, log = self.decoder(d_node, d_edge, d_log, z_node, z_edge, z_log)
			
			device = x['data_edge'].device
			self.graph = self.graph.to(device)
			l_edge = torch.masked_select(x['data_edge'], self.graph.unsqueeze(-1).repeat(1, 1, x['data_edge'].shape[-1]).bool()) \
				.reshape(x['data_edge'].shape[0], x['data_edge'].shape[1], -1, x['data_edge'].shape[-1])

			rec_node = torch.square(self.dense_node(node) - x['data_node'])
			rec_edge1 = torch.square(self.dense_edge(edge) - l_edge)
			rec_log = torch.square(self.dense_log(log) - x['data_log'])
			rec_edge1 = rec_edge1.to(device)
			self.trace2pod = self.trace2pod.to(device)
			rec_edge = torch.matmul(rec_edge1.permute(
				0, 1, 3, 2), self.trace2pod.float()).permute(0, 1, 3, 2)
			rec = torch.concat([rec_node, rec_log, rec_edge], dim=-1)
		elif self.FREQ_DOMAIN in ["FITS_Pai","FITS_LPF","FITS","GPT2","iTransformer","DLinear","FreTS","TimesNet", "FEDformerModel","FITS_Legendre","FITS_chebyshev","FITS_lag","FITS_hermite"]:
			B, T, _,_ = x['data_node'].shape
			# get edge mask
			edge_exists_mask = (self.node_efea.sum(dim=-1) != 0)  # [N, N] boolean mask
			edge_exists_mask_batch = edge_exists_mask.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)  # [B, T, N, N]

			# Get embeddings
			x_node_metric_fits, _ = self.node_emb(x['data_node'])  # Shape: [B, T, N, F]
			x_node_logs_fits, _ = self.log_emb(x['data_log'])  # Shape: [B, T, L, F]
			x_edge_fits, _ = self.egde_emb(x['data_edge'])  # Shape: [B, T, E, F]

			# Permute to FITS input shape: [B*N, T, F]
			_, _, N, F_METRIC = x_node_metric_fits.shape
			x_node_metric_fits_input = x_node_metric_fits.permute(0, 2, 1, 3).reshape(B*N, T, F_METRIC) # [B*N, T, F]
			_, _, N, F_LOG = x_node_logs_fits.shape
			x_node_logs_fits_input = x_node_logs_fits.permute(0, 2, 1, 3).reshape(B*N, T, F_LOG) # [B*N, T, F]
			_, _, _, _, E = x_edge_fits.shape
			x_edge_flat = x_edge_fits.reshape(B, T, N*N, E)  # [B, T, N*N, E]
			edge_mask_flat = edge_exists_mask.view(-1)  # [N*N]
			edge_mask_flat = edge_mask_flat.to(x_edge_flat.device)
			x_edge_masked = x_edge_flat[:, :, edge_mask_flat, :]  # select only existing edges
			x_edge_fits_input = x_edge_masked.permute(0, 2, 1, 3).reshape(B * edge_mask_flat.sum().item(), T, E)  # [B*num_edges, T, E]

			# -------------------------------------------------
			# Node streams → FITS input
			## -------------------------------------------------
			#B, T, N, F_METRIC = x_node_metric_fits.shape
			#_, _, _, F_LOG = x_node_logs_fits.shape
#
			#x_node_metric_fits_input = (
			#	x_node_metric_fits.permute(0, 2, 1, 3)
			#	.reshape(B * N, T, F_METRIC)
			#)
#
			#x_node_logs_fits_input = (
			#	x_node_logs_fits.permute(0, 2, 1, 3)
			#	.reshape(B * N, T, F_LOG)
			#)
#
			## -------------------------------------------------
			## Edge stream → NO MASKING, NO NxN MATERIALIZATION
			## -------------------------------------------------
			#i, j = self.edge_index[:, 0], self.edge_index[:, 1]
			#E = x_edge_fits.shape[-1]
#
			#x_edge_fits_input = (
			#	x_edge_fits[:, :, i, j, :]   # [B, T, num_edges, E]
			#	.permute(0, 2, 1, 3)         # [B, num_edges, T, E]
			#)


			if self.multi_fits == 'false':
				
				#-----------> old version with running a shared FITS per modality
				# ---- projections (unchanged) ----
				x_node_proj = self.modality_proj['node'](x_node_metric_fits_input)
				x_log_proj  = self.modality_proj['log'](x_node_logs_fits_input)

				# ---- shared temporal encoder ----
				h_node = self.shared_fits(x_node_proj)[0]   # [B*N, T, D]
				h_log  = self.shared_fits(x_log_proj)[0]    # [B*N, T, D]

				if self.modules_attn == 'linear_attn':
					# ============================================================
					# Linear Attention fusion (node + log)
					# ============================================================

					# Collapse time for modality interaction
					h_node_t = h_node.mean(dim=1)  # [B*N, D]
					h_log_t  = h_log.mean(dim=1)   # [B*N, D]

					# Stack modalities as tokens
					H = torch.stack([h_node_t, h_log_t], dim=1)  # [B*N, 2, D]

					# Linear Attention
					H = self.linear_attn(H)  # [B*N, 2, D]

					# Split back
					h_node_fused = H[:, 0]   # [B*N, D]
					h_log_fused  = H[:, 1]   # [B*N, D]

					# Restore time dimension
					h_node = h_node_fused.unsqueeze(1).expand(-1, h_node.shape[1], -1)
					h_log  = h_log_fused.unsqueeze(1).expand(-1, h_log.shape[1], -1)

				# ---- output projections (unchanged interfaces) ----
				rec_node_metric_fits = self.modality_proj_out['node'](h_node)
				rec_node_logs_fits   = self.modality_proj_out['log'](h_log)

				# ---- edge stream untouched ----
				x_edge_proj = self.modality_proj['edge'](x_edge_fits_input)
				h_edge = self.shared_fits(x_edge_proj)[0]
				rec_edge_fits = self.modality_proj_out['edge'](h_edge)




			else:# only implemened for FITS 
				rec_node_metric_fits, _ = self.fits_node(x_node_metric_fits_input)  # [B*N, T', F]
				rec_node_logs_fits, _   = self.fits_log(x_node_logs_fits_input)  # [B*N, T', F]
				rec_edge_fits, _ = self.fits_edge(x_edge_fits_input)  # [B*N*N, T', E]
			# Reshape back to original shape
			pred_metric_node = rec_node_metric_fits.reshape(B, N, -1, F_METRIC).permute(0, 2, 1, 3)  # [B, T, N, F]
			pred_log_node    = rec_node_logs_fits.reshape(B, N, -1, F_LOG).permute(0, 2, 1, 3)  # [B, T, N, F]
			# Keep edge predictions in masked form: [B, T, num_edges, E]
			pred_edge_masked = rec_edge_fits.reshape(B, edge_mask_flat.sum().item(), -1, E).permute(0, 2, 1, 3)  # [B, T, num_edges, E]
#
			# Extract ground truth edges using mask: [B, T, num_edges, E]
			mask = edge_exists_mask_batch.to(x['data_edge'].device).unsqueeze(-1)
			l_edge = torch.masked_select(x['data_edge'], mask).reshape(B, T, edge_mask_flat.sum().item(), -1)
			l_edge = torch.masked_select(x['data_edge'], edge_exists_mask_batch.unsqueeze(-1)).reshape(B, T, edge_mask_flat.sum().item(), -1)
			# -------------------------------------------------
			# Node reconstruction (unchanged structure, cleaned)
			# -------------------------------------------------
			#pred_metric_node = rec_node_metric_fits.reshape(B, N, T, F_METRIC).permute(0, 2, 1, 3)
			#pred_log_node    = rec_node_logs_fits.reshape(B, N, T, F_LOG).permute(0, 2, 1, 3)

			# -------------------------------------------------
			# Edge reconstruction (NO MASKING)
			# -------------------------------------------------
			#pred_edge_masked = rec_edge_fits.permute(0, 2, 1, 3)   # [B, T, num_edges, E]
			#i, j = self.edge_index[:, 0], self.edge_index[:, 1]
			#l_edge = x['data_edge'][:, :, i, j, :]   # [B, T, num_edges, E]
			# Square Loss
			if self.req_loss_approach == "Normal-Recreation":
				rec_node_metric_fits = torch.square(self.dense_node(pred_metric_node) - x['data_node'])  # Calculate squared loss on nodes (full) [B, T, N, F]
				rec_node_log_fits 	 = torch.square(self.dense_log(pred_log_node) - x['data_log'])  # Calculate squared loss on nodes (full) [B, T, N, F]
				rec_edge1 = torch.square(self.dense_edge(pred_edge_masked) - l_edge)  #Calculate squared loss on edges (masked only) [B, T, num_edges, E]
			elif self.req_loss_approach == "FreDF-style":
				# Calculate differences (time domain residuals)
				diff_node_metric = self.dense_node(pred_metric_node) - x['data_node']         # [B, T, N, F]
				diff_node_log = self.dense_log(pred_log_node) - x['data_log']                 # [B, T, N, F]
				diff_edge = self.dense_edge(pred_edge_masked) - l_edge                        # [B, T, E, F]

				# Time domain losses (MSE)
				loss_time_node_metric = torch.square(diff_node_metric)
				loss_time_log = torch.square(diff_node_log)
				loss_time_edge = torch.square(diff_edge)

				# Frequency domain losses (magnitude of FFT residuals)
				loss_freq_node_metric = torch.fft.rfft(diff_node_metric, dim=1).abs()  # [B, T', N, F]
				loss_freq_log = torch.fft.rfft(diff_node_log, dim=1).abs()
				loss_freq_edge = torch.fft.rfft(diff_edge, dim=1).abs()

				# Upsample frequency losses to match time domain shape
				loss_freq_node_metric = self.upsample_time_dim(loss_freq_node_metric, target_time=diff_node_metric.shape[1])
				loss_freq_log = self.upsample_time_dim(loss_freq_log, target_time=diff_node_log.shape[1])
				loss_freq_edge = self.upsample_time_dim(loss_freq_edge, target_time=diff_edge.shape[1])

				# Weighted sum of time and frequency losses
				rec_node_metric_fits = self.rec_lambda * loss_time_node_metric + self.auxi_lambda * loss_freq_node_metric
				rec_node_log_fits = self.rec_lambda * loss_time_log + self.auxi_lambda * loss_freq_log
				rec_edge1 = self.rec_lambda * loss_time_edge + self.auxi_lambda * loss_freq_edge
			elif self.req_loss_approach  == "Legendre-style":
				# Helper to merge node and feature dims for Legendre encoding
				def merge_nf(x):
					B, T, N, F = x.shape
					return x.reshape(B, T, N * F)

				# Helper to expand Legendre losses to match [B, T, N, F]
				def expand_leg_loss(loss_leg, ref_tensor):
					# loss_leg: [B, N] or [B, E] — dims after mean over degree
					# ref_tensor: [B, T, N, F] or [B, T, E, F]
					expanded = loss_leg.unsqueeze(1).expand(B, ref_tensor.shape[1], -1)  # [B, T, N or E]
					expanded = expanded.unsqueeze(-1).expand(-1, -1, -1, ref_tensor.shape[-1])  # [B, T, N or E, F]
					return expanded

				# --- Time domain residuals ---
				diff_node_metric = self.dense_node(pred_metric_node) - x['data_node']         # [B, T, N, F]
				diff_node_log = self.dense_log(pred_log_node) - x['data_log']                 # [B, T, N, F]
				diff_edge = self.dense_edge(pred_edge_masked) - l_edge                        # [B, T, E, F]

				# --- Time domain losses (MSE) ---
				loss_time_node_metric = torch.square(diff_node_metric)                        # [B, T, N, F]
				loss_time_log = torch.square(diff_node_log)                                   # [B, T, N, F]
				loss_time_edge = torch.square(diff_edge)                                      # [B, T, E, F]

				# Merge node and feature dims for Legendre encoding: [B, T, N*F]
				pred_metric_merged = merge_nf(self.dense_node(pred_metric_node))
				true_metric_merged = merge_nf(x['data_node'])

				pred_log_merged = merge_nf(self.dense_log(pred_log_node))
				true_log_merged = merge_nf(x['data_log'])

				pred_edge_merged = merge_nf(self.dense_edge(pred_edge_masked))
				true_edge_merged = merge_nf(l_edge)

				if self.basis_type == "legendre":
					# Legendre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_Legendre_operations.legendre_encode(pred_metric_merged, degree=5)  # [B, N*F, D]
					true_metric_leg = FITS_Legendre_operations.legendre_encode(true_metric_merged, degree=5)  # [B, N*F, D]

					pred_log_leg = FITS_Legendre_operations.legendre_encode(pred_log_merged, degree=5)
					true_log_leg = FITS_Legendre_operations.legendre_encode(true_log_merged, degree=5)
					
					pred_edge_leg = FITS_Legendre_operations.legendre_encode(pred_edge_merged, degree=5)
					true_edge_leg = FITS_Legendre_operations.legendre_encode(true_edge_merged, degree=5)
				elif self.basis_type == "chebyshev":
					# Legendre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_chebyshev_operations.chebyshev_encode(pred_metric_merged, degree=5)  # [B, N*F, D]
					true_metric_leg = FITS_chebyshev_operations.chebyshev_encode(true_metric_merged, degree=5)  # [B, N*F, D]

					pred_log_leg = FITS_chebyshev_operations.chebyshev_encode(pred_log_merged, degree=5)
					true_log_leg = FITS_chebyshev_operations.chebyshev_encode(true_log_merged, degree=5)
					pred_edge_leg = FITS_chebyshev_operations.chebyshev_encode(pred_edge_merged, degree=5)
					true_edge_leg = FITS_chebyshev_operations.chebyshev_encode(true_edge_merged, degree=5)
				elif self.basis_type == "laguerre":
					# Lagurre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_lag_operations.laguerre_encode(pred_metric_merged, degree=5)  # [B, N*F, D]
					true_metric_leg = FITS_lag_operations.laguerre_encode(true_metric_merged, degree=5)  # [B, N*F, D]

					pred_log_leg = FITS_lag_operations.laguerre_encode(pred_log_merged, degree=5)
					true_log_leg = FITS_lag_operations.laguerre_encode(true_log_merged, degree=5)
					pred_edge_leg = FITS_lag_operations.laguerre_encode(pred_edge_merged, degree=5)
					true_edge_leg = FITS_lag_operations.laguerre_encode(true_edge_merged, degree=5)
				elif self.basis_type == "hermite":
					# Lagurre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_hermite_operations.hermite_encode(pred_metric_merged, degree=5)  # [B, N*F, D]
					true_metric_leg = FITS_hermite_operations.hermite_encode(true_metric_merged, degree=5)  # [B, N*F, D]

					pred_log_leg = FITS_hermite_operations.hermite_encode(pred_log_merged, degree=5)
					true_log_leg = FITS_hermite_operations.hermite_encode(true_log_merged, degree=5)
					pred_edge_leg = FITS_hermite_operations.hermite_encode(pred_edge_merged, degree=5)
					true_edge_leg = FITS_hermite_operations.hermite_encode(true_edge_merged, degree=5)
				elif self.basis_type == "fourier":
					# Fourier encode: outputs [B, C, degree]
					pred_metric_leg = self._build_fourier_basis(pred_metric_merged, degree=5,device=self.graph.device)  # [B, N*F, D]
					true_metric_leg = self._build_fourier_basis(true_metric_merged, degree=5,device=self.graph.device)  # [B, N*F, D]

					pred_log_leg = self._build_fourier_basis(pred_log_merged, degree=5,device=self.graph.device)
					true_log_leg = self._build_fourier_basis(true_log_merged, degree=5,device=self.graph.device)
					pred_edge_leg = self._build_fourier_basis(pred_edge_merged, degree=5,device=self.graph.device)
					true_edge_leg = self._build_fourier_basis(true_edge_merged, degree=5,device=self.graph.device)

				#				# ground truth projection
				#def project_to_basis(x):
				#	return torch.matmul(x.transpose(1, 2).contiguous(), self.basis_T)
				#pred_metric_leg = project_to_basis(pred_metric_merged)
				#true_metric_leg = project_to_basis(true_metric_merged)
				#pred_log_leg = project_to_basis(pred_log_merged)
				#true_log_leg = project_to_basis(true_log_merged)
				#pred_edge_leg = project_to_basis(pred_edge_merged)
				#true_edge_leg = project_to_basis(true_edge_merged)

				# Compute MSE in Legendre domain, mean over degree dim (last)
				loss_leg_metric = torch.square(pred_metric_leg - true_metric_leg).mean(dim=-1)   # [B, N*F]
				loss_leg_log = torch.square(pred_log_leg - true_log_leg).mean(dim=-1)            # [B, N*F]
				loss_leg_edge = torch.square(pred_edge_leg - true_edge_leg).mean(dim=-1)         # [B, E*F]

				# Reshape back to [B, N, F] or [B, E, F]
				B, T, N, F = diff_node_metric.shape
				B, T, N, FLOG = diff_node_log.shape
				_, _, E, FEDGE = diff_edge.shape

				loss_leg_metric = loss_leg_metric.reshape(B, N, F)  # [B, N, F]
				loss_leg_log = loss_leg_log.reshape(B, N, FLOG)
				loss_leg_edge = loss_leg_edge.reshape(B, E, FEDGE)

				# Expand to match time dim [B, T, N, F] or [B, T, E, F]
				loss_leg_metric = loss_leg_metric.unsqueeze(1).expand(B, T, N, F)
				loss_leg_log = loss_leg_log.unsqueeze(1).expand(B, T, N, FLOG)
				loss_leg_edge = loss_leg_edge.unsqueeze(1).expand(B, T, E, FEDGE)

				# --- Final combined losses ---
				rec_node_metric_fits = self.rec_lambda * loss_time_node_metric + self.auxi_lambda * loss_leg_metric
				rec_node_log_fits = self.rec_lambda * loss_time_log + self.auxi_lambda * loss_leg_log
				rec_edge1 = self.rec_lambda * loss_time_edge + self.auxi_lambda * loss_leg_edge

			rec_edge = torch.matmul(rec_edge1.permute(
				0, 1, 3, 2), self.trace2pod.float().to(rec_edge1.device)).permute(0, 1, 3, 2)
			#rec_edge = torch.matmul(rec_edge1.permute(
			#	0, 1, 3, 2), self.trace2pod.float()).permute(0, 1, 3, 2)
			rec = torch.concat([rec_node_metric_fits,rec_node_log_fits, rec_edge], dim=-1)
		elif self.FREQ_DOMAIN == "FITS_LENGDRE_parallel_oth_compoenents":
			B, T, _,_ = x['data_node'].shape
			# get edge mask
			edge_exists_mask = (self.node_efea.sum(dim=-1) != 0)  # [N, N] boolean mask
			edge_exists_mask_batch = edge_exists_mask.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)  # [B, T, N, N]

			# Get embeddings
			x_node_metric_fits, _ = self.node_emb(x['data_node'])  # Shape: [B, T, N, F]
			x_node_logs_fits, _ = self.log_emb(x['data_log'])  # Shape: [B, T, L, F]
			x_edge_fits, _ = self.egde_emb(x['data_edge'])  # Shape: [B, T, E, F]
			B, T, N, F_METRIC = x_node_metric_fits.shape
			_, _, _, F_LOG = x_node_logs_fits.shape
			_, _, _, _, E = x_edge_fits.shape

			# Isolate masked active edges
			x_edge_flat = x_edge_fits.reshape(B, T, N*N, E)
			edge_mask_flat = edge_exists_mask.view(-1).to(x_edge_flat.device)
			x_edge_masked = x_edge_flat[:, :, edge_mask_flat, :]  # [B, T, num_edges, E]
			num_edges = edge_mask_flat.sum().item()

			# --- STEP 1: Transform Time (T) to Orthogonal Components (K) ---
			# self.time_basis shape: [K, T]. We want to project along T.
			# Node mapping: [B, T, N, F] -> [B, K, N, F]
			x_node_orth = torch.einsum('btnd,kt->bknd', x_node_metric_fits, self.time_basis)
			x_log_orth  = torch.einsum('btnd,kt->bknd', x_node_logs_fits, self.time_basis)
			x_edge_orth = torch.einsum('bthe,kt->bkhe', x_edge_masked, self.time_basis)

			# --- STEP 2: Optional Shared Reduction Projections ---
			if self.multi_fits == 'false':
				x_node_orth = self.modality_proj['node'](x_node_orth) # [B, K, N, 10]
				x_log_orth  = self.modality_proj['log'](x_log_orth)   # [B, K, N, 10]
				x_edge_orth = self.modality_proj['edge'](x_edge_orth) # [B, K, num_edges, 10]

			rec_node_list, rec_log_list, rec_edge_list = [], [], []

			# --- STEP 3: THE COMPONENT LOOP (Over K components) ---
			for k in range(self.degree):
				# Slice single independent component snapshot: [B, N, F]
				node_k = x_node_orth[:, k, :, :]
				log_k  = x_log_orth[:, k, :, :]
				edge_k = x_edge_orth[:, k, :, :]

				if self.multi_fits == 'true':
					rec_node_k = self.spatial_node(node_k)
					rec_log_k  = self.spatial_log(log_k)
					rec_edge_k = self.spatial_edge(edge_k)
				else:
					rec_node_k = self.shared_spatial(node_k)
					rec_log_k  = self.shared_spatial(log_k)
					rec_edge_k = self.shared_spatial(edge_k)

				rec_node_list.append(rec_node_k)
				rec_log_list.append(rec_log_k)
				rec_edge_list.append(rec_edge_k)

			# Stack along the component axis back to: [B, K, Spatial, Feature]
			rec_node_orth = torch.stack(rec_node_list, dim=1)
			rec_log_orth  = torch.stack(rec_log_list, dim=1)
			rec_edge_orth = torch.stack(rec_edge_list, dim=1)

			# --- STEP 4: Optional Shared Output Expansions ---
			if self.multi_fits == 'false':
				rec_node_orth = self.modality_proj_out['node'](rec_node_orth)
				rec_log_orth  = self.modality_proj_out['log'](rec_log_orth)
				rec_edge_orth = self.modality_proj_out['edge'](rec_edge_orth)

			# --- STEP 5: Inverse Transformation back to Full Time (K -> T) ---
			pinv_basis = torch.pinverse(self.time_basis) # [T, K]

			# Reconstruct to [B, T, Spatial, Feature]
			pred_metric_node = torch.einsum('bknd,tk->btnd', rec_node_orth, pinv_basis)
			pred_log_node    = torch.einsum('bknd,tk->btnd', rec_log_orth, pinv_basis)
			pred_edge_masked = torch.einsum('bkhe,tk->bthe', rec_edge_orth, pinv_basis)

			# --- STEP 6: Compute Residual Reconstruction Loss ---
			# Extract ground-truth edges using your existing masking strategy
			mask = edge_exists_mask_batch.to(x['data_edge'].device).unsqueeze(-1)
			l_edge = torch.masked_select(x['data_edge'], mask).reshape(B, T, edge_mask_flat.sum().item(), -1)

			if self.req_loss_approach == "Normal-Recreation":
				# Compute square loss relative to raw data via your dense layers
				rec_node_metric_fits = torch.square(self.dense_node(pred_metric_node) - x['data_node'])
				rec_node_log_fits    = torch.square(self.dense_log(pred_log_node) - x['data_log'])
				rec_edge1            = torch.square(self.dense_edge(pred_edge_masked) - l_edge)
			elif self.req_loss_approach  == "Legendre-style":
				# Helper to merge node and feature dims for Legendre encoding
				def merge_nf(x):
					B, T, N, F = x.shape
					return x.reshape(B, T, N * F)

				# Helper to expand Legendre losses to match [B, T, N, F]
				def expand_leg_loss(loss_leg, ref_tensor):
					# loss_leg: [B, N] or [B, E] — dims after mean over degree
					# ref_tensor: [B, T, N, F] or [B, T, E, F]
					expanded = loss_leg.unsqueeze(1).expand(B, ref_tensor.shape[1], -1)  # [B, T, N or E]
					expanded = expanded.unsqueeze(-1).expand(-1, -1, -1, ref_tensor.shape[-1])  # [B, T, N or E, F]
					return expanded

				# --- Time domain residuals ---
				diff_node_metric = self.dense_node(pred_metric_node) - x['data_node']         # [B, T, N, F]
				diff_node_log = self.dense_log(pred_log_node) - x['data_log']                 # [B, T, N, F]
				diff_edge = self.dense_edge(pred_edge_masked) - l_edge                        # [B, T, E, F]

				# --- Time domain losses (MSE) ---
				loss_time_node_metric = torch.square(diff_node_metric)                        # [B, T, N, F]
				loss_time_log = torch.square(diff_node_log)                                   # [B, T, N, F]
				loss_time_edge = torch.square(diff_edge)                                      # [B, T, E, F]

				# Merge node and feature dims for Legendre encoding: [B, T, N*F]
				pred_metric_merged = merge_nf(self.dense_node(pred_metric_node))
				true_metric_merged = merge_nf(x['data_node'])

				pred_log_merged = merge_nf(self.dense_log(pred_log_node))
				true_log_merged = merge_nf(x['data_log'])

				pred_edge_merged = merge_nf(self.dense_edge(pred_edge_masked))
				true_edge_merged = merge_nf(l_edge)

				if self.basis_type == "legendre":
					# Legendre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_Legendre_operations.legendre_encode(pred_metric_merged, degree=5)  # [B, N*F, D]
					true_metric_leg = FITS_Legendre_operations.legendre_encode(true_metric_merged, degree=5)  # [B, N*F, D]

					pred_log_leg = FITS_Legendre_operations.legendre_encode(pred_log_merged, degree=5)
					true_log_leg = FITS_Legendre_operations.legendre_encode(true_log_merged, degree=5)
					
					pred_edge_leg = FITS_Legendre_operations.legendre_encode(pred_edge_merged, degree=5)
					true_edge_leg = FITS_Legendre_operations.legendre_encode(true_edge_merged, degree=5)
				elif self.basis_type == "chebyshev":
					# Legendre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_chebyshev_operations.chebyshev_encode(pred_metric_merged, degree=5)  # [B, N*F, D]
					true_metric_leg = FITS_chebyshev_operations.chebyshev_encode(true_metric_merged, degree=5)  # [B, N*F, D]

					pred_log_leg = FITS_chebyshev_operations.chebyshev_encode(pred_log_merged, degree=5)
					true_log_leg = FITS_chebyshev_operations.chebyshev_encode(true_log_merged, degree=5)
					pred_edge_leg = FITS_chebyshev_operations.chebyshev_encode(pred_edge_merged, degree=5)
					true_edge_leg = FITS_chebyshev_operations.chebyshev_encode(true_edge_merged, degree=5)
				elif self.basis_type == "laguerre":
					# Lagurre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_lag_operations.laguerre_encode(pred_metric_merged, degree=5)  # [B, N*F, D]
					true_metric_leg = FITS_lag_operations.laguerre_encode(true_metric_merged, degree=5)  # [B, N*F, D]

					pred_log_leg = FITS_lag_operations.laguerre_encode(pred_log_merged, degree=5)
					true_log_leg = FITS_lag_operations.laguerre_encode(true_log_merged, degree=5)
					pred_edge_leg = FITS_lag_operations.laguerre_encode(pred_edge_merged, degree=5)
					true_edge_leg = FITS_lag_operations.laguerre_encode(true_edge_merged, degree=5)
				elif self.basis_type == "hermite":
					# Lagurre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_hermite_operations.hermite_encode(pred_metric_merged, degree=5)  # [B, N*F, D]
					true_metric_leg = FITS_hermite_operations.hermite_encode(true_metric_merged, degree=5)  # [B, N*F, D]

					pred_log_leg = FITS_hermite_operations.hermite_encode(pred_log_merged, degree=5)
					true_log_leg = FITS_hermite_operations.hermite_encode(true_log_merged, degree=5)
					pred_edge_leg = FITS_hermite_operations.hermite_encode(pred_edge_merged, degree=5)
					true_edge_leg = FITS_hermite_operations.hermite_encode(true_edge_merged, degree=5)
				elif self.basis_type == "fourier":
					# Fourier encode: outputs [B, C, degree]
					pred_metric_leg = self._build_fourier_basis(pred_metric_merged, degree=5,device=self.graph.device)  # [B, N*F, D]
					true_metric_leg = self._build_fourier_basis(true_metric_merged, degree=5,device=self.graph.device)  # [B, N*F, D]

					pred_log_leg = self._build_fourier_basis(pred_log_merged, degree=5,device=self.graph.device)
					true_log_leg = self._build_fourier_basis(true_log_merged, degree=5,device=self.graph.device)
					pred_edge_leg = self._build_fourier_basis(pred_edge_merged, degree=5,device=self.graph.device)
					true_edge_leg = self._build_fourier_basis(true_edge_merged, degree=5,device=self.graph.device)

				#				# ground truth projection
				#def project_to_basis(x):
				#	return torch.matmul(x.transpose(1, 2).contiguous(), self.basis_T)
				#pred_metric_leg = project_to_basis(pred_metric_merged)
				#true_metric_leg = project_to_basis(true_metric_merged)
				#pred_log_leg = project_to_basis(pred_log_merged)
				#true_log_leg = project_to_basis(true_log_merged)
				#pred_edge_leg = project_to_basis(pred_edge_merged)
				#true_edge_leg = project_to_basis(true_edge_merged)

				# Compute MSE in Legendre domain, mean over degree dim (last)
				loss_leg_metric = torch.square(pred_metric_leg - true_metric_leg).mean(dim=-1)   # [B, N*F]
				loss_leg_log = torch.square(pred_log_leg - true_log_leg).mean(dim=-1)            # [B, N*F]
				loss_leg_edge = torch.square(pred_edge_leg - true_edge_leg).mean(dim=-1)         # [B, E*F]

				# Reshape back to [B, N, F] or [B, E, F]
				B, T, N, F = diff_node_metric.shape
				B, T, N, FLOG = diff_node_log.shape
				_, _, E, FEDGE = diff_edge.shape

				loss_leg_metric = loss_leg_metric.reshape(B, N, F)  # [B, N, F]
				loss_leg_log = loss_leg_log.reshape(B, N, FLOG)
				loss_leg_edge = loss_leg_edge.reshape(B, E, FEDGE)

				# Expand to match time dim [B, T, N, F] or [B, T, E, F]
				loss_leg_metric = loss_leg_metric.unsqueeze(1).expand(B, T, N, F)
				loss_leg_log = loss_leg_log.unsqueeze(1).expand(B, T, N, FLOG)
				loss_leg_edge = loss_leg_edge.unsqueeze(1).expand(B, T, E, FEDGE)

				# --- Final combined losses ---
				rec_node_metric_fits = self.rec_lambda * loss_time_node_metric + self.auxi_lambda * loss_leg_metric
				rec_node_log_fits = self.rec_lambda * loss_time_log + self.auxi_lambda * loss_leg_log
				rec_edge1 = self.rec_lambda * loss_time_edge + self.auxi_lambda * loss_leg_edge
				

			# --- STEP 7: Scatter Masked Edges Back into Full Graph Space [B, T, N, N, E] ---
			rec_edge = torch.matmul(rec_edge1.permute(
				0, 1, 3, 2), self.trace2pod.float().to(rec_edge1.device)).permute(0, 1, 3, 2)
			#rec_edge = torch.matmul(rec_edge1.permute(
			#	0, 1, 3, 2), self.trace2pod.float()).permute(0, 1, 3, 2)
			rec = torch.concat([rec_node_metric_fits,rec_node_log_fits, rec_edge], dim=-1)

		elif self.FREQ_DOMAIN in ["FourierGNN"]:
			B, T, _,_ = x['data_node'].shape
			# get edge mask
			edge_exists_mask = (self.node_efea.sum(dim=-1) != 0)  # [N, N] boolean mask
			edge_exists_mask_batch = edge_exists_mask.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)  # [B, T, N, N]

			# Get embeddings
			x_node_metric_fits, _ = self.node_emb(x['data_node'])  # Shape: [B, T, N, F]
			x_node_logs_fits, _ = self.log_emb(x['data_log'])  # Shape: [B, T, L, F]
			x_edge_fits, _ = self.egde_emb(x['data_edge'])  # Shape: [B, T, E, F]
			x_node_combined = torch.cat([x_node_metric_fits, x_node_logs_fits], dim=-1)  # [B, T, N, F_m + F_l]

			# Permute to FITS input shape: [B*N, T, F]
			_, _, N, F_METRIC = x_node_metric_fits.shape
			x_node_input = x_node_combined.permute(0, 2, 1, 3).contiguous()  # [B, N, T, F]
			_, _, N, F_LOG = x_node_logs_fits.shape
			#x_node_logs_fits_input = x_node_logs_fits.permute(0, 2, 1, 3).reshape(B*N, T, F_LOG) # [B*N, T, F]
			_, _, _, _, E = x_edge_fits.shape
			#x_edge_flat = x_edge_fits.reshape(B, T, N*N, E)  # [B, T, N*N, E]
			edge_mask_flat = edge_exists_mask.view(-1)  # [N*N]
			#x_edge_masked = x_edge_flat[:, :, edge_mask_flat, :]  # select only existing edges --> # [B, T, N, N]
			#x_edge_fits_input = x_edge_masked.permute(0, 2, 1, 3).reshape(B * edge_mask_flat.sum().item(), T, E)  # [B*num_edges, T, E]
			#As fourierGC expects the full adjacency matrix of shape [B, T, N, N], not masked edges -> but zero masked edges
			adj_ft_full = self.adj_proj(x_edge_fits)  # project edge features to scalar weights [B, T, N, N, 1]
			adj_ft_full = adj_ft_full.squeeze(-1)    # [B, T, N, N]
			adj_ft = adj_ft_full * edge_exists_mask.unsqueeze(0).unsqueeze(0)  # zero masked edges

			rec_node_features = self.shared_fgn(x_node_input, adj_ft)  # [B, N, pre_length]
			
			"""
			if self.FREQ_DOMAIN == "FourierGNN":
				inner_model = self.shared_fgn
			elif self.FREQ_DOMAIN == "FITS":
				inner_model = self.shared_fits
			elif self.FREQ_DOMAIN == "GPT2":
				inner_model = self.shared_GPT2
			
			
			if not self.multi_fits:
				x_node_proj = self.modality_proj['node'](x_node_metric_fits_input)
				rec_node_metric_fits = self.modality_proj_out['node'](inner_model(x_node_proj))

				x_log_proj = self.modality_proj['log'](x_node_logs_fits_input)
				rec_node_logs_fits = self.modality_proj_out['log'](inner_model(x_log_proj))

				x_edge_proj = self.modality_proj['edge'](x_edge_fits_input)
				rec_edge_fits = self.modality_proj_out['edge'](inner_model(x_edge_proj))
			else:# only implemened for FITS
				rec_node_metric_fits, _ = inner_model(x_node_metric_fits_input)  # [B*N, T', F]
				rec_node_logs_fits, _   = inner_model(x_node_logs_fits_input)  # [B*N, T', F]
				rec_edge_fits, _ = inner_model(x_edge_fits_input)  # [B*N*N, T', E]
			"""
			rec_node_metric_fits = self.modality_proj_out['node'](rec_node_features)  # [B, N, F_m]
			rec_node_logs_fits   = self.modality_proj_out['log'](rec_node_features)    # [B, N, F_l]
			
			rec_edge_fits = self.shared_fgn.compute_edge_features(rec_node_features)  # [B, N*N, 2*D]
			rec_edge_fits = self.modality_proj_out['edge'](rec_edge_fits)  # [B, N*N, E]

			# Reshape back to original shape
			pred_metric_node = rec_node_metric_fits.reshape(B, N, -1, F_METRIC).permute(0, 2, 1, 3)  # [B, T, N, F]
			pred_log_node    = rec_node_logs_fits.reshape(B, N, -1, F_LOG).permute(0, 2, 1, 3)  # [B, T, N, F]
			# Keep edge predictions in masked form: [B, T, num_edges, E]
			pred_edge_masked = rec_edge_fits[:, edge_mask_flat, :]  # [B, num_edges, E]
			pred_edge_masked = pred_edge_masked.reshape(B, edge_mask_flat.sum().item(), -1, E).permute(0, 2, 1, 3)  # [B, T, num_edges, E]

			# Extract ground truth edges using mask: [B, T, num_edges, E]
			l_edge = torch.masked_select(x['data_edge'], edge_exists_mask_batch.unsqueeze(-1)).reshape(B, T, edge_mask_flat.sum().item(), -1)

			# Square Loss
			rec_node_metric_fits = torch.square(self.dense_node(pred_metric_node) - x['data_node'])  # Calculate squared loss on nodes (full) [B, T, N, F]
			rec_node_log_fits 	 = torch.square(self.dense_log(pred_log_node) - x['data_log'])  # Calculate squared loss on nodes (full) [B, T, N, F]
			rec_edge1 = torch.square(self.dense_edge(pred_edge_masked) - l_edge)  #Calculate squared loss on edges (masked only) [B, T, num_edges, E]
			rec_edge = torch.matmul(rec_edge1.permute(
				0, 1, 3, 2), self.trace2pod.float()).permute(0, 1, 3, 2)
			rec = torch.concat([rec_node_metric_fits,rec_node_log_fits, rec_edge], dim=-1)
		elif self.FREQ_DOMAIN in ["Eadro"]:
			device = x['data_edge'].device
			self.graph = self.graph.to(device)
			rec = self.Eadro_Model(self.graph,x['data_node'], x['data_log'], x['data_edge'])
		
		elif self.FREQ_DOMAIN in ["AnoFusion"]:
			device = x['data_edge'].device
			self.graph = self.graph.to(device)
			
			#datanode (metric) -> torch.Size([batch, time, num_services, dim])
			#datalog -> torch.Size([batch, time, num_services, dim])
			#dataedge -> torch.Size([batch, time, num_services, num_services, dim])
			rec = self.AnoFusion(self.graph,x['data_node'], x['data_log'], x['data_edge'],device)
			a=1
		elif self.FREQ_DOMAIN in ["Art"]:
			device = x['data_edge'].device
			self.graph = self.graph.to(device)
			rec = self.Art_Model(x['data_node'], x['data_log'], x['data_edge'])


		if evaluate:
			if rec.dim() == 4:	 
				rec = rec[:, -1].squeeze()
			cls_result = torch.softmax(self.show(rec), dim=-1)
			return cls_result, x['groundtruth_cls']#torch.Size([50, 5, 3])
		else:
			cls_label = x['groundtruth_cls']

			#cls_label
			# if 4d
			if rec.dim() == 4:	 
				rec = rec[:, -1].squeeze()
			# B, N, F
			cls_result = self.show(rec)
			cls_result = cls_result.reshape(-1, cls_result.shape[-1])
			cls_label = cls_label.reshape(-1, cls_label.shape[-1])

			if cls_label.shape[-1] == 3:
				mask = cls_label[:, -1]
				cls_result, cls_label = cls_result[mask == 0], cls_label[mask == 0]
				cls_label = cls_label[:, :cls_result.shape[-1]]

			# rec_loss
			label_pod = torch.argmax(x['groundtruth_cls'], dim=-1)  # B*N

			#node_rec = torch.sum(rec, dim=-1)
			node_rec = torch.mean(rec, dim=-1)
			node_right = torch.where(label_pod == 0, node_rec,
			                         torch.zeros_like(node_rec).to(node_rec.device))
			m = 1.0
			node_wrong = torch.where(
				label_pod == 1,
				torch.relu(node_rec - m),
				torch.zeros_like(node_rec)
			)
			node_unkown = torch.where(label_pod == 2, self.label_weight *
			                          node_rec, torch.zeros_like(node_rec).to(node_rec.device))
			rec_loss = [node_right, node_wrong, node_unkown]


			param = label_pod.shape[0] * label_pod.shape[1]
			rec_loss = list(map(lambda x: x.sum() / param, rec_loss))

			return rec_loss, cls_result, cls_label
