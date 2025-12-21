import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from src.model_util import *
from src.inner_models.FITS import Model as FITSModel 
from src.inner_models.FITS_LPF import Model as FITSModel_LPF 
from src.inner_models.FITS_Pai import Model as FITSModel_Pai
from src.inner_models.iTransformer import Model as iTransformerModel
from src.inner_models.DLinear import Model as DLinearModel
from src.inner_models.FourierGNN import FGN
from src.inner_models.GPT4TS import Model as GPT2Model
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
import argparse

from numpy.polynomial import Legendre as L

def phi(x):
    return torch.nn.functional.elu(x) + 1

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
		self.name = 'my'
		self.graph = torch.tensor(graph).cuda()
		self.label_weight = args['label_weight']
		self.multi_fits = args["MULTI_FITS"]
		adj = dense_to_sparse(self.graph)[0]
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

	def upsample_time_dim(self, tensor_4d: torch.Tensor, target_time: int) -> torch.Tensor:
		B, T_old, N, F_ = tensor_4d.shape
		tensor_3d = tensor_4d.permute(0, 2, 3, 1).reshape(B, N * F_, T_old)  # [B, C, T_old]
		tensor_upsampled = torch.nn.functional.interpolate(tensor_3d, size=target_time, mode='linear', align_corners=False)
		tensor_upsampled = tensor_upsampled.reshape(B, N, F_, target_time).permute(0, 3, 1, 2)  # [B, T_new, N, F]
		return tensor_upsampled

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


			if self.multi_fits == 'false':
				# ---- projections (unchanged) ----
				x_node_proj = self.modality_proj['node'](x_node_metric_fits_input)
				x_log_proj  = self.modality_proj['log'](x_node_logs_fits_input)

				# ---- shared temporal encoder ----
				h_node = self.shared_fits(x_node_proj)[0]   # [B*N, T, D]
				h_log  = self.shared_fits(x_log_proj)[0]    # [B*N, T, D]

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

			# Extract ground truth edges using mask: [B, T, num_edges, E]
			mask = edge_exists_mask_batch.to(x['data_edge'].device).unsqueeze(-1)
			l_edge = torch.masked_select(x['data_edge'], mask).reshape(B, T, edge_mask_flat.sum().item(), -1)
			#l_edge = torch.masked_select(x['data_edge'], edge_exists_mask_batch.unsqueeze(-1)).reshape(B, T, edge_mask_flat.sum().item(), -1)

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
			elif self.req_loss_approach in ["Legendre-style","chebyshev-style","lag-style","hermite-style"]:
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

				if self.req_loss_approach == "Legendre-style":
					# Legendre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_Legendre_operations.legendre_encode(pred_metric_merged, degree=10)  # [B, N*F, D]
					true_metric_leg = FITS_Legendre_operations.legendre_encode(true_metric_merged, degree=10)  # [B, N*F, D]

					pred_log_leg = FITS_Legendre_operations.legendre_encode(pred_log_merged, degree=10)
					true_log_leg = FITS_Legendre_operations.legendre_encode(true_log_merged, degree=10)
					
					pred_edge_leg = FITS_Legendre_operations.legendre_encode(pred_edge_merged, degree=10)
					true_edge_leg = FITS_Legendre_operations.legendre_encode(true_edge_merged, degree=10)
				elif self.req_loss_approach == "chebyshev-style":
					# Legendre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_chebyshev_operations.chebyshev_encode(pred_metric_merged, degree=2)  # [B, N*F, D]
					true_metric_leg = FITS_chebyshev_operations.chebyshev_encode(true_metric_merged, degree=2)  # [B, N*F, D]

					pred_log_leg = FITS_chebyshev_operations.chebyshev_encode(pred_log_merged, degree=2)
					true_log_leg = FITS_chebyshev_operations.chebyshev_encode(true_log_merged, degree=2)

					pred_edge_leg = FITS_chebyshev_operations.chebyshev_encode(pred_edge_merged, degree=2)
					true_edge_leg = FITS_chebyshev_operations.chebyshev_encode(true_edge_merged, degree=2)
				elif self.req_loss_approach == "lag-style":
					# Lagurre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_lag_operations.laguerre_encode(pred_metric_merged, degree=2)  # [B, N*F, D]
					true_metric_leg = FITS_lag_operations.laguerre_encode(true_metric_merged, degree=2)  # [B, N*F, D]

					pred_log_leg = FITS_lag_operations.laguerre_encode(pred_log_merged, degree=2)
					true_log_leg = FITS_lag_operations.laguerre_encode(true_log_merged, degree=2)

					pred_edge_leg = FITS_lag_operations.laguerre_encode(pred_edge_merged, degree=2)
					true_edge_leg = FITS_lag_operations.laguerre_encode(true_edge_merged, degree=2)
				elif self.req_loss_approach == "hermite-style":
					# Lagurre encode: outputs [B, C, degree]
					pred_metric_leg = FITS_hermite_operations.hermite_encode(pred_metric_merged, degree=2)  # [B, N*F, D]
					true_metric_leg = FITS_hermite_operations.hermite_encode(true_metric_merged, degree=2)  # [B, N*F, D]

					pred_log_leg = FITS_hermite_operations.hermite_encode(pred_log_merged, degree=2)
					true_log_leg = FITS_hermite_operations.hermite_encode(true_log_merged, degree=2)

					pred_edge_leg = FITS_hermite_operations.hermite_encode(pred_edge_merged, degree=2)
					true_edge_leg = FITS_hermite_operations.hermite_encode(true_edge_merged, degree=2)

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
			rec = self.AnoFusion(self.graph,x['data_node'], x['data_log'], x['data_edge'])
			a=1
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
