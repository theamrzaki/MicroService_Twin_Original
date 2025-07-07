import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from src.model_util import *
from src.FITS import Model as FITSModel 
import argparse

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
		
		self.FREQ_DOMAIN = args['FREQ_DOMAIN']
		if not self.FREQ_DOMAIN:
			self.encoder = Encoder(graph=self.graph, node_embedding=args['feature_node'], edge_embedding=args['feature_edge'], log_embedding=args['feature_log'],
							node_heads=args['num_heads_node'], log_heads=args['num_heads_log'], edge_heads=args['num_heads_edge'],
							n2e_heads=args['num_heads_n2e'], e2n_heads=args['num_heads_e2n'],
							dropout=args['dropout'], batch_size=args['batch_size'], window_size=args['window'], num_layer=args['num_layer'], trace2pod=trace2pod)
			self.decoder = Decoder(graph=self.graph, node_embedding=args['feature_node'], edge_embedding=args['feature_edge'], log_embedding=args['feature_log'],
							node_heads=args['num_heads_node'], log_heads=args['num_heads_log'], edge_heads=args['num_heads_edge'],
							n2e_heads=args['num_heads_n2e'], e2n_heads=args['num_heads_e2n'],
							dropout=args['dropout'], batch_size=args['batch_size'], window_size=args['window'], num_layer=args['num_layer'], trace2pod=trace2pod)
		else:
			parser = argparse.ArgumentParser()
			config = parser.parse_args()

			config.win_size = args['window']  # Window size
			config.DSR = 1  # Downsampling rate
			config.cutfreq = 0  # Cut frequency for FITS, set to 0 for automatic calculation
			if config.cutfreq == 0:
				config.cutfreq = int((config.win_size / config.DSR)/2)
			assert (config.win_size / config.DSR)/2 >= config.cutfreq, 'cutfreq should be smaller than half of the window size after downsampling'

			config.seq_len = config.win_size//config.DSR
			config.pred_len = config.win_size-config.win_size//config.DSR
			config.individual	= False  
			if self.multi_fits:
				config.enc_in = args['feature_node']
				self.fits_node = FITSModel(config)  

				config.enc_in = args['feature_log'] 
				self.fits_log = FITSModel(config)  

				config.enc_in = args['feature_edge'] 
				self.fits_edge = FITSModel(config)  
			else:
				config.enc_in = 10
				self.shared_fits = FITSModel(config)
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

		self.node_emb = Embed(args['raw_node'], args['feature_node'], dim=4)
		self.log_emb = Embed(args['log_len'], args['feature_log'], dim=4)
		self.egde_emb = Embed(args['raw_edge'], args['feature_edge'], dim=5)

		self.trace2pod = torch.nn.functional.one_hot(adj[0], num_classes=self.graph.shape[0]) \
			+ torch.nn.functional.one_hot(adj[1], num_classes=self.graph.shape[0])
		self.trace2pod = self.trace2pod / 2

		self.dense_node = nn.Linear(args['feature_node'], args['raw_node'])
		self.dense_log = nn.Linear(args['feature_log'], args['log_len'])
		self.dense_edge = nn.Linear(args['feature_edge'], args['raw_edge'])

		self.show = nn.Sequential(nn.Linear(args['raw_node'] + args['raw_edge'] + args['log_len'], (args['raw_node'] + args['raw_edge'] + args['log_len']) // 2),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear((args['raw_node'] + args['raw_edge'] + args['log_len']) // 2, 2))

	def forward(self, x, evaluate=False):
		if self.FREQ_DOMAIN:
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
			x_edge_masked = x_edge_flat[:, :, edge_mask_flat, :]  # select only existing edges
			x_edge_fits_input = x_edge_masked.permute(0, 2, 1, 3).reshape(B * edge_mask_flat.sum().item(), T, E)  # [B*num_edges, T, E]


			# Pass through FITS
			if not self.multi_fits:
				x_node_proj = self.modality_proj['node'](x_node_metric_fits_input)
				rec_node_metric_fits = self.modality_proj_out['node'](self.shared_fits(x_node_proj)[0])

				x_log_proj = self.modality_proj['log'](x_node_logs_fits_input)
				rec_node_logs_fits = self.modality_proj_out['log'](self.shared_fits(x_log_proj)[0])

				x_edge_proj = self.modality_proj['edge'](x_edge_fits_input)
				rec_edge_fits = self.modality_proj_out['edge'](self.shared_fits(x_edge_proj)[0])
			else:
				rec_node_metric_fits, _ = self.fits_node(x_node_metric_fits_input)  # [B*N, T', F]
				rec_node_logs_fits, _   = self.fits_log(x_node_logs_fits_input)  # [B*N, T', F]
				rec_edge_fits, _ = self.fits_edge(x_edge_fits_input)  # [B*N*N, T', E]

			# Reshape back to original shape
			pred_metric_node = rec_node_metric_fits.reshape(B, N, -1, F_METRIC).permute(0, 2, 1, 3)  # [B, T, N, F]
			pred_log_node    = rec_node_logs_fits.reshape(B, N, -1, F_LOG).permute(0, 2, 1, 3)  # [B, T, N, F]
			# Keep edge predictions in masked form: [B, T, num_edges, E]
			pred_edge_masked = rec_edge_fits.reshape(B, edge_mask_flat.sum().item(), -1, E).permute(0, 2, 1, 3)  # [B, T, num_edges, E]

			# Extract ground truth edges using mask: [B, T, num_edges, E]
			l_edge = torch.masked_select(x['data_edge'], edge_exists_mask_batch.unsqueeze(-1)).reshape(B, T, edge_mask_flat.sum().item(), -1)

			# Square Loss
			rec_node_metric_fits = torch.square(self.dense_node(pred_metric_node) - x['data_node'])  # Calculate squared loss on nodes (full) [B, T, N, F]
			rec_node_log_fits 	 = torch.square(self.dense_log(pred_log_node) - x['data_log'])  # Calculate squared loss on nodes (full) [B, T, N, F]
			rec_edge1 = torch.square(self.dense_edge(pred_edge_masked) - l_edge)  #Calculate squared loss on edges (masked only) [B, T, num_edges, E]
			rec_edge = torch.matmul(rec_edge1.permute(
				0, 1, 3, 2), self.trace2pod.float()).permute(0, 1, 3, 2)
			rec = torch.concat([rec_node_metric_fits,rec_node_log_fits, rec_edge], dim=-1)


		else:
			x_node, d_node = self.node_emb(x['data_node'])
			x_edge, d_edge = self.egde_emb(x['data_edge'])
			x_log, d_log = self.log_emb(x['data_log'])

			z_node, z_edge, z_log = self.encoder(x_node, x_edge, x_log)
			node, edge, log = self.decoder(d_node, d_edge, d_log, z_node, z_edge, z_log)
			
			l_edge = torch.masked_select(x['data_edge'], self.graph.unsqueeze(-1).repeat(1, 1, x['data_edge'].shape[-1]).bool()) \
				.reshape(x['data_edge'].shape[0], x['data_edge'].shape[1], -1, x['data_edge'].shape[-1])

			rec_node = torch.square(self.dense_node(node) - x['data_node'])
			rec_edge1 = torch.square(self.dense_edge(edge) - l_edge)
			rec_log = torch.square(self.dense_log(log) - x['data_log'])

			rec_edge = torch.matmul(rec_edge1.permute(
				0, 1, 3, 2), self.trace2pod.float()).permute(0, 1, 3, 2)
			rec = torch.concat([rec_node, rec_log, rec_edge], dim=-1)
		if evaluate:
			rec = rec[:, -1].squeeze()
			cls_result = torch.softmax(self.show(rec), dim=-1)
			return cls_result, x['groundtruth_cls']
		else:
			cls_label = x['groundtruth_cls']

			#cls_label
			rec = rec[:, -1].squeeze()
			cls_result = self.show(rec)
			cls_result = cls_result.reshape(-1, cls_result.shape[-1])
			cls_label = cls_label.reshape(-1, cls_label.shape[-1])

			if cls_label.shape[-1] == 3:
				mask = cls_label[:, -1]
				cls_result, cls_label = cls_result[mask == 0], cls_label[mask == 0]
				cls_label = cls_label[:, :cls_result.shape[-1]]

			# rec_loss
			label_pod = torch.argmax(x['groundtruth_cls'], dim=-1)  # B*N

			node_rec = torch.sum(rec, dim=-1)
			node_right = torch.where(label_pod == 0, node_rec,
			                         torch.zeros_like(node_rec).to(node_rec.device))
			node_wrong = torch.where(label_pod == 1, torch.pow(node_rec, torch.tensor(
				-1, device=node_rec.device)), torch.zeros_like(node_rec).to(node_rec.device))
			node_unkown = torch.where(label_pod == 2, self.label_weight *
			                          node_rec, torch.zeros_like(node_rec).to(node_rec.device))
			rec_loss = [node_right, node_wrong, node_unkown]


			param = label_pod.shape[0] * label_pod.shape[1]
			rec_loss = list(map(lambda x: x.sum() / param, rec_loss))

			return rec_loss, cls_result, cls_label
