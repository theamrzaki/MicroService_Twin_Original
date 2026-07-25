import torch
import torch.nn as nn
import torch.nn.functional as F
from src.inner_models.layers.muad.encoders import MetricEncoder, TraceEncoder, LogEncoder
from src.inner_models.layers.muad.graph_model import GraphModel1, SimpleAttention
from src.inner_models.layers.muad.layers import LinearLayer

"""

Uncertainty-Aware Multimodal Anomaly Detection
for Microservice Systems With Active Learning

"""
class UncertainBlock(nn.Module):
    def __init__(self, in_dim=64, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )
        self.fc_mu = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid())
        self.fc_var = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid())
        self.attention_mu = SimpleAttention(64)
        self.attention_var = SimpleAttention(64)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        var = self.fc_var(encoded)
        mu_attn = self.attention_mu(mu)
        var_attn = self.attention_var(var)
        new_sample = self.reparametrize(mu_attn, var_attn)
        return new_sample, self.kl_loss(mu_attn, var_attn)

    def reparametrize(self, mu, var):
        std = var.sqrt()
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu, var):
        return -0.5 * torch.mean(torch.sum(1 + var - mu ** 2 - var.exp(), dim=-1))


class MainModel(nn.Module):
    def __init__(self,  raw_metric, raw_logs, raw_traces,
                 feature_metric, feature_logs, feature_traces,
                 device, num_class=2, hidden_dim=[64], aloss=0.4, b_loss=0.01, **kwargs):      
        super().__init__()
        self.device = device
        self.metric_encoder = MetricEncoder(raw_metric,device, **kwargs)
        self.log_encoder = LogEncoder(raw_logs, device, **kwargs)
        self.trace_encoder = TraceEncoder(raw_traces, device, **kwargs)

        self.metric_uncertain_block = UncertainBlock()
        self.log_uncertain_block = UncertainBlock()
        self.trace_uncertain_block = UncertainBlock()


        self.TCPClassifierLayer_trace = LinearLayer(hidden_dim[0], num_class)
        self.TCPConfidenceLayer_trace = LinearLayer(hidden_dim[0], 1)
        self.TCPClassifierLayer_metric = LinearLayer(hidden_dim[0], num_class)
        self.TCPConfidenceLayer_metric = LinearLayer(hidden_dim[0], 1)
        self.TCPClassifierLayer_log = LinearLayer(hidden_dim[0], num_class)
        self.TCPConfidenceLayer_log = LinearLayer(hidden_dim[0], 1)
        self.TCPConfidenceLayer = LinearLayer(192, 1)


        # Outputs reconstructed features of size (raw_metric + raw_logs + raw_traces)
        total_raw_dim = raw_metric + raw_logs + raw_traces
        fused_in_dim = 3 * hidden_dim[0]
        mm_layers = []
        for i in range(1, len(hidden_dim)):
            #in_dim = 3 * hidden_dim[0] if i == 1 else hidden_dim[i - 1]
            mm_layers.append(LinearLayer(fused_in_dim, hidden_dim[i]))
            mm_layers.append(nn.ReLU())
        mm_layers.append(LinearLayer(fused_in_dim, total_raw_dim))
        #self.MMClasifier = nn.Sequential(*mm_layers)
        self.reconstructor = nn.Sequential(*mm_layers)

        self.k1 = aloss
        self.k2 = b_loss

    def forward(self, graph, data_node, data_log, data_edge):#, #fault_indexs):#graph
        #batch_size = graph.batch_size
        batch_size, T, N, F_metric = data_node.shape  
        _,_, _, F_log = data_log.shape  
        _,_, _,_, F_trace = data_edge.shape 
        #change data_edge to [B, T, N, F_trace] for consistency
        data_edge = data_edge.mean(dim=3)  # Shape: [B, T, N, F_trace]
        graph = graph.nonzero().t().contiguous()  # Shape: [2, num_edges]
        metric = self.metric_encoder(data_node.view(batch_size * N, T, F_metric), graph)
        log = self.log_encoder(data_log.view(batch_size * N, T, F_log), graph)
        trace = self.trace_encoder(data_edge.view(batch_size * N, T, F_trace), graph)


        new_metric, m_kl_loss = self.metric_uncertain_block(metric["metric_embedding1"])
        new_trace, t_kl_loss = self.trace_uncertain_block(trace["trace_embedding1"])
        new_log, l_kl_loss = self.log_uncertain_block(log["log_embedding"])

        #mean_kl_loss = (m_kl_loss + t_kl_loss + l_kl_loss) / 3
        #y_anomaly = (fault_indexs >= 1).long()
#
        #criterion = nn.CrossEntropyLoss()
#
        #confidence_loss_trace = self._compute_confidence_loss(
        #    new_trace, y_anomaly, self.TCPClassifierLayer_trace, self.TCPConfidenceLayer_trace, criterion)
        #confidence_loss_metric = self._compute_confidence_loss(
        #    new_metric, y_anomaly, self.TCPClassifierLayer_metric, self.TCPConfidenceLayer_metric, criterion)
        #confidence_loss_log = self._compute_confidence_loss(
        #    new_log, y_anomaly, self.TCPClassifierLayer_log, self.TCPConfidenceLayer_log, criterion)
        #confidence_loss = confidence_loss_trace + confidence_loss_metric + confidence_loss_log


        TCPConfidence_trace_sig = torch.sigmoid(self.TCPConfidenceLayer_trace(new_trace))
        TCPConfidence_metric_sig = torch.sigmoid(self.TCPConfidenceLayer_metric(new_metric))
        TCPConfidence_log_sig = torch.sigmoid(self.TCPConfidenceLayer_log(new_log))

        feature = torch.cat((
            new_trace * TCPConfidence_trace_sig,
            new_metric * TCPConfidence_metric_sig,
            new_log * TCPConfidence_log_sig
        ), dim=-1)

        #MMlogit = self.MMClasifier(feature)
        #MMLoss = criterion(MMlogit, y_anomaly)


        #total_loss = MMLoss + 0.6 * confidence_loss + mean_kl_loss
        #y_pred = self.inference(MMlogit)
#
        #return {
        #    'MMlogit': MMlogit,
        #    'loss': total_loss,
        #    'y_pred': y_pred,
        #    'feture': feature,
        #    'TCPConfidence_sig': torch.sigmoid(self.TCPConfidenceLayer(feature))
        #}
        # Map fused latent features back to raw input space
        rec = self.reconstructor(feature)
        rec = F.relu(rec)  # Or F.softplus(rec) to ensure non-negative outputs
        rec = rec.view(batch_size, N, -1) # Reshape back to match [B, T, N, total_raw_dim]

        return rec

    def _compute_confidence_loss(self, x, y, classifier, confidence_layer, criterion):
        logit = classifier(x)
        prob = F.softmax(logit, dim=1)
        p_target = torch.gather(prob, 1, y.unsqueeze(1)).squeeze()
        confidence = torch.sigmoid(confidence_layer(x)).squeeze()
        return F.mse_loss(confidence, p_target) + criterion(logit, y)

    def inference(self, MMlogit):
        dect_logit = MMlogit.detach().argmax(dim=1)
        return (dect_logit >= 1).long().tolist()
    


