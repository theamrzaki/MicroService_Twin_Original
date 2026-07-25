import torch.nn as nn
from .graph_model import GraphModel1
from .layers import GRUEncoder


class  MetricEncoder(nn.Module):
    def __init__(self, in_dim, device, out_dim=64, **kwargs):
        super().__init__()
        self.metric_model = GRUEncoder(in_dim, out_dim).to(device)
        self.status_model = GraphModel1(in_dim=out_dim, device=device, **kwargs)

    def forward(self, metric, graph):
        metric_embedding = self.metric_model(metric)
        return {'metric_embedding1': self.status_model(graph, metric_embedding)}


class TraceEncoder(nn.Module):
    def __init__(self, in_dim, device, out_dim=64, **kwargs):
        super().__init__()
        self.trace_model = GRUEncoder(in_dim, out_dim).to(device)
        self.status_model = GraphModel1(in_dim=out_dim, device=device, **kwargs)

    def forward(self, traces, graph):
        trace_embedding = self.trace_model(traces)
        return {"trace_embedding1": self.status_model(graph, trace_embedding)}


class LogEncoder(nn.Module):
    def __init__(self, in_dim,  device, out_dim=64,**kwargs):
        super().__init__()
        self.log_embedder = nn.Linear(in_dim, out_dim).to(device)
        self.status_model = GraphModel1(in_dim=out_dim, device=device, **kwargs)

    def forward(self, logs, graph):
        log_emb = self.log_embedder(logs)
        log_emb = log_emb.mean(dim=1)  # Or log_emb.max(dim=1)[0] for anomaly spikes
        return {'log_embedding': self.status_model(graph, log_emb)}