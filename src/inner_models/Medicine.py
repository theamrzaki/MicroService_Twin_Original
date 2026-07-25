from typing import Literal
from torch import nn
from src.inner_models.layers.medicine.encoder import LogEncoder, MetricEncoder, TraceEncoder
import torch


class GatedFusion(nn.Module):
    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return output

class ConcatFusion_for_classification_with_samespace(nn.Module):#for classigication
    def __init__(self, input_dim=1024, output_dim=100, num_class=5):
        super(ConcatFusion, self).__init__()
        self.linear_x_out = nn.Linear(output_dim, output_dim)
        self.linear_y_out = nn.Linear(output_dim, output_dim)
        self.linear_z_out = nn.Linear(output_dim, output_dim)
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.squeeze = nn.AdaptiveAvgPool2d((1, output_dim))
        self.clf = nn.Linear(output_dim, num_class)

    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=1)
        fc_out = self.fc_out(output)
        x_out = self.linear_x_out(x)
        y_out = self.linear_y_out(y)
        z_out = self.linear_z_out(z)
        output = torch.stack(
            (
                x,
                y,
                z,
                self.sigmoid(x_out),
                self.sigmoid(y_out),
                self.sigmoid(z_out),
                self.sigmoid(fc_out),
            ),
            dim=1,
        )
        output = self.squeeze(output)
        output = output.squeeze(dim=1)
        output = self.clf(output)
        return x_out, y_out, z_out, output


class ConcatFusion(nn.Module):
    def __init__(
        self,
        feature_metric: int = 8,
        feature_log: int = 32,
        feature_trace: int = 8,
        kpi_num: int = 15,
        num_log_templates: int = 10,
        invoke_num: int = 10,
        d_model: int = 64,  # Unified intermediate fusion dimension
    ):
        super().__init__()
        
        # 1. Project unequal modality features to a unified dimension (d_model)
        self.proj_x = nn.Linear(feature_metric, d_model)
        self.proj_y = nn.Linear(feature_log, d_model)
        self.proj_z = nn.Linear(feature_trace, d_model)
        
        # Total concatenated dimension
        fused_dim = feature_metric + feature_log + feature_trace
        self.fc_fused = nn.Linear(fused_dim, d_model)
        self.sigmoid = nn.Sigmoid()
        
        # Adaptive pooling across the 7 stacked attention/modality blocks
        self.squeeze = nn.AdaptiveAvgPool2d((1, d_model))
        
        # 2. Reconstruction Heads (Output Space mapping)
        self.rec_metric = nn.Linear(d_model, kpi_num)
        self.rec_log = nn.Linear(d_model, num_log_templates)
        self.rec_trace = nn.Linear(d_model, invoke_num)

    def forward(self, x, y, z):
        # x: (B, feature_metric), y: (B, feature_log), z: (B, feature_trace)
        
        # Step A: Project each modality to unified dimension d_model
        x_proj = self.proj_x(x)
        y_proj = self.proj_y(y)
        z_proj = self.proj_z(z)
        
        # Step B: Concatenate raw representations and project
        fused_raw = torch.cat((x, y, z), dim=1)
        fused_proj = self.fc_fused(fused_raw)
        
        # Step C: Stack projected representations (All are now shape [B, d_model])
        stacked = torch.stack(
            (
                x_proj,
                y_proj,
                z_proj,
                self.sigmoid(x_proj),
                self.sigmoid(y_proj),
                self.sigmoid(z_proj),
                self.sigmoid(fused_proj),
            ),
            dim=1,  # Shape: (B, 7, d_model)
        )
        
        # Step D: Squeeze / Aggregate across the 7 representations
        # (B, 7, d_model) -> pool -> (B, 1, d_model) -> squeeze -> (B, d_model)
        fused_repr = self.squeeze(stacked).squeeze(dim=1)
        
        # Step E: Reconstruction Outputs
        rec_x = self.rec_metric(fused_repr)
        rec_y = self.rec_log(fused_repr)
        rec_z = self.rec_trace(fused_repr)
        
        return rec_x, rec_y, rec_z, fused_repr
    
class AdaFusion(nn.Module):
    def __init__(
        self,
        kpi_num: int = 15,
        invoke_num: int = 10,
        instance_num: int = 10,
        num_log_templates: int = 10,
        feature_metric: int = 1,
        feature_log: int = 1,
        feature_trace: int = 1,
        max_len: int = 512,
        d_model: int = 768,
        nhead: int = 8,
        d_ff: int = 256,
        layer_num: int = 2,
        dropout: float = 0.1,
        num_class: int = 4,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.log_feature_encoder = nn.Linear(num_log_templates, feature_log)
        self.log_encoder = LogEncoder(
            max_len, feature_log, 1, d_ff, layer_num, dropout, device
        )
        self.metric_encoder = MetricEncoder(
            kpi_num,
            instance_num,
            max_len,
            feature_metric,
            nhead,
            d_ff,
            layer_num,
            dropout,
            device,
        )
        self.trace_encoder = TraceEncoder(invoke_num, feature_trace, nhead, d_ff, dropout)

        #self.clf = nn.Sequential(
        #    nn.Linear(d_model, num_class),
        #)
        #self.clf_cat = nn.Sequential(
        #    nn.Linear(d_model * 3, num_class),
        #)

        #self.concat_fusion = ConcatFusion(d_model * 3, d_model, num_class)
        self.concat_fusion = ConcatFusion(
            feature_metric=feature_metric,    # 8
            feature_log=feature_log,          # 32
            feature_trace=feature_trace,      # 8
            kpi_num=kpi_num,                  # Metric reconstruction dim
            num_log_templates=num_log_templates, # Log reconstruction dim
            invoke_num=invoke_num,            # Trace reconstruction dim
            d_model=d_model,                  # Shared fusion size (e.g. 64 or 128)
        )

    def forward(self, data_node, data_log, data_edge):#x_list):
        #make sure both are on the same device
        x_metric = self.metric_encoder(data_node.permute(0, 2, 1, 3))
        B, T, N, F = data_log.shape
        data_log = data_log.permute(0, 2, 1, 3).reshape(B * N, T, F)
        data_log = self.log_feature_encoder(data_log)
        x_log = self.log_encoder(data_log)
        
        data_edge = data_edge.mean(dim=3)                    # Collapse destination nodes
        B_e, T_e, N_e, F_e = data_edge.shape
        data_edge = data_edge.permute(0, 2, 1, 3).reshape(B_e * N_e, T_e, F_e)
        x_trace = self.trace_encoder(data_edge)
        x_trace = x_trace.mean(dim=1)

        rec_metric, rec_log, rec_trace, _ = self.concat_fusion(x_metric, x_log, x_trace)
        #concatinate all recs 
        rec = torch.cat([rec_metric, rec_log, rec_trace], dim=-1)
        # instead of B*T, F --> B,T,F
        rec = rec.reshape(B, N, -1)
        return rec


class ExperimentModel(nn.Module):
    def __init__(
        self,
        kpi_num: int,
        invoke_num: int,
        instance_num: int,
        max_len: int,
        d_model: int,
        nhead: int,
        d_ff: int,
        layer_num: int,
        dropout: float,
        num_class: int,
        device: str,
    ) -> None:
        super().__init__()
        self.log_encoder = LogEncoder(
            max_len, d_model, nhead, d_ff, layer_num, dropout, device
        )
        self.metric_encoder = MetricEncoder(
            kpi_num,
            instance_num,
            max_len,
            d_model,
            nhead,
            d_ff,
            layer_num,
            dropout,
            device,
        )
        self.trace_encoder = TraceEncoder(invoke_num, d_model, nhead, d_ff, dropout)

        self.clf = nn.Sequential(
            nn.Linear(d_model, num_class),
        )

        self.clf_cat = nn.Sequential(
            nn.Linear(d_model * 3, num_class),
        )

        self.use_modal = "all"

    def set_use_modal(self, modal: Literal["log", "metric", "trace", "all"] = "all"):
        self.use_modal = modal

    def forward(self, x_list):
        if self.use_modal == "all":
            return self.__fusion_forward__(x_list)
        else:
            return self.__single_forward__(x_list)

    def __fusion_forward__(self, x_list):
        x_log = self.log_encoder(x_list[0])
        x_metric = self.metric_encoder(x_list[1])
        x_trace = self.trace_encoder(x_list[2])
        return self.clf_cat(torch.cat([x_log, x_metric, x_trace], dim=-1))

    def __single_forward__(self, x_list):
        modal = self.use_modal
        hiddens = None
        if modal == "log":
            hiddens = self.log_encoder(x_list[0])
        elif modal == "metric":
            hiddens = self.metric_encoder(x_list[1])
        elif modal == "trace":
            hiddens = self.trace_encoder(x_list[2])
        else:
            raise Exception("unknown modal")
        return self.clf(hiddens)