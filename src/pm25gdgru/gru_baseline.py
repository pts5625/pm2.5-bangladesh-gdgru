"""
GRU (no graph): a direct ablation of GD-GRU with every GDConv operator
replaced by a standard linear/recurrent projection -- eliminates graph-based
message passing while keeping per-city output heads, dropout, weight decay,
input noise, and the teacher-forcing schedule identical to GD-GRU, so the
comparison isolates the contribution of graph diffusion specifically.

``PlainGRUNet.forward`` accepts (and ignores) a ``graph`` argument so it
shares the training engine's ``forward(x, graph, target,
teacher_forcing_ratio)`` interface with GDGRUNet and DCRNNNet.
"""

import torch
import torch.nn as nn

from ..config import Config


class PlainGRUNet(nn.Module):
    def __init__(self, in_features: int, n_nodes: int, cfg: Config):
        super().__init__()
        H = cfg.HIDDEN_DIM
        drop = cfg.DROPOUT
        self.H = H
        self.horizon = cfg.HORIZON
        self.n_nodes = n_nodes

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, H), nn.LayerNorm(H),
            nn.GELU(), nn.Dropout(drop))
        self.enc_gru = nn.GRU(H, H, batch_first=True)
        self.dec_gru = nn.GRUCell(1, H)
        self.drop = nn.Dropout(drop)
        self.proj_shared = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Dropout(drop))
        self.proj_city = nn.Linear(H, n_nodes, bias=True)

    def forward(self, x, graph=None, target=None, teacher_forcing_ratio: float = 0.0):
        """graph is accepted for interface parity with the graph-based
        models but is unused -- each city's sequence is processed
        independently."""
        B, L, N, Fd = x.shape
        x_proj = self.input_proj(x.reshape(B * N, L, Fd))
        _, h = self.enc_gru(x_proj)
        h = self.drop(h.squeeze(0))

        dec_inp = x[:, -1, :, 0].reshape(B * N, 1)
        preds = []
        for step in range(self.horizon):
            h = self.dec_gru(dec_inp, h)
            h = self.drop(h)
            shared = self.proj_shared(h)
            out_all = self.proj_city(shared).reshape(B, N, N)
            idx = torch.arange(N, device=h.device)
            out = out_all[:, idx, idx]
            preds.append(out)
            use_tf = (target is not None
                      and torch.rand(1).item() < teacher_forcing_ratio)
            dec_inp = (target[:, step].reshape(B * N, 1) if use_tf
                       else out.reshape(B * N, 1).detach())
        return torch.stack(preds, dim=1)
