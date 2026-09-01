"""
GD-GRU: Chebyshev graph diffusion GRU encoder-decoder (manuscript Eq. 10-23).

``GDGRUNet.forward`` takes ``graph`` as a single normalised adjacency tensor
``A`` [N, N], matching the shared training engine's
``forward(x, graph, target, teacher_forcing_ratio)`` interface.
"""

import torch
import torch.nn as nn

from ..config import Config


class GDConv(nn.Module):
    """Graph Diffusion Convolution: a truncated Chebyshev polynomial
    expansion of the normalised adjacency (Eq. 10-11)."""

    def __init__(self, in_dim: int, out_dim: int, cheb_k: int):
        super().__init__()
        self.cheb_k = cheb_k
        self.W = nn.Parameter(torch.empty(cheb_k * in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor, A: torch.Tensor, drop_edge_rate: float = 0.0):
        if self.training and drop_edge_rate > 0:
            # Stochastic edge dropout (Eq. 8-9), re-sampled every forward pass.
            mask = (torch.rand_like(A) > drop_edge_rate).float()
            A = A * mask
            row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
            A = A / row_sum

        A_exp = A.unsqueeze(0).expand(x.size(0), -1, -1)
        x0 = x
        x1 = torch.bmm(A_exp, x)
        supports = [x0, x1]
        for _ in range(2, self.cheb_k):
            x2 = 2 * torch.bmm(A_exp, x1) - x0
            supports.append(x2)
            x0, x1 = x1, x2

        h = torch.cat(supports, dim=-1)
        return h @ self.W + self.b


class GDGRUCell(nn.Module):
    """GRU cell whose three linear gates are replaced by GDConv (Eq. 12-15)."""

    def __init__(self, in_dim: int, hid_dim: int, cheb_k: int):
        super().__init__()
        self.gc_r = GDConv(in_dim + hid_dim, hid_dim, cheb_k)
        self.gc_u = GDConv(in_dim + hid_dim, hid_dim, cheb_k)
        self.gc_c = GDConv(in_dim + hid_dim, hid_dim, cheb_k)

    def forward(self, x, h, A, drop_edge_rate: float = 0.0):
        xh = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.gc_r(xh, A, drop_edge_rate))
        u = torch.sigmoid(self.gc_u(xh, A, drop_edge_rate))
        xrh = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.gc_c(xrh, A, drop_edge_rate))
        return u * h + (1 - u) * c


class GDGRUNet(nn.Module):
    """Encoder-decoder GD-GRU with per-city output heads."""

    def __init__(self, in_features: int, n_nodes: int, cfg: Config):
        super().__init__()
        H = cfg.HIDDEN_DIM
        self.horizon = cfg.HORIZON
        self.H = H
        self.n_nodes = n_nodes
        self.drop_edge_rate = cfg.DROP_EDGE_RATE
        self.city_heads = cfg.CITY_HEADS

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
        )
        self.enc_cell = GDGRUCell(H, H, cfg.CHEB_K)
        self.dec_cell = GDGRUCell(1, H, cfg.CHEB_K)
        self.drop = nn.Dropout(cfg.DROPOUT)

        if cfg.CITY_HEADS:
            self.proj_shared = nn.Sequential(
                nn.Linear(H, H), nn.GELU(), nn.Dropout(cfg.DROPOUT))
            self.proj_city = nn.Linear(H, n_nodes, bias=True)
        else:
            self.proj = nn.Sequential(
                nn.Linear(H, H), nn.GELU(),
                nn.Dropout(cfg.DROPOUT), nn.Linear(H, 1))

    def _decode_step(self, h):
        if self.city_heads:
            shared = self.proj_shared(h)
            out_all = self.proj_city(shared)
            idx = torch.arange(self.n_nodes, device=h.device)
            return out_all[:, idx, idx]
        return self.proj(h).squeeze(-1)

    def forward(self, x, graph, target=None, teacher_forcing_ratio: float = 0.0):
        """graph: normalised adjacency A [N, N]."""
        A = graph
        B, L, N, _ = x.shape
        h = torch.zeros(B, N, self.H, device=x.device)
        for t in range(L):
            inp = self.input_proj(x[:, t])
            h = self.enc_cell(inp, h, A, self.drop_edge_rate if self.training else 0.0)
            h = self.drop(h)

        dec_inp = x[:, -1, :, 0:1]
        preds = []
        for step in range(self.horizon):
            h = self.dec_cell(dec_inp, h, A, self.drop_edge_rate if self.training else 0.0)
            out = self._decode_step(h)
            preds.append(out)
            use_tf = (target is not None
                      and torch.rand(1).item() < teacher_forcing_ratio)
            dec_inp = (target[:, step].unsqueeze(-1) if use_tf
                       else out.unsqueeze(-1).detach())

        return torch.stack(preds, dim=1)   # [B, H, N]
