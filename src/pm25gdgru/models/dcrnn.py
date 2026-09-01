"""
DCRNN: bidirectional random-walk diffusion convolution GRU (Li et al.,
ICLR 2018), architecturally mirroring GD-GRU (same input projection,
per-city output heads, teacher-forcing decoder) except the diffusion
operator is bidirectional random-walk rather than Chebyshev polynomial.

``DCRNNNet.forward`` takes ``graph`` as a ``(T_fwd, T_bwd)`` tuple of
row-normalised transition matrices, matching the shared training engine's
``forward(x, graph, target, teacher_forcing_ratio)`` interface.
"""

import torch
import torch.nn as nn

from ..config import Config

DCRNN_K = 2   # diffusion steps, matches GD-GRU's Chebyshev order K=2


def transition_matrix(A: torch.Tensor) -> torch.Tensor:
    """Row-normalises an adjacency matrix into a Markov transition matrix
    T = D^-1 A."""
    row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return A / row_sum


class DiffusionConv(nn.Module):
    """Bidirectional K-step diffusion convolution."""

    def __init__(self, in_features: int, out_features: int, K: int = DCRNN_K):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(2 * K * in_features, out_features, bias=True)

    def _diffuse(self, T, X):
        steps, X_k = [], X
        for _ in range(self.K):
            steps.append(X_k)
            X_k = torch.einsum("mn,bnd->bmd", T, X_k)
        return steps

    def forward(self, X, T_fwd, T_bwd):
        fwd_steps = self._diffuse(T_fwd, X)
        bwd_steps = self._diffuse(T_bwd, X)
        Z = torch.cat(fwd_steps + bwd_steps, dim=-1)
        return self.linear(Z)


class DCGRUCell(nn.Module):
    """Diffusion Convolutional GRU Cell."""

    def __init__(self, in_features: int, hidden_size: int, K: int = DCRNN_K):
        super().__init__()
        combined = in_features + hidden_size
        self.diff_reset = DiffusionConv(combined, hidden_size, K)
        self.diff_update = DiffusionConv(combined, hidden_size, K)
        self.diff_cand = DiffusionConv(combined, hidden_size, K)

    def forward(self, X_t, H_prev, T_fwd, T_bwd):
        XH = torch.cat([X_t, H_prev], dim=-1)
        r_t = torch.sigmoid(self.diff_reset(XH, T_fwd, T_bwd))
        u_t = torch.sigmoid(self.diff_update(XH, T_fwd, T_bwd))
        XrH = torch.cat([X_t, r_t * H_prev], dim=-1)
        c_t = torch.tanh(self.diff_cand(XrH, T_fwd, T_bwd))
        return u_t * H_prev + (1.0 - u_t) * c_t


class DCRNNNet(nn.Module):
    def __init__(self, in_features: int, n_nodes: int, cfg: Config, K: int = DCRNN_K):
        super().__init__()
        H = cfg.HIDDEN_DIM
        self.H = H
        self.n_nodes = n_nodes
        self.horizon = cfg.HORIZON
        self.dropout_rate = cfg.DROPOUT

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, H), nn.LayerNorm(H),
            nn.GELU(), nn.Dropout(self.dropout_rate))
        self.enc_cell = DCGRUCell(H, H, K)
        self.dec_cell = DCGRUCell(1, H, K)
        self.drop = nn.Dropout(self.dropout_rate)
        self.proj_shared = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Dropout(self.dropout_rate))
        self.proj_city = nn.Linear(H, n_nodes, bias=True)

    def _extract_city_output(self, H):
        B, N, _ = H.shape
        shared = self.proj_shared(H)
        out_all = self.proj_city(shared)
        idx = torch.arange(N, device=H.device)
        return out_all[:, idx, idx]

    def forward(self, x, graph, target=None, teacher_forcing_ratio: float = 0.0):
        """graph: (T_fwd, T_bwd) tuple of row-normalised transition matrices."""
        T_fwd, T_bwd = graph
        B, L, N, Fd = x.shape
        x_proj = self.input_proj(x)
        H_enc = torch.zeros(B, N, self.H, device=x.device)
        for t in range(L):
            H_enc = self.enc_cell(x_proj[:, t], H_enc, T_fwd, T_bwd)
            H_enc = self.drop(H_enc)

        H_dec = H_enc
        dec_inp = x[:, -1, :, 0:1]
        preds = []
        for step in range(self.horizon):
            H_dec = self.dec_cell(dec_inp, H_dec, T_fwd, T_bwd)
            H_dec = self.drop(H_dec)
            out = self._extract_city_output(H_dec)
            preds.append(out)
            use_tf = (target is not None
                      and torch.rand(1).item() < teacher_forcing_ratio)
            dec_inp = (target[:, step].unsqueeze(-1) if use_tf
                       else out.unsqueeze(-1).detach())
        return torch.stack(preds, dim=1)
