"""
Training loss: weighted Huber loss (manuscript Eq. 25).

Only forecast-horizon weighting is applied, giving higher importance to
shorter (more actionable) forecast horizons. Every city is weighted equally.
"""

import torch
import torch.nn.functional as F

from .config import Config


def build_horizon_weights(cfg: Config) -> torch.Tensor:
    return torch.tensor(cfg.HORIZON_WEIGHTS, device=cfg.DEVICE, dtype=torch.float32)


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                         horiz_w: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """pred/target: [B, H, N]. horiz_w: [H]."""
    elem = F.huber_loss(pred, target, delta=delta, reduction="none")  # [B, H, N]
    elem = elem * horiz_w[None, :, None]
    return elem.mean()
