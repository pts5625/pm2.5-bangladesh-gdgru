"""
pm25gdgru — Graph Diffusion GRU for Multi-City PM₂.₅ Forecasting in Bangladesh
================================================================================

Public API (re-exported for convenience):
    GDGRUNet        — Graph Diffusion GRU encoder-decoder network
    DCRNNNet        — Diffusion Convolutional RNN (baseline comparison)
    cfg             — Default configuration object
    load_and_preprocess, build_adjacency, make_windows — data pipeline helpers
    train_ensemble  — unified training loop (used by all model variants)
"""

from .models import GDGRUNet, DCRNNNet
from .config import cfg
from .data import load_and_preprocess, build_adjacency, make_windows
from .engine import train_ensemble

__all__ = [
    "GDGRUNet",
    "DCRNNNet",
    "cfg",
    "load_and_preprocess",
    "build_adjacency",
    "make_windows",
    "train_ensemble",
]
