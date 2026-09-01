"""
Shared configuration for the PM2.5 spatiotemporal forecasting pipeline.

This is the single source of truth for every hyperparameter used by the
GD-GRU model, the DCRNN / GRU-no-graph baselines, and the extended
analyses (graph comparison, sensitivity sweeps, lookback ablation). All
other modules import ``Config`` / ``cfg`` from here rather than redefining
their own copies.
"""

import random

import numpy as np
import torch


class Config:
    # ---- Data --------------------------------------------------------------
    DATA_PATH = "data/pm25_bangladesh.csv"   # set to your CSV path
    TARGET_COL = "pm2_5"
    DATE_COL = "date"
    CITY_COL = "city"
    LAT_COL = "lat"
    LON_COL = "lon"



    # ---- Temporal windows ----------------------------------------------------
    LOOKBACK = 21
    HORIZON = 3
    USE_DATE_FEATURES = True

    # ---- Graph ---------------------------------------------------------------
    # "gaussian": Haversine distance -> Gaussian RBF kernel -> symmetric
    #   normalization -> stochastic edge dropout during training. This is the
    #   graph construction adopted throughout the manuscript.
    # "knn":       same kernel, restricted to each node's K nearest neighbours.
    #   Used only for the alternative-graph comparison (see graph_comparison.py).
    ADJ_MODE = "gaussian"
    KNN_K = 5
    DROP_EDGE_RATE = 0.10

    # ---- Model -----------------------------------------------------------------
    HIDDEN_DIM = 32
    CHEB_K = 2
    DROPOUT = 0.35
    CITY_HEADS = True

    # ---- Training --------------------------------------------------------------
    EPOCHS = 300
    BATCH_SIZE = 32
    LR = 3e-4
    WEIGHT_DECAY = 5e-3
    PATIENCE = 40
    MIN_EPOCHS = 80
    LR_PATIENCE = 15
    LR_FACTOR = 0.5
    MIN_LR = 5e-7
    MIN_DELTA = 1e-4
    VAL_SMOOTH = 0.3
    NOISE_STD = 0.15
    MIXUP_ALPHA = 0.3
    MAPE_FLOOR = 5.0

    # Fixed per-horizon loss weights: shorter horizons are weighted more
    # heavily, matching the manuscript's weighted Huber loss (Eq. 25).
    HORIZON_WEIGHTS = [1.4, 1.2, 1.0]

    # Teacher-forcing schedule (manuscript Eq. 19): ratio decays linearly
    # from TF_INITIAL to 0 over TF_DECAY_EPOCHS. TF_DECAY_EPOCHS=None decays
    # over the full training budget (EPOCHS).
    TF_INITIAL = 0.6
    TF_DECAY_EPOCHS = None

    # ---- Ensemble ----------------------------------------------------------------
    ENSEMBLE_SEEDS = [42, 7, 123, 999, 2024]

    # ---- Split -------------------------------------------------------------------
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15

    # ---- Device --------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def set_seed(seed: int) -> None:
    """Fixes every source of stochasticity for one training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
