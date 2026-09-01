"""
Trains the 5-seed DCRNN ensemble on exactly the same data split, scaler,
and adjacency matrix as GD-GRU, so its predictions land on the same test
windows and can be paired directly against GD-GRU in the significance
suite (``significance.py``). The only architectural difference from
GD-GRU is the diffusion operator (bidirectional random-walk vs. Chebyshev
polynomial) -- the training discipline (optimizer, scheduler, early
stopping, noise, mixup, teacher-forcing schedule, loss) is identical.

Usage
-----
    python -m pm25gdgru.train_dcrnn
"""

import os

import numpy as np
import torch

from .config import cfg
from .data import load_and_preprocess, build_adjacency, make_windows
from .engine import train_ensemble
from .models import DCRNNNet
from .models.dcrnn import transition_matrix

RESULTS_DIR = os.environ.get("PM25_RESULTS_DIR", "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PRED_PATH = os.path.join(RESULTS_DIR, "predictions", "dcrnn_final_predictions.npz")


def run(cfg=cfg):
    print(f"cfg.ENSEMBLE_SEEDS = {cfg.ENSEMBLE_SEEDS}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    (tr, va, te, scaler, feat_cols, tgt_idx,
     city_meta, cities, N, dates) = load_and_preprocess(cfg)
    A_np, _ = build_adjacency(city_meta, cfg)
    A_geo = torch.from_numpy(A_np).to(cfg.DEVICE)
    T_fwd = transition_matrix(A_geo)
    T_bwd = transition_matrix(A_geo.T)   # equals T_fwd for a symmetric graph,
                                          # kept separate for DCRNN's independently
                                          # weighted bidirectional formulation
    print(f"Graph: {N}x{N} (same adjacency GD-GRU uses)")

    X_tr, y_tr = make_windows(tr, cfg.LOOKBACK, cfg.HORIZON)
    X_va, y_va = make_windows(va, cfg.LOOKBACK, cfg.HORIZON)
    X_te, y_te = make_windows(te, cfg.LOOKBACK, cfg.HORIZON)
    feat_dim = len(feat_cols)

    result = train_ensemble(
        DCRNNNet, dict(in_features=feat_dim, n_nodes=N, cfg=cfg),
        X_tr, y_tr, X_va, y_va, X_te, y_te, (T_fwd, T_bwd), cfg, cfg.ENSEMBLE_SEEDS,
        scaler, tgt_idx, ckpt_dir=CKPT_DIR, tag="dcrnn", verbose=True)

    print(f"\n  DCRNN ENSEMBLE  MAE={result.overall['MAE']:.3f}  "
          f"RMSE={result.overall['RMSE']:.3f}  MAPE={result.overall['MAPE (%)']:.2f}%  "
          f"R2={result.overall['R2']:.4f}  PCC={result.overall['PCC']:.4f}")

    os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)
    np.savez_compressed(PRED_PATH, pred_orig=result.pred_orig, true_orig=result.true_orig)
    print(f"Saved predictions -> {PRED_PATH}")

    return result


if __name__ == "__main__":
    run()
