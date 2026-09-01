"""
Lookback-window ablation (manuscript Table S3).

The temporal sensitivity analysis (``explainability.py``) shows that
gradient sensitivity is concentrated in the most recent 5-10 days of the
21-day lookback window. This script tests that observation directly by
retraining the full 5-seed GD-GRU ensemble with lookback windows of
7, 10, 14, and 21 days, holding every other setting fixed.

Usage
-----
    python -m pm25gdgru.analysis.lookback_ablation
"""

import copy
import os

import torch

from ..config import cfg as default_cfg
from ..data import load_and_preprocess, build_adjacency, make_windows
from ..models import GDGRUNet
from ..engine import train_ensemble
from .common import load_existing, append_row, flatten_metrics

RESULTS_DIR = os.environ.get("PM25_RESULTS_DIR", "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints", "lookback_ablation")
CSV_PATH = os.path.join(RESULTS_DIR, "table_s3_lookback_ablation.csv")

LOOKBACKS = [7, 10, 14, 21]


def run(cfg=default_cfg, lookbacks=LOOKBACKS, ckpt_dir: str = CKPT_DIR,
        csv_path: str = CSV_PATH):
    print("=" * 70)
    print("LOOKBACK-WINDOW ABLATION (Table S3)")
    print("=" * 70)

    # The scaled train/val/test arrays and the graph do not depend on
    # LOOKBACK, so they only need to be built once.
    (tr, va, te, scaler, feat_cols, tgt_idx,
     city_meta, cities, N, dates) = load_and_preprocess(cfg)
    A_np, _ = build_adjacency(city_meta, cfg)
    A_geo = torch.from_numpy(A_np).to(cfg.DEVICE)
    feat_dim = len(feat_cols)

    done = load_existing(csv_path)
    rows = []
    for lb in lookbacks:
        key = f"lookback{lb}"
        if key in done:
            print(f"[skip] {key} already in {csv_path}")
            continue

        print(f"\n--- LOOKBACK = {lb} days ---")
        cfg_v = copy.deepcopy(cfg)
        cfg_v.LOOKBACK = lb

        X_tr, y_tr = make_windows(tr, lb, cfg.HORIZON)
        X_va, y_va = make_windows(va, lb, cfg.HORIZON)
        X_te, y_te = make_windows(te, lb, cfg.HORIZON)

        result = train_ensemble(
            GDGRUNet, dict(in_features=feat_dim, n_nodes=N, cfg=cfg_v),
            X_tr, y_tr, X_va, y_va, X_te, y_te, A_geo, cfg_v,
            cfg.ENSEMBLE_SEEDS, scaler, tgt_idx,
            ckpt_dir=ckpt_dir, tag=key, verbose=False)

        row = dict(key=key, lookback=lb, n_test_windows=len(X_te))
        row.update(flatten_metrics(result.overall))
        append_row(csv_path, row)
        rows.append(row)
        print(f"  MAE={result.overall['MAE']:.3f}  R2={result.overall['R2']:.4f}")

    print(f"\nSaved -> {csv_path}")
    return rows


if __name__ == "__main__":
    run()
