"""
Trains the 5-seed GD-GRU ensemble end-to-end: loads and preprocesses the
data, builds the geographic adjacency matrix, trains one model per
ensemble seed, ensemble-averages the test predictions (Eq. 24), reports
overall / per-horizon / per-city metrics, saves publication figures and
LaTeX tables, and writes the cached prediction array used by every
downstream analysis (Table IV/V, explainability, sensitivity sweeps).

Usage
-----
    python -m pm25gdgru.train_gdgru
"""

import os

import numpy as np
import pandas as pd
import torch

from .config import cfg
from .data import load_and_preprocess, build_adjacency, make_windows
from .engine import train_ensemble
from .models import GDGRUNet
from .plotting import plot_all, print_latex_tables

RESULTS_DIR = os.environ.get("PM25_RESULTS_DIR", "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PRED_PATH = os.path.join(RESULTS_DIR, "predictions", "gdgru_final_predictions.npz")


def run(cfg=cfg):
    print("=" * 70)
    print("GRAPH DIFFUSION GRU ENSEMBLE -- PM2.5 FORECASTING")
    print("=" * 70)

    (tr, va, te, scaler, feat_cols, tgt_idx,
     city_meta, cities, N, dates) = load_and_preprocess(cfg)

    A_np, dist_km = build_adjacency(city_meta, cfg)
    A_geo = torch.from_numpy(A_np).to(cfg.DEVICE)
    print(f"Adjacency mode: {cfg.ADJ_MODE}  |  N={N} cities")

    X_tr, y_tr = make_windows(tr, cfg.LOOKBACK, cfg.HORIZON)
    X_va, y_va = make_windows(va, cfg.LOOKBACK, cfg.HORIZON)
    X_te, y_te = make_windows(te, cfg.LOOKBACK, cfg.HORIZON)
    print(f"Samples -- Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")

    feat_dim = len(feat_cols)
    n_params = sum(p.numel() for p in GDGRUNet(feat_dim, N, cfg).parameters()
                   if p.requires_grad)
    print(f"Parameters per model: {n_params:,}")
    print(f"Ensemble size: {len(cfg.ENSEMBLE_SEEDS)} seeds")

    result = train_ensemble(
        GDGRUNet, dict(in_features=feat_dim, n_nodes=N, cfg=cfg),
        X_tr, y_tr, X_va, y_va, X_te, y_te, A_geo, cfg, cfg.ENSEMBLE_SEEDS,
        scaler, tgt_idx, ckpt_dir=CKPT_DIR, tag="gdgru", verbose=True)

    print("\n" + "-" * 57)
    print("OVERALL TEST METRICS  (all cities x all horizons)")
    print("-" * 57)
    for k, v in result.overall.items():
        print(f"  {k:<12}: {v:.4f}")

    print(f"\n{'':6} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R2':>8} {'PCC':>8}")
    for h in range(cfg.HORIZON):
        m = result.per_horizon[h]
        print(f"  +{h + 1}d   {m['MAE']:>8.3f} {m['RMSE']:>8.3f} "
              f"{m['MAPE (%)']:>8.2f} {m['R2']:>8.4f} {m['PCC']:>8.4f}")

    from .metrics import compute_metrics
    city_metrics = {}
    print(f"\n{'City':<22} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R2':>8}")
    for n, city in enumerate(cities):
        m = compute_metrics(result.true_orig[:, :, n], result.pred_orig[:, :, n], cfg.MAPE_FLOOR)
        city_metrics[city] = m
        print(f"  {city:<20} {m['MAE']:>8.3f} {m['RMSE']:>8.3f} "
              f"{m['MAPE (%)']:>8.2f} {m['R2']:>8.4f}")

    os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)
    np.savez_compressed(PRED_PATH, pred_orig=result.pred_orig, true_orig=result.true_orig)
    print(f"\nSaved predictions -> {PRED_PATH}")

    plot_all(result.histories, result.preds_orig_per_seed,
             result.pred_orig, result.true_orig, result.per_horizon,
             city_metrics, cities, cfg, A_np, out_dir=os.path.join(RESULTS_DIR, "figures"))
    print_latex_tables(result.per_horizon, city_metrics, cfg)

    rows = []
    S, H, Nc = result.pred_orig.shape
    for s in range(S):
        for h in range(H):
            for n, city in enumerate(cities):
                rows.append(dict(sample=s, horizon=h + 1, city=city,
                                  observed=float(result.true_orig[s, h, n]),
                                  predicted=float(result.pred_orig[s, h, n])))
    out_csv = os.path.join(RESULTS_DIR, "test_predictions.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Predictions saved -> {out_csv}")

    return result, cities, city_metrics, A_np, dist_km


if __name__ == "__main__":
    run()
