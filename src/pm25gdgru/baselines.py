"""
Baseline models compared against GD-GRU in Table IV/V: Persistence,
Climatology, Random Forest, XGBoost, and the GRU (no graph) ablation.

All baselines are built from ``data.load_and_preprocess`` /
``data.make_windows`` -- the SAME split and scaler GD-GRU uses -- so their
predictions are directly comparable to GD-GRU's own test predictions for
the significance suite in ``significance.py``.

Usage
-----
    python -m pm25gdgru.baselines
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost not installed -- `pip install xgboost` to include "
          "the XGBoost baseline. It will be skipped otherwise.")

from .config import cfg
from .data import load_and_preprocess, make_windows, inverse_target
from .engine import train_ensemble
from .models import PlainGRUNet

RESULTS_DIR = os.environ.get("PM25_RESULTS_DIR", "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
BASELINE_PRED_DIR = os.path.join(RESULTS_DIR, "predictions", "baselines")

# Explicit, complete hyperparameters for every baseline.
RF_HYPERPARAMS = dict(
    n_estimators=300, max_depth=12, min_samples_split=5, min_samples_leaf=2,
    max_features="sqrt", n_jobs=-1, random_state=42,
)
XGB_HYPERPARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0, n_jobs=-1,
)
CLIMATOLOGY_NOTES = (
    "Day-of-year mean computed on the training split only, then smoothed "
    "with a +/-7-day rolling window; the window wraps circularly at the "
    "year boundary (Dec 29-31 borrow from early January and vice versa)."
)


def _baseline_pred_path(name: str) -> str:
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    return os.path.join(BASELINE_PRED_DIR, f"{safe}.npz")


def _load_cached(name: str):
    p = _baseline_pred_path(name)
    if os.path.exists(p):
        with np.load(p) as z:
            return z["pred"]
    return None


def _save_cached(name: str, pred: np.ndarray):
    os.makedirs(BASELINE_PRED_DIR, exist_ok=True)
    np.savez_compressed(_baseline_pred_path(name), pred=pred)


def _unscale_full(arr_sc, scaler):
    T, N, Fd = arr_sc.shape
    return scaler.inverse_transform(arr_sc.reshape(-1, Fd)).reshape(T, N, Fd)


def _inverse_last_obs(X_sc, scaler, target_idx):
    """Inverse-transforms the last lookback-day's target value for each
    window -> [S, N] in original units (used by the persistence baseline)."""
    last_sc = X_sc[:, -1, :, target_idx]
    S, N = last_sc.shape
    dummy = np.zeros((S * N, scaler.mean_.shape[0]))
    dummy[:, target_idx] = last_sc.reshape(-1)
    return scaler.inverse_transform(dummy)[:, target_idx].reshape(S, N)


def run_persistence(X_te_sc, y_te_orig, scaler, tgt_idx, cfg):
    """forecast(t+h) = observed(t) for every horizon."""
    print("  Running: Persistence")
    last_obs = _inverse_last_obs(X_te_sc, scaler, tgt_idx)
    S, H, N = y_te_orig.shape
    pred = np.repeat(last_obs[:, None, :], H, axis=1)
    return np.clip(pred, 0, None)


def run_climatology(tr_sc, va_sc, te_sc, dates, cfg, scaler, n_train, n_val,
                     tgt_idx, y_te_orig):
    print("  Running: Climatological Mean")
    raw_train = _unscale_full(tr_sc, scaler)[:, :, tgt_idx]
    N = raw_train.shape[1]

    train_dates = dates[:n_train]
    doy_train = pd.to_datetime(train_dates).day_of_year.values
    doy_map = {}
    for doy in range(1, 367):
        idx = np.where(doy_train == doy)[0]
        doy_map[doy] = raw_train[idx].mean(axis=0) if len(idx) > 0 else raw_train.mean(axis=0)

    smooth_map = {}
    for doy in range(1, 367):
        window = [doy_map[((doy + d - 1) % 365) + 1] for d in range(-7, 8)]
        smooth_map[doy] = np.mean(window, axis=0)

    test_start = n_train + n_val
    test_dates = dates[test_start:]
    doy_test = pd.to_datetime(test_dates).day_of_year.values

    LOOKBACK, HORIZON = cfg.LOOKBACK, cfg.HORIZON
    S = len(te_sc) - HORIZON - LOOKBACK + 1
    pred = np.zeros((S, HORIZON, N))
    for i in range(S):
        for h in range(HORIZON):
            t_idx = LOOKBACK + i + h
            future_doy = doy_test[t_idx] if t_idx < len(doy_test) else doy_test[-1]
            pred[i, h, :] = smooth_map[future_doy]
    return np.clip(pred[:len(y_te_orig)], 0, None)


def run_random_forest(X_tr_sc, y_tr_orig, X_te_sc, N, cfg):
    print(f"  Training: Random Forest (per city x horizon) -- {RF_HYPERPARAMS}")
    S_tr, L, _, Fd = X_tr_sc.shape
    S_te = len(X_te_sc)
    HORIZON = cfg.HORIZON
    preds = np.zeros((S_te, HORIZON, N))
    for n in tqdm(range(N), desc="    RF city"):
        X_tr_n = X_tr_sc[:, :, n, :].reshape(S_tr, L * Fd)
        X_te_n = X_te_sc[:, :, n, :].reshape(S_te, L * Fd)
        for h in range(HORIZON):
            rf = RandomForestRegressor(**RF_HYPERPARAMS)
            rf.fit(X_tr_n, y_tr_orig[:, h, n])
            preds[:, h, n] = rf.predict(X_te_n)
    return np.clip(preds, 0, None)


def run_xgboost(X_tr_sc, y_tr_orig, X_te_sc, N, cfg):
    if not HAS_XGB:
        print("  Skipping XGBoost (not installed).")
        return None
    print(f"  Training: XGBoost (per city x horizon) -- {XGB_HYPERPARAMS}")
    S_tr, L, _, Fd = X_tr_sc.shape
    S_te = len(X_te_sc)
    HORIZON = cfg.HORIZON
    preds = np.zeros((S_te, HORIZON, N))
    for n in tqdm(range(N), desc="    XGB city"):
        X_tr_n = X_tr_sc[:, :, n, :].reshape(S_tr, L * Fd)
        X_te_n = X_te_sc[:, :, n, :].reshape(S_te, L * Fd)
        for h in range(HORIZON):
            xgb = XGBRegressor(**XGB_HYPERPARAMS)
            xgb.fit(X_tr_n, y_tr_orig[:, h, n])
            preds[:, h, n] = xgb.predict(X_te_n)
    return np.clip(preds, 0, None)


def run_gru_no_graph(X_tr, y_tr, X_va, y_va, X_te, y_te, scaler, tgt_idx,
                      feat_dim, N, cfg):
    """5-seed GRU (no graph) ensemble, trained through the shared engine
    with graph=None (see models/gru_baseline.py)."""
    print(f"  Training: GRU (no graph), {len(cfg.ENSEMBLE_SEEDS)}-seed ensemble")
    result = train_ensemble(
        PlainGRUNet, dict(in_features=feat_dim, n_nodes=N, cfg=cfg),
        X_tr, y_tr, X_va, y_va, X_te, y_te, graph=None, cfg=cfg,
        seeds=cfg.ENSEMBLE_SEEDS, scaler=scaler, target_idx=tgt_idx,
        ckpt_dir=CKPT_DIR, tag="gru_nograph", verbose=False)
    return result.pred_orig


def run_all_baselines(cfg=cfg):
    """Builds every baseline's test-set predictions on GD-GRU's exact test
    windows. Returns {model_name: pred_orig [S,H,N]}, true_orig, cities."""
    (tr, va, te, scaler, feat_cols, tgt_idx,
     city_meta, cities, N, dates) = load_and_preprocess(cfg)
    n_train = int(len(dates) * cfg.TRAIN_RATIO)
    n_val = int(len(dates) * cfg.VAL_RATIO)
    feat_dim = len(feat_cols)

    X_tr, y_tr = make_windows(tr, cfg.LOOKBACK, cfg.HORIZON)
    X_va, y_va = make_windows(va, cfg.LOOKBACK, cfg.HORIZON)
    X_te, y_te = make_windows(te, cfg.LOOKBACK, cfg.HORIZON)

    y_tr_orig = np.clip(inverse_target(y_tr, scaler, tgt_idx), 0, None)
    y_te_orig = np.clip(inverse_target(y_te, scaler, tgt_idx), 0, None)

    results = {}

    print("\n[1/5] Persistence...")
    cached = _load_cached("Persistence")
    results["Persistence"] = cached if cached is not None else run_persistence(
        X_te, y_te_orig, scaler, tgt_idx, cfg)
    if cached is None:
        _save_cached("Persistence", results["Persistence"])

    print("\n[2/5] Climatology...")
    cached = _load_cached("Climatology")
    results["Climatology"] = cached if cached is not None else run_climatology(
        tr, va, te, dates, cfg, scaler, n_train, n_val, tgt_idx, y_te_orig)
    if cached is None:
        _save_cached("Climatology", results["Climatology"])

    print("\n[3/5] Random Forest...")
    cached = _load_cached("Random Forest")
    results["Random Forest"] = cached if cached is not None else run_random_forest(
        X_tr, y_tr_orig, X_te, N, cfg)
    if cached is None:
        _save_cached("Random Forest", results["Random Forest"])

    print("\n[4/5] XGBoost...")
    cached = _load_cached("XGBoost")
    if cached is not None:
        results["XGBoost"] = cached
    else:
        xgb_pred = run_xgboost(X_tr, y_tr_orig, X_te, N, cfg)
        if xgb_pred is not None:
            results["XGBoost"] = xgb_pred
            _save_cached("XGBoost", xgb_pred)

    print("\n[5/5] GRU (no graph)...")
    cached = _load_cached("GRU (no graph)")
    results["GRU (no graph)"] = cached if cached is not None else run_gru_no_graph(
        X_tr, y_tr, X_va, y_va, X_te, y_te, scaler, tgt_idx, feat_dim, N, cfg)
    if cached is None:
        _save_cached("GRU (no graph)", results["GRU (no graph)"])

    return results, y_te_orig, cities


if __name__ == "__main__":
    run_all_baselines()
