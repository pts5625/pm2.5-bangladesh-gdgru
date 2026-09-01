"""
Builds Table IV: overall test-set performance (MAE, RMSE, MAPE, R2, PCC)
for every model, with a 95% block-bootstrap confidence interval around
MAE/RMSE/R2.

The point estimate for every metric is always computed once on the real,
non-resampled test set -- the bootstrap is used only to derive the CI
around that point, never to redefine the point itself (the mean of many
block-bootstrap resamples of a right-skewed series like PM2.5 is a biased
stand-in for the value computed on the actual data).

Reads the cached prediction arrays written by ``train_gdgru.py``,
``train_dcrnn.py``, and ``baselines.py``; does not retrain anything.

Usage
-----
    python -m pm25gdgru.evaluate
"""

import os

import numpy as np
import pandas as pd

from .metrics import mae_fn, rmse_fn, r2_fn, mape_fn, pcc_fn, bootstrap_ci

RESULTS_DIR = os.environ.get("PM25_RESULTS_DIR", "results")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
BASELINE_PRED_DIR = os.path.join(PRED_DIR, "baselines")

BASELINE_NAMES = ["Persistence", "Climatology", "Random Forest", "XGBoost", "GRU (no graph)"]


def _baseline_pred_path(name: str) -> str:
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    return os.path.join(BASELINE_PRED_DIR, f"{safe}.npz")


def load_all_predictions():
    """Loads every cached prediction array and validates they all share the
    same ground truth (i.e. the same test windows)."""
    with np.load(os.path.join(PRED_DIR, "gdgru_final_predictions.npz")) as z:
        pred_dict = {"GD-GRU": z["pred_orig"]}
        true_orig = z["true_orig"]

    dcrnn_path = os.path.join(PRED_DIR, "dcrnn_final_predictions.npz")
    if os.path.exists(dcrnn_path):
        with np.load(dcrnn_path) as z:
            if not np.allclose(z["true_orig"], true_orig, atol=1e-4):
                raise ValueError("DCRNN predictions were not generated on the same "
                                  "test windows as GD-GRU -- retrain with the same "
                                  "Config (LOOKBACK/HORIZON/TRAIN_RATIO/VAL_RATIO"
                                  ") before comparing.")
            pred_dict["DCRNN"] = z["pred_orig"]

    for name in BASELINE_NAMES:
        p = _baseline_pred_path(name)
        if not os.path.exists(p):
            print(f"  [!] missing cached prediction for '{name}' at {p} -- "
                  f"Table IV will be incomplete without it")
            continue
        with np.load(p) as z:
            pred = z["pred"]
        if pred.shape != true_orig.shape:
            print(f"  [!] shape mismatch for '{name}': {pred.shape} vs "
                  f"{true_orig.shape} -- skipping")
            continue
        pred_dict[name] = pred

    return pred_dict, true_orig


def build_table_iv(pred_dict: dict, true_orig: np.ndarray, n_boot: int = 1000,
                    block_size: int = 10, mape_floor: float = 5.0) -> pd.DataFrame:
    rows = []
    for name, pred in pred_dict.items():
        yt = true_orig.reshape(true_orig.shape[0], -1)
        yp = pred.reshape(pred.shape[0], -1)

        mae_point = mae_fn(yt, yp)
        rmse_point = rmse_fn(yt, yp)
        r2_point = r2_fn(yt, yp)
        mape_point = mape_fn(yt, yp, mape_floor)
        pcc_point = pcc_fn(yt, yp)

        mae_lo, mae_hi = bootstrap_ci(yt, yp, mae_fn, n_boot, block_size)
        rmse_lo, rmse_hi = bootstrap_ci(yt, yp, rmse_fn, n_boot, block_size)
        r2_lo, r2_hi = bootstrap_ci(yt, yp, r2_fn, n_boot, block_size)

        # Self-check: the point estimate should always fall inside its own
        # CI. If it doesn't, the point and the CI were computed on
        # inconsistent data -- flag it rather than silently reporting it.
        flags = []
        if not (mae_lo <= mae_point <= mae_hi):
            flags.append("MAE point outside its own CI")
        if not (rmse_lo <= rmse_point <= rmse_hi):
            flags.append("RMSE point outside its own CI")
        if not (r2_lo <= r2_point <= r2_hi):
            flags.append("R2 point outside its own CI")

        rows.append({
            "model": name,
            "MAE": round(mae_point, 3), "MAE_CI": f"[{mae_lo:.3f}, {mae_hi:.3f}]",
            "RMSE": round(rmse_point, 3), "RMSE_CI": f"[{rmse_lo:.3f}, {rmse_hi:.3f}]",
            "MAPE (%)": round(mape_point, 2),
            "R2": round(r2_point, 4), "R2_CI": f"[{r2_lo:.4f}, {r2_hi:.4f}]",
            "PCC": round(pcc_point, 4),
            "check": "OK" if not flags else "; ".join(flags),
        })
    return pd.DataFrame(rows)


def run():
    pred_dict, true_orig = load_all_predictions()
    print(f"Models loaded: {list(pred_dict.keys())}")

    table_iv = build_table_iv(pred_dict, true_orig)
    out_path = os.path.join(RESULTS_DIR, "table_iv.csv")
    table_iv.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}\n")
    print(table_iv.to_string(index=False))

    if (table_iv["check"] != "OK").any():
        print("\n[WARNING] One or more point estimates fell outside their own CI. "
              "Do not use this table until investigated.")
    else:
        print("\nAll point estimates fall within their own CI.")

    return table_iv, pred_dict, true_orig


if __name__ == "__main__":
    run()
