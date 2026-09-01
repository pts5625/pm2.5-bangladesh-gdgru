"""
Builds Table V: statistical significance of GD-GRU vs. every baseline, at
the pooled ("ALL") level and at each individual forecast horizon.

For every (baseline, horizon) pair this computes:
  - Diebold-Mariano test (HLN small-sample correction), two-sided
  - One-sided Wilcoxon signed-rank test on |error|
  - Cohen's d (paired) on |error|
  - A 95% block-bootstrap CI on the MAE gap
Bonferroni and Benjamini-Hochberg FDR correction are then applied once,
across the full family of comparisons.

Reads the same cached prediction arrays as ``evaluate.py``.

Usage
-----
    python -m pm25gdgru.significance
"""

import os

import numpy as np
import pandas as pd

from .evaluate import load_all_predictions, build_table_iv
from .metrics import diebold_mariano_test, wilcoxon_test, cohens_d_paired, \
    bootstrap_diff_ci, multiplicity_correction

RESULTS_DIR = os.environ.get("PM25_RESULTS_DIR", "results")


def build_table_v(pred_dict: dict, true_orig: np.ndarray, gd_gru_key: str = "GD-GRU",
                   horizon_labels=("ALL", "+1d", "+2d", "+3d"),
                   n_boot: int = 2000, block_size: int = 10,
                   alpha: float = 0.05) -> pd.DataFrame:
    other_models = [m for m in pred_dict if m != gd_gru_key]
    rows = []
    for other in other_models:
        for h_idx, h_label in enumerate(horizon_labels):
            if h_label == "ALL":
                e_gd = (pred_dict[gd_gru_key] - true_orig).ravel()
                e_oth = (pred_dict[other] - true_orig).ravel()
                h_dm = 1
            else:
                h = h_idx - 1
                e_gd = (pred_dict[gd_gru_key][:, h, :] - true_orig[:, h, :]).ravel()
                e_oth = (pred_dict[other][:, h, :] - true_orig[:, h, :]).ravel()
                h_dm = h + 1

            dm_stat, dm_p, T = diebold_mariano_test(e_gd, e_oth, h=h_dm, loss="squared")
            wc_stat, wc_p = wilcoxon_test(e_gd, e_oth, alternative="less")
            d = cohens_d_paired(e_gd, e_oth)
            diff_mean, diff_lo, diff_hi = bootstrap_diff_ci(e_gd, e_oth, n_boot, block_size)

            rows.append({
                "horizon": h_label, "baseline": other,
                "gd_gru_mae": np.abs(e_gd).mean(), "baseline_mae": np.abs(e_oth).mean(),
                "mae_diff": diff_mean, "mae_diff_ci_lo": diff_lo, "mae_diff_ci_hi": diff_hi,
                "dm_stat": dm_stat, "dm_p_raw": dm_p,
                "wilcoxon_stat": wc_stat, "wilcoxon_p_raw": wc_p,
                "cohens_d": d, "n_test_points": T,
            })

    df = pd.DataFrame(rows)
    p_bonf, sig_bonf, p_fdr, sig_fdr = multiplicity_correction(df["dm_p_raw"].values, alpha)
    df["dm_p_bonferroni"] = p_bonf
    df["dm_p_fdr_bh"] = p_fdr
    df["sig_bonferroni"] = sig_bonf
    df["sig_fdr_bh"] = sig_fdr
    return df


def mark_best_if_significant(table_iv_df: pd.DataFrame, table_v_df: pd.DataFrame,
                              gd_gru_key: str = "GD-GRU") -> pd.DataFrame:
    """Marks a baseline's Table IV row 'significant' vs. GD-GRU only where
    the FDR-corrected DM test at the pooled ('ALL') level rejects the null
    -- a non-significant MAE gap should never be read as a baseline win."""
    sig_map = (table_v_df[table_v_df["horizon"] == "ALL"]
               .set_index("baseline")["sig_fdr_bh"].to_dict())
    out = table_iv_df.copy()

    def marker(row):
        if row["model"] == gd_gru_key:
            return ""
        return "significant" if sig_map.get(row["model"], False) else "ns"

    out["sig_vs_GD-GRU_FDR"] = out.apply(marker, axis=1)
    return out


def run():
    pred_dict, true_orig = load_all_predictions()
    print(f"Models loaded for comparison: {list(pred_dict.keys())}")

    table_v = build_table_v(pred_dict, true_orig)
    out_path = os.path.join(RESULTS_DIR, "table_v.csv")
    table_v.to_csv(out_path, index=False)
    print(f"\nSaved Table V ({len(table_v)} comparisons) -> {out_path}\n")
    print(table_v[["horizon", "baseline", "gd_gru_mae", "baseline_mae", "mae_diff",
                    "dm_p_fdr_bh", "cohens_d", "sig_fdr_bh"]].to_string(index=False))

    table_iv = build_table_iv(pred_dict, true_orig)
    table_iv = mark_best_if_significant(table_iv, table_v)
    iv_path = os.path.join(RESULTS_DIR, "table_iv_with_significance.csv")
    table_iv.to_csv(iv_path, index=False)
    print(f"\nSaved Table IV (with significance markers) -> {iv_path}\n")
    print(table_iv.to_string(index=False))

    return table_v, table_iv


if __name__ == "__main__":
    run()
