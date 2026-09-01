"""
Explainability suite for the trained GD-GRU ensemble (manuscript Sec.
"Explainability Analysis" / Results "Explainability Results", Fig. 8-12,
Table 7).

Five complementary methods are applied to the SAME 5-seed ensemble trained
by ``train_gdgru.py`` (loaded from its cached checkpoints), so results here
are directly comparable to the paper:

  1. Gradient x Input      (Eq. 28)  -- which features drive forecasts most?
  2. Temporal Sensitivity  (Eq. 29)  -- which past lookback days matter most?
  3. Permutation Feature Importance (Eq. 30) -- MAE increase when a feature
                                                  is shuffled
  4. KernelSHAP             (Sec. "SHAP Values") -- marginal feature
                                                     contribution to the +1d
                                                     forecast
  5. Spatial Influence (occlusion, Eq. 31) -- how much does masking one
     city change every city's +1-day forecast, and how well does that
     match the fixed geographic adjacency?

Every attribution method below is averaged across all 5 ensemble members
(rather than a single representative model), which is a slightly more
faithful reading of "applied to the trained ensemble" than scoring one
seed alone.

Method agreement (manuscript: "the Spearman rank correlation and the
top-10 feature-set Jaccard overlap are computed between each pair of
methods") is computed directly from the in-memory Gradient x Input / PFI /
SHAP scores, no intermediate CSV round-trip required.

Usage
-----
    python -m pm25gdgru.explainability
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

import torch
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[warn] shap not installed -- KernelSHAP section will be skipped. "
          "Install with: pip install shap")

from .config import cfg as default_cfg
from .data import load_and_preprocess, build_adjacency, make_windows, inverse_target
from .models import GDGRUNet

RESULTS_DIR = os.environ.get("PM25_RESULTS_DIR", "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
OUT_DIR = os.path.join(RESULTS_DIR, "figures", "explainability")

N_SAMPLES = 200        # windows sampled for gradient / temporal / spatial methods
N_REPEATS = 5           # permutation repeats
TOP_K = 12
DPI = 300
EXCLUDE_FROM_IMP = None  # set to [cfg.TARGET_COL] below once cfg is known


# ------------------------------------------------------------------------
# Ensemble loading / inference
# ------------------------------------------------------------------------

def load_ensemble(cfg, feat_dim: int, N: int, ckpt_dir: str = CKPT_DIR,
                   tag: str = "gdgru"):
    """Loads every seed checkpoint written by ``train_gdgru.py`` (file names
    ``{ckpt_dir}/{tag}_seed{seed}.pt``)."""
    models = []
    for seed in cfg.ENSEMBLE_SEEDS:
        path = os.path.join(ckpt_dir, f"{tag}_seed{seed}.pt")
        if not os.path.exists(path):
            print(f"  [!] missing checkpoint: {path} -- skipping seed {seed}")
            continue
        m = GDGRUNet(feat_dim, N, cfg).to(cfg.DEVICE)
        m.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
        m.eval()
        models.append(m)
        print(f"  Loaded: {path}")
    if not models:
        raise FileNotFoundError(
            f"No GD-GRU checkpoints found under {ckpt_dir}. Run "
            f"`python -m pm25gdgru.train_gdgru` first.")
    print(f"Ensemble: {len(models)} model(s) loaded.")
    return models


def _predict_np(model, X_np: np.ndarray, A_geo, device, batch_size: int = 64):
    model.eval()
    loader = torch.utils.data.DataLoader(
        torch.from_numpy(X_np.astype(np.float32)), batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for xb in loader:
            out.append(model(xb.to(device), A_geo, target=None,
                              teacher_forcing_ratio=0.0).cpu().numpy())
    return np.concatenate(out, 0)


def _ensemble_predict_np(models, X_np: np.ndarray, A_geo, device):
    return np.mean([_predict_np(m, X_np, A_geo, device) for m in models], axis=0)


def _non_target_mask(feat_cols, target_col: str):
    arr = np.array(feat_cols)
    return arr != target_col


def _set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9, "figure.dpi": DPI,
    })


def _savefig(out_dir: str, basename: str):
    os.makedirs(out_dir, exist_ok=True)
    path_png = os.path.join(out_dir, f"{basename}.png")
    path_pdf = os.path.join(out_dir, f"{basename}.pdf")
    plt.savefig(path_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(path_pdf, bbox_inches="tight")
    print(f"Saved: {path_png} / .pdf")


# ------------------------------------------------------------------------
# 1. Gradient x Input  (Eq. 28)
# ------------------------------------------------------------------------

def gradient_input_importance(models, X_te, A_geo, cfg, n_samples: int = N_SAMPLES,
                               seed: int = 0):
    """Ensemble-averaged Gradient x Input saliency. Returns imp [H, N, F]."""
    rng = np.random.default_rng(seed)
    S = min(n_samples, len(X_te))
    idx = rng.choice(len(X_te), S, replace=False)
    X_np = X_te[idx]
    device = cfg.DEVICE
    horizon = cfg.HORIZON

    imp_per_model = []
    for model in models:
        model.eval()
        X_s = torch.from_numpy(X_np).float().to(device)
        X_s.requires_grad_(True)
        pred = model(X_s, A_geo, target=None, teacher_forcing_ratio=0.0)
        N, F = X_s.shape[2], X_s.shape[3]
        imp = np.zeros((horizon, N, F))
        for h in range(horizon):
            model.zero_grad()
            if X_s.grad is not None:
                X_s.grad.zero_()
            pred[:, h, :].sum().backward(retain_graph=(h < horizon - 1))
            gi = X_s.grad.detach().cpu().numpy() * X_s.detach().cpu().numpy()
            imp[h] = np.abs(gi).mean(axis=(0, 1))
        imp_per_model.append(imp)

    return np.mean(imp_per_model, axis=0)   # [H, N, F]


def plot_grad_importance(imp, feat_cols, cities, target_col, out_dir):
    _set_style()
    mask = _non_target_mask(feat_cols, target_col)
    F_arr = np.array(feat_cols)
    mean = imp.mean(axis=(0, 1))
    mean_masked = mean.copy(); mean_masked[~mask] = -np.inf
    order = np.argsort(mean_masked)[::-1]
    horizon = imp.shape[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    k = min(TOP_K, len(order))
    top = order[:k]
    pal = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, k))
    ax.barh(range(k), mean[top][::-1], color=pal[::-1], edgecolor="k", lw=0.4)
    ax.set_yticks(range(k)); ax.set_yticklabels(F_arr[top][::-1])
    ax.set_xlabel("Mean |Gradient x Input|")
    ax.set_title(f"Feature Importance -- Gradient x Input\n({len(cities)} cities, all horizons)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(out_dir, "fig8a_grad_bar")
    plt.close()

    top15 = order[:15]
    hm_data = imp[0][:, top15]
    hm_n = hm_data / hm_data.max(axis=1, keepdims=True).clip(min=1e-9)
    fig, ax = plt.subplots(figsize=(13, max(5, len(cities) * 0.55)))
    sns.heatmap(hm_n, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=F_arr[top15], yticklabels=cities, ax=ax, linewidths=0.3,
                cbar_kws={"label": "Normalised importance"})
    ax.set_title(f"Feature Importance by City -- +1 Day Horizon\n(row-normalised, {len(cities)} cities)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    _savefig(out_dir, "fig10_grad_heatmap_percity")
    plt.close()

    top5 = order[:5]
    fig, axes = plt.subplots(1, horizon, figsize=(4 * horizon, 4), sharey=True, sharex=True)
    if horizon == 1:
        axes = [axes]
    for h, ax in enumerate(axes):
        vals = imp[h].mean(axis=0)[top5]
        ax.barh(range(5), vals[::-1], color=plt.cm.Blues(np.linspace(0.4, 0.9, 5))[::-1],
                edgecolor="k", lw=0.4)
        ax.set_yticks(range(5)); ax.set_yticklabels(F_arr[top5][::-1])
        ax.set_title(f"+{h + 1} day"); ax.set_xlabel("|Grad x Input|")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle(f"Top-5 Feature Importance per Forecast Horizon ({len(cities)} Cities)")
    plt.tight_layout()
    _savefig(out_dir, "fig8c_grad_by_horizon")
    plt.close()
    return order


# ------------------------------------------------------------------------
# 2. Temporal Sensitivity  (Eq. 29)
# ------------------------------------------------------------------------

def temporal_sensitivity(models, X_te, A_geo, cfg, n_samples: int = N_SAMPLES, seed: int = 1):
    rng = np.random.default_rng(seed)
    S = min(n_samples, len(X_te))
    idx = rng.choice(len(X_te), S, replace=False)
    X_np = X_te[idx]
    device = cfg.DEVICE
    horizon, lookback = cfg.HORIZON, cfg.LOOKBACK

    ts_per_model = []
    for model in models:
        model.eval()
        X_s = torch.from_numpy(X_np).float().to(device)
        X_s.requires_grad_(True)
        pred = model(X_s, A_geo, target=None, teacher_forcing_ratio=0.0)
        ts_imp = np.zeros((horizon, lookback))
        for h in range(horizon):
            model.zero_grad()
            if X_s.grad is not None:
                X_s.grad.zero_()
            pred[:, h, :].sum().backward(retain_graph=(h < horizon - 1))
            grad = X_s.grad.detach().cpu().numpy()
            ts_imp[h] = np.linalg.norm(grad, axis=(2, 3)).mean(axis=0)
        ts_per_model.append(ts_imp)

    return np.mean(ts_per_model, axis=0)   # [H, LOOKBACK]


def plot_temporal_sensitivity(ts_imp, lookback: int, out_dir: str):
    _set_style()
    horizon = ts_imp.shape[0]
    days = [f"t-{lookback - t}" for t in range(lookback)]
    hlabels = [f"+{h + 1}d" for h in range(horizon)]
    ts_n = ts_imp / (ts_imp.max(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    sns.heatmap(ts_n, cmap="magma", xticklabels=days, yticklabels=hlabels,
                ax=ax, cbar_kws={"label": "Normalised gradient norm"})
    ax.set_xlabel("Lookback day (relative to forecast origin)")
    ax.set_ylabel("Forecast horizon")
    ax.set_title("Temporal Sensitivity -- Which Past Days Drive Each Forecast?")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    _savefig(out_dir, "fig11a_temporal_heatmap")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 4))
    pal = sns.color_palette("tab10", horizon)
    for h in range(horizon):
        ax.plot(range(lookback), ts_n[h], label=hlabels[h], color=pal[h], lw=1.8, marker="o", ms=3)
    ax.set_xticks(range(lookback))
    ax.set_xticklabels(days, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Lookback day"); ax.set_ylabel("Normalised sensitivity")
    ax.set_title("Temporal Sensitivity by Forecast Horizon")
    ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout()
    _savefig(out_dir, "fig11b_temporal_lines")
    plt.close()


# ------------------------------------------------------------------------
# 3. Permutation Feature Importance  (Eq. 30)
# ------------------------------------------------------------------------

def permutation_feature_importance(models, X_te, y_te_orig, A_geo, scaler,
                                    feat_cols, target_idx, cfg, n_repeats: int = N_REPEATS,
                                    seed: int = 2):
    rng = np.random.default_rng(seed)
    device = cfg.DEVICE
    base_pred = _ensemble_predict_np(models, X_te, A_geo, device)
    base_orig = np.clip(inverse_target(base_pred, scaler, target_idx), 0, None)
    base_mae = mean_absolute_error(y_te_orig.ravel(), base_orig.ravel())
    print(f"  Baseline MAE: {base_mae:.3f}")

    F_dim = len(feat_cols)
    pfi = np.zeros(F_dim)
    pfi_std = np.zeros(F_dim)

    for f in tqdm(range(F_dim), desc="  PFI"):
        maes = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            Xp[:, :, :, f] = X_te[rng.permutation(len(X_te)), :, :, f]
            pred_p = _ensemble_predict_np(models, Xp, A_geo, device)
            orig_p = np.clip(inverse_target(pred_p, scaler, target_idx), 0, None)
            maes.append(mean_absolute_error(y_te_orig.ravel(), orig_p.ravel()))
        pfi[f] = np.mean(maes) - base_mae
        pfi_std[f] = np.std(maes)

    return pfi, pfi_std


def plot_pfi(pfi, pfi_std, feat_cols, target_col, out_dir):
    _set_style()
    mask = _non_target_mask(feat_cols, target_col)
    F_arr = np.array(feat_cols)
    pfi_masked = pfi.copy(); pfi_masked[~mask] = -np.inf
    k = min(TOP_K, len(pfi_masked))
    order = np.argsort(pfi_masked)[::-1][:k]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    pal = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, k))
    ax.barh(range(k), pfi[order][::-1], xerr=pfi_std[order][::-1],
            color=pal[::-1], edgecolor="k", lw=0.4, capsize=3, error_kw={"elinewidth": 1})
    ax.set_yticks(range(k)); ax.set_yticklabels(F_arr[order][::-1])
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("MAE increase when feature is shuffled")
    ax.set_title(f"Permutation Feature Importance -- Top {TOP_K}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(out_dir, "fig8b_pfi")
    plt.close()


# ------------------------------------------------------------------------
# 4. KernelSHAP
# ------------------------------------------------------------------------

def kernel_shap_importance(models, X_te, A_geo, feat_cols, cfg, lookback: int,
                            n_background: int = 50, n_explain: int = 80,
                            nsamples: int = 100, seed: int = 3):
    if not HAS_SHAP:
        print("  Skipping KernelSHAP (shap not installed).")
        return None, None

    device = cfg.DEVICE
    rng = np.random.default_rng(seed)
    X_summary = X_te.mean(axis=(1, 2))   # [S, F] temporal/spatial collapse
    n_avail = len(X_summary)
    if n_avail < n_background + 1:
        print(f"  [warn] only {n_avail} test windows available -- reducing "
              f"n_background from {n_background} to {max(1, n_avail // 2)} "
              f"(sampling with replacement).")
        n_background = max(1, n_avail // 2)
        n_explain = min(n_explain, n_avail)
        replace = True
    else:
        replace = False
    bg = X_summary[rng.choice(n_avail, n_background, replace=replace)]
    ex = X_summary[rng.choice(n_avail, min(n_explain, n_avail), replace=replace)]

    def model_fn(x_flat):
        n = len(x_flat)
        x_bc = x_flat[:, None, None, :] * np.ones((n, lookback, X_te.shape[2], 1), dtype=np.float32)
        pred = _ensemble_predict_np(models, x_bc, A_geo, device)
        return pred[:, 0, :].mean(axis=1)   # +1-day forecast, city-averaged

    print("  Running KernelSHAP (ensemble-averaged, this can take a few minutes)...")
    explainer = shap.KernelExplainer(model_fn, bg)
    shap_vals = explainer.shap_values(ex, nsamples=nsamples, silent=True)
    return shap_vals, ex


def plot_shap(shap_vals, ex, feat_cols, target_col, out_dir):
    if shap_vals is None:
        return
    _set_style()
    F_arr = np.array(feat_cols)
    mask = _non_target_mask(feat_cols, target_col)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_abs_masked = mean_abs.copy(); mean_abs_masked[~mask] = -np.inf
    order = np.argsort(mean_abs_masked)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    top15 = order[:15]
    pal = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, 15))
    ax.barh(range(15), mean_abs[top15][::-1], color=pal[::-1], edgecolor="k", lw=0.4)
    ax.set_yticks(range(15)); ax.set_yticklabels(F_arr[top15][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance -- +1 Day Forecast")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(out_dir, "fig8d_shap_bar")
    plt.close()

    top10 = order[:10]
    fig, ax = plt.subplots(figsize=(9, 6))
    for rank, fi in enumerate(top10[::-1]):
        sv = shap_vals[:, fi]
        fv = ex[:, fi]
        norm_fv = (fv - fv.min()) / (np.ptp(fv) + 1e-9)
        ax.scatter(sv, rank + np.random.uniform(-0.25, 0.25, len(sv)),
                   c=plt.cm.coolwarm(norm_fv), s=15, alpha=0.7, rasterized=True)
    ax.set_yticks(range(10)); ax.set_yticklabels(F_arr[top10][::-1])
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("SHAP value (impact on +1d forecast)")
    ax.set_title("SHAP Beeswarm -- Top 10 Features\n(blue=low feature value, red=high)")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    _savefig(out_dir, "fig9_shap_beeswarm")
    plt.close()


# ------------------------------------------------------------------------
# 5. Spatial Influence (occlusion)  (Eq. 31)
# ------------------------------------------------------------------------

def spatial_influence(models, X_te, A_geo, cfg, n_samples: int = N_SAMPLES, seed: int = 4):
    rng = np.random.default_rng(seed)
    device = cfg.DEVICE
    N = X_te.shape[2]
    S = min(n_samples, len(X_te))
    idx = rng.choice(len(X_te), S, replace=False)
    X_sub = X_te[idx]

    base = _ensemble_predict_np(models, X_sub, A_geo, device)[:, 0, :]   # [S, N]
    mat = np.zeros((N, N))
    for src in tqdm(range(N), desc="  Spatial occlusion"):
        Xm = X_sub.copy()
        Xm[:, :, src, :] = 0.0
        masked = _ensemble_predict_np(models, Xm, A_geo, device)[:, 0, :]
        mat[src, :] = np.abs(masked - base).mean(axis=0)
    return mat


def plot_spatial_influence(inf_mat, cities, A_np, out_dir):
    _set_style()
    N = len(cities)

    fig, ax = plt.subplots(figsize=(max(8, N * 0.75), max(6, N * 0.65)))
    sns.heatmap(inf_mat, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=cities, yticklabels=cities, ax=ax, linewidths=0.3,
                cbar_kws={"label": "Mean |Delta PM2.5| (+1d)"})
    ax.set_xlabel("Target city"); ax.set_ylabel("Masked source city")
    ax.set_title(f"Spatial Influence: Impact of Masking Each City (+1d, {N} cities)")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    _savefig(out_dir, "fig12a_spatial_heatmap")
    plt.close()

    adj_v, inf_v = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                adj_v.append(A_np[i, j])
                inf_v.append(inf_mat[i, j])
    adj_v = np.array(adj_v); inf_v = np.array(inf_v)
    pcc, pval = pearsonr(adj_v, inf_v)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(adj_v, inf_v, s=30, alpha=0.6, c=inf_v, cmap="YlOrRd",
                     edgecolors="grey", lw=0.3)
    plt.colorbar(sc, ax=ax, label="Influence |Delta PM2.5|")
    z = np.polyfit(adj_v, inf_v, 1)
    xl = np.linspace(adj_v.min(), adj_v.max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), "k--", lw=1.5, label=f"Trend  PCC={pcc:.3f} (p={pval:.1e})")
    ax.set_xlabel("Graph Adjacency Weight")
    ax.set_ylabel("Occlusion Influence (mean |Delta PM2.5|)")
    ax.set_title("Does Graph Topology Reflect Real Spatial Influence?")
    ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout()
    _savefig(out_dir, "fig12b_adjacency_vs_influence")
    plt.close()
    print(f"  Adjacency-Influence Pearson r = {pcc:.4f} (p = {pval:.2e})")
    return pcc, pval


# ------------------------------------------------------------------------
# Combined summary figure
# ------------------------------------------------------------------------

def plot_summary(grad_imp, ts_imp, pfi, inf_mat, feat_cols, cities, target_col,
                  lookback: int, out_dir: str):
    _set_style()
    F_arr = np.array(feat_cols)
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    mask_s = _non_target_mask(feat_cols, target_col)
    mean_g = grad_imp.mean(axis=(0, 1))
    mg_mask = mean_g.copy(); mg_mask[~mask_s] = -np.inf
    top_a = np.argsort(mg_mask)[::-1][:10]
    pal_a = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 10))
    ax_a.barh(range(10), mean_g[top_a][::-1], color=pal_a[::-1], edgecolor="k", lw=0.4)
    ax_a.set_yticks(range(10)); ax_a.set_yticklabels(F_arr[top_a][::-1])
    ax_a.set_xlabel("|Gradient x Input|")
    ax_a.set_title("(A) Gradient x Input Feature Importance")
    ax_a.grid(axis="x", alpha=0.3)

    ax_b = fig.add_subplot(gs[0, 1])
    horizon = ts_imp.shape[0]
    ts_n = ts_imp / (ts_imp.max(axis=1, keepdims=True) + 1e-9)
    days = [f"t-{lookback - t}" for t in range(lookback)]
    pal_b = sns.color_palette("tab10", horizon)
    for h in range(horizon):
        ax_b.plot(range(lookback), ts_n[h], label=f"+{h + 1}d", color=pal_b[h], lw=1.8, marker="o", ms=3)
    step = max(1, lookback // 7)
    ax_b.set_xticks(range(0, lookback, step))
    ax_b.set_xticklabels([days[i] for i in range(0, lookback, step)], rotation=30, ha="right", fontsize=9)
    ax_b.set_xlabel("Lookback day"); ax_b.set_ylabel("Norm. sensitivity")
    ax_b.set_title("(B) Temporal Sensitivity by Horizon")
    ax_b.legend(fontsize=8); ax_b.grid(alpha=0.25)

    ax_c = fig.add_subplot(gs[1, 0])
    pfi_m2 = pfi.copy(); pfi_m2[~mask_s] = -np.inf
    top_c = np.argsort(pfi_m2)[::-1][:10]
    pal_c = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, 10))
    ax_c.barh(range(10), pfi[top_c][::-1], color=pal_c[::-1], edgecolor="k", lw=0.4)
    ax_c.set_yticks(range(10)); ax_c.set_yticklabels(F_arr[top_c][::-1])
    ax_c.axvline(0, color="k", lw=0.8, ls="--")
    ax_c.set_xlabel("MAE increase")
    ax_c.set_title("(C) Permutation Feature Importance")
    ax_c.grid(axis="x", alpha=0.3)

    ax_d = fig.add_subplot(gs[1, 1])
    short = [c[:8] for c in cities]
    im = ax_d.imshow(inf_mat, cmap="YlOrRd", aspect="auto")
    N_ = len(cities)
    ax_d.set_xticks(range(N_)); ax_d.set_yticks(range(N_))
    ax_d.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax_d.set_yticklabels(short, fontsize=8)
    ax_d.set_xlabel("Target city"); ax_d.set_ylabel("Masked source city")
    ax_d.set_title("(D) Spatial Influence (Occlusion)")
    plt.colorbar(im, ax=ax_d, fraction=0.046, label="|Delta PM2.5|")

    plt.suptitle(f"Explainable AI Summary -- PM2.5 Spatiotemporal GNN ({N_} cities)",
                 fontsize=14, fontweight="bold", y=1.01)
    _savefig(out_dir, "fig_xai_summary")
    plt.close()


# ------------------------------------------------------------------------
# Method agreement (Spearman rank correlation + top-k Jaccard overlap)
# ------------------------------------------------------------------------

def method_agreement(grad_scores: dict, pfi_scores: dict, shap_scores: dict,
                      top_k: int = 10, out_dir: str = RESULTS_DIR):
    """Quantifies agreement between the three feature-attribution methods
    beyond eyeballing bar charts: Spearman rank correlation over the
    shared feature set, plus top-k Jaccard overlap."""
    methods = {"GradInput": grad_scores, "PFI": pfi_scores, "SHAP": shap_scores}
    shared_feats = sorted(set(grad_scores) & set(pfi_scores) & set(shap_scores))
    if not shared_feats:
        raise ValueError("No overlapping features across the three score dicts.")

    rank_vecs = {name: np.array([scores[f] for f in shared_feats])
                 for name, scores in methods.items()}
    names = list(methods.keys())
    corr = pd.DataFrame(index=names, columns=names, dtype=float)
    pval = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            rho, p = spearmanr(rank_vecs[a], rank_vecs[b])
            corr.loc[a, b] = rho
            pval.loc[a, b] = p

    top_sets = {}
    for name, scores in methods.items():
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_sets[name] = set(f for f, _ in ranked[:top_k])
    overlap = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            inter = len(top_sets[a] & top_sets[b])
            union = len(top_sets[a] | top_sets[b])
            overlap.loc[a, b] = inter / union if union else np.nan

    print("\nSpearman rank correlation between attribution methods:")
    print(corr.round(3).to_string())
    print(f"\nTop-{top_k} feature-set Jaccard overlap:")
    print(overlap.round(3).to_string())

    os.makedirs(out_dir, exist_ok=True)
    corr.to_csv(os.path.join(out_dir, "xai_method_rank_correlation.csv"))
    overlap.to_csv(os.path.join(out_dir, f"xai_method_top{top_k}_overlap.csv"))
    return corr, pval, overlap


# ------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------

def save_tables(feat_cols, grad_imp, pfi, pfi_std, shap_vals, inf_mat, cities,
                 out_dir: str = RESULTS_DIR):
    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "grad_input_mean": grad_imp.mean(axis=(0, 1)),
        "pfi_mae_increase": pfi,
        "pfi_std": pfi_std,
        "shap_mean_abs": (np.abs(shap_vals).mean(axis=0) if shap_vals is not None
                           else np.full(len(feat_cols), np.nan)),
    }).sort_values("grad_input_mean", ascending=False)
    os.makedirs(out_dir, exist_ok=True)
    imp_path = os.path.join(out_dir, "xai_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"Saved: {imp_path}")

    inf_path = os.path.join(out_dir, "xai_spatial_influence.csv")
    pd.DataFrame(inf_mat, index=cities, columns=cities).to_csv(inf_path)
    print(f"Saved: {inf_path}")
    return imp_df


# ------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------

def run(cfg=default_cfg, ckpt_dir: str = CKPT_DIR, out_dir: str = OUT_DIR,
        n_samples: int = N_SAMPLES, n_repeats: int = N_REPEATS, top_k: int = 10):
    print("=" * 70)
    print("EXPLAINABILITY SUITE -- GD-GRU ENSEMBLE")
    print("=" * 70)

    (tr, va, te, scaler, feat_cols, tgt_idx,
     city_meta, cities, N, dates) = load_and_preprocess(cfg)
    A_np, _ = build_adjacency(city_meta, cfg)
    A_geo = torch.from_numpy(A_np).to(cfg.DEVICE)

    X_te, y_te = make_windows(te, cfg.LOOKBACK, cfg.HORIZON)
    y_te_orig = np.clip(inverse_target(y_te, scaler, tgt_idx), 0, None)
    print(f"Test windows: {len(X_te)}")

    feat_dim = len(feat_cols)
    models = load_ensemble(cfg, feat_dim, N, ckpt_dir=ckpt_dir)

    print("\n[1/5] Gradient x Input Feature Importance...")
    grad_imp = gradient_input_importance(models, X_te, A_geo, cfg, n_samples=n_samples)
    grad_order = plot_grad_importance(grad_imp, feat_cols, cities, cfg.TARGET_COL, out_dir)

    print("\n[2/5] Temporal Sensitivity...")
    ts_imp = temporal_sensitivity(models, X_te, A_geo, cfg, n_samples=n_samples)
    plot_temporal_sensitivity(ts_imp, cfg.LOOKBACK, out_dir)

    print("\n[3/5] Permutation Feature Importance...")
    pfi, pfi_std = permutation_feature_importance(
        models, X_te, y_te_orig, A_geo, scaler, feat_cols, tgt_idx, cfg, n_repeats=n_repeats)
    plot_pfi(pfi, pfi_std, feat_cols, cfg.TARGET_COL, out_dir)

    print("\n[4/5] KernelSHAP...")
    shap_vals, X_exp = kernel_shap_importance(models, X_te, A_geo, feat_cols, cfg, cfg.LOOKBACK)
    plot_shap(shap_vals, X_exp, feat_cols, cfg.TARGET_COL, out_dir)

    print("\n[5/5] Spatial Influence (Occlusion)...")
    inf_mat = spatial_influence(models, X_te, A_geo, cfg, n_samples=n_samples)
    pcc, pval = plot_spatial_influence(inf_mat, cities, A_np, out_dir)

    print("\n[+] Summary figure...")
    plot_summary(grad_imp, ts_imp, pfi, inf_mat, feat_cols, cities, cfg.TARGET_COL,
                 cfg.LOOKBACK, out_dir)

    print("\n[+] Saving tables...")
    imp_df = save_tables(feat_cols, grad_imp, pfi, pfi_std, shap_vals, inf_mat, cities, RESULTS_DIR)

    corr = overlap = None
    if shap_vals is not None:
        print("\n[+] Method agreement (Gradient x Input / PFI / SHAP)...")
        non_target = imp_df[imp_df["feature"] != cfg.TARGET_COL].set_index("feature")
        grad_scores = non_target["grad_input_mean"].to_dict()
        pfi_scores = non_target["pfi_mae_increase"].to_dict()
        shap_scores = non_target["shap_mean_abs"].dropna().to_dict()
        corr, _, overlap = method_agreement(grad_scores, pfi_scores, shap_scores,
                                             top_k=top_k, out_dir=RESULTS_DIR)
    else:
        print("\n[info] Skipping method agreement (SHAP unavailable).")

    print("\n" + "=" * 70)
    print("EXPLAINABILITY COMPLETE")
    print(f"Adjacency-influence Pearson r = {pcc:.4f} (p = {pval:.2e})")
    print("=" * 70)

    return dict(grad_imp=grad_imp, ts_imp=ts_imp, pfi=pfi, pfi_std=pfi_std,
                shap_vals=shap_vals, inf_mat=inf_mat, adj_influence_pcc=pcc,
                adj_influence_p=pval, method_corr=corr, method_overlap=overlap,
                importance_table=imp_df)


if __name__ == "__main__":
    run()
