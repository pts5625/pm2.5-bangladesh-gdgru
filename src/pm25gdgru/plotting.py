"""
Publication-style figures for the main GD-GRU training run: training
curves, ensemble uncertainty, observed-vs-predicted scatter, per-horizon
metric bars, the per-city/per-horizon R2 heatmap, sample time series,
error analysis, and the adjacency matrix itself.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import Config
from .metrics import compute_metrics

DPI = 300


def _set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9, "figure.dpi": DPI,
    })


def plot_all(histories, preds_orig_per_seed, pred_orig, true_orig,
             horizon_metrics, city_metrics, cities, cfg: Config,
             A_np, out_dir: str = "results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    _set_style()
    PAL = sns.color_palette("tab10", max(cfg.HORIZON, len(cfg.ENSEMBLE_SEEDS)))

    def savefig(name):
        path = os.path.join(out_dir, name)
        plt.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {path}")

    # Fig 1 -- Training curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for i, hist in enumerate(histories):
        if not hist["train"]:
            continue
        ep = range(1, len(hist["train"]) + 1)
        axes[0].plot(ep, hist["train"], color=PAL[i], alpha=0.7, lw=1.2,
                     label=f"seed {cfg.ENSEMBLE_SEEDS[i]}")
        axes[1].plot(ep, hist["val"], color=PAL[i], alpha=0.7, lw=1.2,
                     label=f"seed {cfg.ENSEMBLE_SEEDS[i]}")
    for ax, title in zip(axes, ["Train Huber Loss", "Val Huber Loss"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss (scaled)")
        ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle("Ensemble Training and Validation Curves")
    plt.tight_layout()
    savefig("fig1_training_curves.png")
    plt.close()

    # Fig 2 -- Ensemble uncertainty (spread across seeds, original units)
    preds_stack = np.array(preds_orig_per_seed)
    pred_std = preds_stack.std(axis=0)
    fig, axes = plt.subplots(1, cfg.HORIZON, figsize=(4 * cfg.HORIZON, 4), sharey=True)
    if cfg.HORIZON == 1:
        axes = [axes]
    for h, ax in enumerate(axes):
        ax.hist(pred_std[:, h, :].ravel(), bins=40, color=PAL[h], alpha=0.75)
        ax.set_xlabel("Pred std (ug/m3)"); ax.set_title(f"+{h + 1} day")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Count")
    plt.suptitle("Ensemble Prediction Uncertainty")
    plt.tight_layout()
    savefig("fig2_ensemble_uncertainty.png")
    plt.close()

    # Fig 3 -- Scatter per horizon
    fig, axes = plt.subplots(1, cfg.HORIZON, figsize=(4 * cfg.HORIZON, 4),
                              sharey=True, sharex=True)
    if cfg.HORIZON == 1:
        axes = [axes]
    for h, ax in enumerate(axes):
        t = true_orig[:, h, :].ravel()
        p = pred_orig[:, h, :].ravel()
        ax.scatter(t, p, s=5, alpha=0.3, color=PAL[h], rasterized=True)
        lim = max(t.max(), p.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("Observed (ug/m3)")
        if h == 0:
            ax.set_ylabel("Predicted (ug/m3)")
        ax.set_title(f"+{h + 1} day\nR2={horizon_metrics[h]['R2']:.3f}, "
                     f"MAE={horizon_metrics[h]['MAE']:.2f}")
        ax.grid(alpha=0.2)
    plt.suptitle("Observed vs Predicted PM2.5 by Forecast Horizon", y=1.02)
    plt.tight_layout()
    savefig("fig3_scatter_horizon.png")
    plt.close()

    # Fig 4 -- Metric bars per horizon
    hlabels = [f"+{h + 1}d" for h in range(cfg.HORIZON)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, key, col in zip(axes, ["MAE", "RMSE", "MAPE (%)", "R2"],
                             ["#4393c3", "#d6604d", "#74c476", "#9970ab"]):
        vals = [horizon_metrics[h][key] for h in range(cfg.HORIZON)]
        bars = ax.bar(hlabels, vals, color=col, edgecolor="black", lw=0.5)
        ax.set_xlabel("Horizon"); ax.set_ylabel(key)
        ax.set_title(f"{key} vs Horizon"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.01, f"{v:.2f}",
                    ha="center", fontsize=8)
    plt.suptitle("Performance Metrics Across Forecast Horizons")
    plt.tight_layout()
    savefig("fig4_metric_horizon.png")
    plt.close()

    # Fig 5 -- R2 heatmap city x horizon
    r2_mat = np.array([[
        compute_metrics(true_orig[:, h, n], pred_orig[:, h, n], cfg.MAPE_FLOOR)["R2"]
        for n in range(len(cities))] for h in range(cfg.HORIZON)])
    fig, ax = plt.subplots(figsize=(max(9, len(cities) * 0.85), 4))
    sns.heatmap(r2_mat, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=cities, yticklabels=[f"+{h + 1}d" for h in range(cfg.HORIZON)],
                ax=ax, cbar_kws={"label": "R2"}, vmin=0, vmax=1)
    ax.set_xlabel("City"); ax.set_ylabel("Horizon")
    ax.set_title(f"R2 by City and Forecast Horizon ({len(cities)} Cities)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    savefig("fig5_r2_heatmap.png")
    plt.close()

    # Fig 6 -- Time series for up to 4 sample cities
    sample_cities = cities[:min(4, len(cities))]
    fig, axes = plt.subplots(len(sample_cities), 1, figsize=(14, 3.5 * len(sample_cities)))
    if len(sample_cities) == 1:
        axes = [axes]
    for ax, city in zip(axes, sample_cities):
        ni = cities.index(city)
        n_show = min(100, len(true_orig))
        tv = true_orig[-n_show:, 0, ni]
        pv = pred_orig[-n_show:, 0, ni]
        x_ax = range(n_show)
        ax.plot(x_ax, tv, label="Observed", color="#2166ac", lw=1.5)
        ax.plot(x_ax, pv, label="+1d forecast", color="#d6604d", lw=1.5, ls="--")
        ax.fill_between(x_ax, tv, pv, alpha=0.12, color="gray")
        ax.set_ylabel("PM2.5 (ug/m3)"); ax.set_title(city)
        ax.legend(loc="upper right"); ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Test sample index")
    plt.suptitle("Observed vs +1 Day Ensemble Forecast", y=1.01)
    plt.tight_layout()
    savefig("fig6_timeseries.png")
    plt.close()

    # Fig 7 -- Error analysis
    errors = (pred_orig - true_orig).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(errors, bins=60, color="#4393c3", edgecolor="none", alpha=0.85)
    axes[0].axvline(0, color="k", ls="--", lw=1.2)
    axes[0].set_xlabel("Error (ug/m3)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Forecast Error Distribution"); axes[0].grid(alpha=0.25)
    q25, q75 = np.percentile(true_orig, 25), np.percentile(true_orig, 75)
    grps = [errors[true_orig.ravel() < q25],
            errors[(true_orig.ravel() >= q25) & (true_orig.ravel() < q75)],
            errors[true_orig.ravel() >= q75]]
    bp = axes[1].boxplot(grps, labels=["Low\n(<Q1)", "Mid\n(Q1-Q3)", "High\n(>Q3)"],
                          patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#c6dbef")
    axes[1].axhline(0, color="k", ls="--", lw=1.2)
    axes[1].set_ylabel("Error (ug/m3)")
    axes[1].set_title("Error by PM2.5 Regime"); axes[1].grid(alpha=0.25)
    plt.suptitle("Forecast Error Analysis")
    plt.tight_layout()
    savefig("fig7_error_analysis.png")
    plt.close()

    # Fig 8 -- Adjacency matrix
    fig, ax = plt.subplots(figsize=(max(8, len(cities) * 0.75), max(6, len(cities) * 0.65)))
    sns.heatmap(A_np, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=cities, yticklabels=cities,
                ax=ax, cbar_kws={"label": "Adjacency weight"})
    ax.set_title(f"Spatial Adjacency Matrix -- {cfg.ADJ_MODE.capitalize()} "
                 f"({len(cities)} Cities)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    savefig("fig8_adjacency.png")
    plt.close()

    print("\nAll figures saved.")


def print_latex_tables(horizon_metrics, city_metrics, cfg: Config):
    print("\n" + "=" * 65)
    print("LaTeX TABLE - Per-Horizon Performance")
    print("=" * 65)
    print(r"\begin{table}[htbp]\centering")
    print(r"\caption{PM$_{2.5}$ forecast performance by horizon}")
    print(r"\label{tab:horizon}")
    print(r"\begin{tabular}{lccccc}\hline")
    print(r"Horizon & MAE & RMSE & MAPE (\%) & R$^{2}$ & PCC \\\hline")
    for h in range(cfg.HORIZON):
        m = horizon_metrics[h]
        print(f"+{h + 1} day & {m['MAE']:.3f} & {m['RMSE']:.3f} & "
              f"{m['MAPE (%)']:.2f} & {m['R2']:.4f} & {m['PCC']:.4f} \\\\")
    print(r"\hline\end{tabular}\end{table}")

    print("\n" + "=" * 65)
    print("LaTeX TABLE - Per-City Performance")
    print("=" * 65)
    print(r"\begin{table}[htbp]\centering")
    print(r"\caption{Per-city PM$_{2.5}$ performance (all horizons)}")
    print(r"\label{tab:city}")
    print(r"\begin{tabular}{lcccc}\hline")
    print(r"City & MAE & RMSE & MAPE (\%) & R$^{2}$ \\\hline")
    for city, m in city_metrics.items():
        print(f"{city} & {m['MAE']:.3f} & {m['RMSE']:.3f} & "
              f"{m['MAPE (%)']:.2f} & {m['R2']:.4f} \\\\")
    print(r"\hline\end{tabular}\end{table}")
