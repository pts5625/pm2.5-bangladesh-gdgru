"""
Alternative graph construction comparison (manuscript Table S4).

Retrains the 5-seed GD-GRU ensemble under five graph constructions, all
using the same data split and training protocol as the main model:

  gaussian        the adopted graph throughout the manuscript
                   (Haversine distance -> Gaussian RBF kernel -> symmetric
                   normalisation; see data.build_adjacency)
  knn             the same kernel restricted to each node's K nearest
                   neighbours
  wind_informed   edge weight scaled by the alignment between each city's
                   prevailing wind direction (training-split circular mean)
                   and the inter-city bearing, then symmetrised
  hybrid          convex combination beta * gaussian + (1-beta) * wind_informed
  adaptive        a fully learnable adjacency (sigmoid(logits), symmetrised,
                   degree-normalised, L1-regularised) trained jointly with
                   the model, warm-started from the Gaussian kernel weights

The wind_informed / hybrid variants are skipped automatically if
``cfg.WIND_SPEED_COL`` / ``cfg.WIND_DIR_COL`` are not present in the data.

Usage
-----
    python -m pm25gdgru.analysis.graph_comparison
"""

import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from ..config import cfg as default_cfg, set_seed
from ..data import (load_and_preprocess, build_adjacency, make_windows,
                     haversine_matrix, STDataset, mixup_batch, inverse_target)
from ..models import GDGRUNet
from ..engine import train_ensemble, EarlyStopping, predict
from ..losses import build_horizon_weights, weighted_huber_loss
from ..metrics import compute_metrics
from .common import load_existing, append_row, flatten_metrics

RESULTS_DIR = os.environ.get("PM25_RESULTS_DIR", "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints", "graph_comparison")
CSV_PATH = os.path.join(RESULTS_DIR, "table_s4_graph_comparison.csv")


# ------------------------------------------------------------------------
# Alternative graph constructions
# ------------------------------------------------------------------------

def load_training_wind_stats(cfg):
    """Per-city circular-mean wind direction and mean wind speed, computed
    from the TRAINING split only (to avoid leaking val/test info into the
    fixed graph). Returns None if the wind columns aren't in the data."""
    df = pd.read_csv(cfg.DATA_PATH, parse_dates=[cfg.DATE_COL])
    df = df.sort_values([cfg.DATE_COL, cfg.CITY_COL]).reset_index(drop=True)

    if cfg.WIND_SPEED_COL not in df.columns or cfg.WIND_DIR_COL not in df.columns:
        print(f"[info] '{cfg.WIND_SPEED_COL}' / '{cfg.WIND_DIR_COL}' not found in "
              f"{cfg.DATA_PATH} -- wind_informed / hybrid graph variants will be skipped.")
        return None

    dates = sorted(df[cfg.DATE_COL].unique())
    n_train = int(len(dates) * cfg.TRAIN_RATIO)
    train_dates = set(dates[:n_train])
    df_tr = df[df[cfg.DATE_COL].isin(train_dates)]

    stats = {}
    for city, g in df_tr.groupby(cfg.CITY_COL):
        theta = np.radians(g[cfg.WIND_DIR_COL].values)
        u, v = np.sin(theta).mean(), np.cos(theta).mean()
        stats[city] = {"mean_dir_rad": np.arctan2(u, v) % (2 * np.pi),
                        "mean_speed": float(g[cfg.WIND_SPEED_COL].mean())}
    return stats


def _bearing_matrix(city_meta, cfg):
    lats = np.radians(city_meta[cfg.LAT_COL].values)
    lons = np.radians(city_meta[cfg.LON_COL].values)
    N = len(city_meta)
    bearing = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dlon = lons[j] - lons[i]
            x = np.sin(dlon) * np.cos(lats[j])
            y = (np.cos(lats[i]) * np.sin(lats[j])
                 - np.sin(lats[i]) * np.cos(lats[j]) * np.cos(dlon))
            bearing[i, j] = np.arctan2(x, y) % (2 * np.pi)
    return bearing


def build_wind_adjacency(city_meta, cfg, wind_stats):
    """Directional, wind-informed graph. Edge weight i->j increases when
    city i's prevailing wind blows TOWARD j, scaled by geographic
    proximity, then symmetrised for compatibility with the symmetric
    GDConv formulation. Returns (normalised A, symmetric raw weights)."""
    cities = city_meta[cfg.CITY_COL].tolist()
    N = len(cities)
    dist = haversine_matrix(city_meta, cfg)
    sigma = np.median(dist[dist > 0])
    W_geo = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    bearing = _bearing_matrix(city_meta, cfg)

    W_dir = np.zeros((N, N))
    for i, ci in enumerate(cities):
        if ci not in wind_stats:
            continue
        blows_toward = (wind_stats[ci]["mean_dir_rad"] + np.pi) % (2 * np.pi)
        for j in range(N):
            if i == j:
                continue
            ang_diff = np.abs((bearing[i, j] - blows_toward + np.pi) % (2 * np.pi) - np.pi)
            W_dir[i, j] = max(0.0, np.cos(ang_diff))

    W_wind = W_geo * W_dir
    W_wind_sym = (W_wind + W_wind.T) / 2.0
    np.fill_diagonal(W_wind_sym, 1.0)

    D_inv_sqrt = np.diag(1.0 / np.sqrt(W_wind_sym.sum(axis=1).clip(min=1e-6)))
    A_wind = (D_inv_sqrt @ W_wind_sym @ D_inv_sqrt).astype(np.float32)
    return A_wind, W_wind_sym


def build_hybrid_adjacency(city_meta, cfg, wind_stats, beta: float = 0.5):
    """Convex combination beta * geographic + (1-beta) * wind-informed."""
    dist = haversine_matrix(city_meta, cfg)
    sigma = np.median(dist[dist > 0])
    W_geo = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(W_geo, 1.0)

    _, W_wind_sym = build_wind_adjacency(city_meta, cfg, wind_stats)
    W_hybrid = beta * W_geo + (1 - beta) * W_wind_sym
    np.fill_diagonal(W_hybrid, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W_hybrid.sum(axis=1).clip(min=1e-6)))
    return (D_inv_sqrt @ W_hybrid @ D_inv_sqrt).astype(np.float32)


class LearnableAdjacency(nn.Module):
    """Fully-learnable graph: sigmoid(logits), symmetrised, degree-
    normalised the same way as the fixed graphs so results stay
    comparable. ``l1_penalty()`` discourages the graph from densifying
    beyond what a small (~12-node) network can support without overfitting."""

    def __init__(self, n_nodes: int, init_W: np.ndarray = None):
        super().__init__()
        if init_W is not None:
            init_W = np.clip(init_W, 1e-4, 1 - 1e-4)
            logits = np.log(init_W / (1 - init_W))
        else:
            logits = np.random.randn(n_nodes, n_nodes) * 0.1
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))
        self.n_nodes = n_nodes

    def forward(self):
        W = torch.sigmoid(self.logits)
        W = (W + W.T) / 2.0
        eye = torch.eye(self.n_nodes, device=W.device)
        W = W * (1 - eye) + eye
        deg = W.sum(dim=1).clamp(min=1e-6)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
        return D_inv_sqrt @ W @ D_inv_sqrt, W

    def l1_penalty(self):
        eye = torch.eye(self.n_nodes, device=self.logits.device)
        return (torch.sigmoid(self.logits) * (1 - eye)).abs().mean()


def train_adaptive_ensemble(X_tr, y_tr, X_va, y_va, X_te, y_te, init_W_np,
                             feat_dim, N, cfg, seeds, scaler, tgt_idx,
                             ckpt_dir: str, l1_lambda: float = 1e-4,
                             tag: str = "adaptive", verbose: bool = False):
    """Same optimiser / LR schedule / early stopping as
    ``engine.train_one_seed``, except the adjacency is a joint learnable
    parameter (``LearnableAdjacency``) trained alongside the model."""
    os.makedirs(ckpt_dir, exist_ok=True)
    horiz_w = build_horizon_weights(cfg)
    te_ld = torch.utils.data.DataLoader(STDataset(X_te, y_te), cfg.BATCH_SIZE, shuffle=False)

    preds_sc_per_seed, true_sc = [], None
    for seed in seeds:
        model_path = os.path.join(ckpt_dir, f"{tag}_seed{seed}_model.pt")
        adj_path = os.path.join(ckpt_dir, f"{tag}_seed{seed}_adj.pt")

        if os.path.exists(model_path) and os.path.exists(adj_path):
            model = GDGRUNet(feat_dim, N, cfg).to(cfg.DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
            adj_module = LearnableAdjacency(N).to(cfg.DEVICE)
            adj_module.load_state_dict(torch.load(adj_path, map_location=cfg.DEVICE))
            print(f"  [cache hit] {tag} seed {seed}")
        else:
            set_seed(seed)
            tr_ld = torch.utils.data.DataLoader(STDataset(X_tr, y_tr), cfg.BATCH_SIZE, shuffle=True)
            va_ld = torch.utils.data.DataLoader(STDataset(X_va, y_va), cfg.BATCH_SIZE, shuffle=False)

            model = GDGRUNet(feat_dim, N, cfg).to(cfg.DEVICE)
            adj_module = LearnableAdjacency(N, init_W=init_W_np).to(cfg.DEVICE)
            params = list(model.parameters()) + list(adj_module.parameters())
            optimizer = torch.optim.AdamW(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE, min_lr=cfg.MIN_LR)
            stopper = EarlyStopping(cfg.PATIENCE, cfg.MIN_DELTA, cfg.MIN_EPOCHS,
                                     cfg.VAL_SMOOTH, path=model_path)
            best_smoothed = np.inf

            for epoch in range(1, cfg.EPOCHS + 1):
                model.train(); adj_module.train()
                tf_ratio = max(0.0, cfg.TF_INITIAL * (1.0 - epoch / cfg.EPOCHS))
                for X_b, y_b in tr_ld:
                    X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
                    if cfg.NOISE_STD > 0:
                        X_b = X_b + torch.randn_like(X_b) * cfg.NOISE_STD
                    X_b, y_b = mixup_batch(X_b, y_b, cfg.MIXUP_ALPHA)
                    A_dyn, _ = adj_module()
                    optimizer.zero_grad()
                    pred = model(X_b, A_dyn, target=y_b, teacher_forcing_ratio=tf_ratio)
                    loss = weighted_huber_loss(pred, y_b, horiz_w) + l1_lambda * adj_module.l1_penalty()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 3.0)
                    optimizer.step()

                model.eval(); adj_module.eval()
                va_total = 0
                with torch.no_grad():
                    A_dyn, _ = adj_module()
                    for X_b, y_b in va_ld:
                        X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
                        pred = model(X_b, A_dyn, target=None, teacher_forcing_ratio=0.0)
                        va_total += Fnn.huber_loss(pred, y_b, delta=1.0).item() * len(X_b)
                va_loss = va_total / len(va_ld.dataset)
                scheduler.step(va_loss)
                stopper(va_loss, model, epoch)
                if stopper.smoothed is not None and stopper.smoothed < best_smoothed:
                    best_smoothed = stopper.smoothed
                    torch.save(adj_module.state_dict(), adj_path)
                if verbose and (epoch % 30 == 0 or stopper.stop):
                    print(f"    [{tag} seed {seed}] ep {epoch:4d} val={va_loss:.4f} "
                          f"smooth={stopper.smoothed:.4f}")
                if stopper.stop:
                    break

            model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
            adj_module.load_state_dict(torch.load(adj_path, map_location=cfg.DEVICE))

        model.eval(); adj_module.eval()
        with torch.no_grad():
            A_final, _ = adj_module()
            pred_sc, true_sc = predict(model, te_ld, A_final, cfg.DEVICE)
        preds_sc_per_seed.append(pred_sc)

    pred_sc_ens = np.mean(preds_sc_per_seed, axis=0)
    pred_orig = np.clip(inverse_target(pred_sc_ens, scaler, tgt_idx), 0, None)
    true_orig = inverse_target(true_sc, scaler, tgt_idx)
    overall = compute_metrics(true_orig, pred_orig, cfg.MAPE_FLOOR)
    return overall


# ------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------

def run(cfg=default_cfg, ckpt_dir: str = CKPT_DIR, csv_path: str = CSV_PATH,
        wind_beta: float = 0.5, adaptive_l1: float = 1e-4, knn_k: int = None):
    print("=" * 70)
    print("GRAPH CONSTRUCTION COMPARISON (Table S4)")
    print("=" * 70)

    (tr, va, te, scaler, feat_cols, tgt_idx,
     city_meta, cities, N, dates) = load_and_preprocess(cfg)
    X_tr, y_tr = make_windows(tr, cfg.LOOKBACK, cfg.HORIZON)
    X_va, y_va = make_windows(va, cfg.LOOKBACK, cfg.HORIZON)
    X_te, y_te = make_windows(te, cfg.LOOKBACK, cfg.HORIZON)
    feat_dim = len(feat_cols)

    variants = {}

    cfg_g = copy.deepcopy(cfg); cfg_g.ADJ_MODE = "gaussian"
    A_g, _ = build_adjacency(city_meta, cfg_g)
    variants["gaussian"] = (cfg_g, torch.from_numpy(A_g).to(cfg.DEVICE))

    cfg_k = copy.deepcopy(cfg); cfg_k.ADJ_MODE = "knn"
    if knn_k is not None:
        cfg_k.KNN_K = knn_k
    A_k, _ = build_adjacency(city_meta, cfg_k)
    variants["knn"] = (cfg_k, torch.from_numpy(A_k).to(cfg.DEVICE))

    wind_stats = load_training_wind_stats(cfg)
    if wind_stats is not None:
        A_w, _ = build_wind_adjacency(city_meta, cfg, wind_stats)
        variants["wind_informed"] = (cfg, torch.from_numpy(A_w).to(cfg.DEVICE))
        A_h = build_hybrid_adjacency(city_meta, cfg, wind_stats, beta=wind_beta)
        variants["hybrid"] = (cfg, torch.from_numpy(A_h).to(cfg.DEVICE))

    done = load_existing(csv_path)
    for tag, (cfg_v, A_geo) in variants.items():
        if tag in done:
            print(f"[skip] {tag} already in {csv_path}")
            continue
        print(f"\n--- GRAPH VARIANT: {tag} ---")
        result = train_ensemble(
            GDGRUNet, dict(in_features=feat_dim, n_nodes=N, cfg=cfg_v),
            X_tr, y_tr, X_va, y_va, X_te, y_te, A_geo, cfg_v,
            cfg.ENSEMBLE_SEEDS, scaler, tgt_idx, ckpt_dir=ckpt_dir, tag=tag, verbose=False)
        row = dict(key=tag, graph=tag)
        row.update(flatten_metrics(result.overall))
        append_row(csv_path, row)
        print(f"  MAE={result.overall['MAE']:.3f}  R2={result.overall['R2']:.4f}")

    if "adaptive" not in done:
        print("\n--- GRAPH VARIANT: adaptive (fully learnable, L1-regularised) ---")
        dist = haversine_matrix(city_meta, cfg)
        sigma0 = np.median(dist[dist > 0])
        W_init = np.exp(-(dist ** 2) / (2 * sigma0 ** 2))
        np.fill_diagonal(W_init, 1.0)
        overall = train_adaptive_ensemble(
            X_tr, y_tr, X_va, y_va, X_te, y_te, W_init, feat_dim, N, cfg,
            cfg.ENSEMBLE_SEEDS, scaler, tgt_idx, ckpt_dir=ckpt_dir,
            l1_lambda=adaptive_l1, tag="adaptive")
        row = dict(key="adaptive", graph="adaptive")
        row.update(flatten_metrics(overall))
        append_row(csv_path, row)
        print(f"  MAE={overall['MAE']:.3f}  R2={overall['R2']:.4f}")
    else:
        print("[skip] adaptive already in", csv_path)

    df = pd.read_csv(csv_path)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved -> {csv_path}")
    return df


if __name__ == "__main__":
    run()
