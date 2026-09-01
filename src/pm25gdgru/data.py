"""
Data loading, spatial graph construction, and windowing.

Every training and analysis script in this repository builds its train/val/
test arrays and adjacency matrix through this module, so a test window
produced for GD-GRU is guaranteed to match the one used for DCRNN, the
GRU-no-graph ablation, and every baseline -- which is required for the
paired significance tests in ``significance.py``.

The source CSV (``cfg.DATA_PATH``) is expected to already contain exactly
the cities and feature columns intended for modelling -- this module does
not drop or filter any rows or columns beyond the fixed metadata columns
(date / city / lat / lon) and the target.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import Config


def haversine_matrix(city_meta: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Pairwise great-circle distance (km) between all cities (Eq. 1-2)."""
    lats = np.radians(city_meta[cfg.LAT_COL].values)
    lons = np.radians(city_meta[cfg.LON_COL].values)
    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def build_adjacency(city_meta: pd.DataFrame, cfg: Config, sigma_mult: float = 1.0):
    """Builds the symmetric, normalised adjacency matrix (Eq. 3-7).

    ``cfg.ADJ_MODE``:
      "gaussian" (default, adopted throughout the manuscript): Haversine
        distance -> Gaussian RBF kernel with bandwidth = median pairwise
        distance -> unit self-weight -> symmetric degree normalisation.
      "knn": the same kernel restricted to each node's ``cfg.KNN_K`` nearest
        neighbours, used only as an alternative construction in the graph
        comparison (see ``analysis/graph_comparison.py``).

    ``sigma_mult`` scales the Gaussian bandwidth (Eq. 5) by a fixed factor;
    used only by the bandwidth sensitivity sweep in
    ``analysis/hyperparameter_sensitivity.py`` (manuscript Table S1a). The
    default of 1.0 reproduces the adopted bandwidth exactly.

    Stochastic edge dropout (Eq. 8-9) is applied inside the model at train
    time, not here -- this function always returns the full (non-dropped)
    normalised adjacency.
    """
    dist = haversine_matrix(city_meta, cfg)
    N = len(city_meta)

    sigma = np.median(dist[dist > 0]) * sigma_mult
    W = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(W, 1.0)

    if cfg.ADJ_MODE == "knn":
        mask = np.zeros((N, N), dtype=bool)
        np.fill_diagonal(mask, True)
        for i in range(N):
            knn = np.argsort(dist[i])[1: cfg.KNN_K + 1]
            mask[i, knn] = True
            mask[knn, i] = True
        W = W * mask
    elif cfg.ADJ_MODE != "gaussian":
        raise ValueError(f"Unknown ADJ_MODE: {cfg.ADJ_MODE}")

    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1).clip(min=1e-6)))
    A = D_inv_sqrt @ W @ D_inv_sqrt
    return A.astype(np.float32), dist


def load_and_preprocess(cfg: Config):
    """Loads the raw CSV, interpolates short gaps, adds cyclic date
    features, splits by time, and fits/applies a StandardScaler on the
    training split only.

    Every city and every feature column present in the CSV (other than the
    fixed metadata columns) is used -- if you need to exclude specific
    cities or columns, do so in the source data itself, not here.

    Returns
    -------
    train, val, test : scaled arrays [T, N, F]
    scaler : fitted StandardScaler
    feature_cols : list[str]
    target_idx : int (index of the target column within feature_cols)
    city_meta : DataFrame with one row per city (name, lat, lon)
    cities : list[str]
    N : int
    dates : list of sorted unique dates
    """
    df = pd.read_csv(cfg.DATA_PATH, parse_dates=[cfg.DATE_COL])
    df = df.sort_values([cfg.DATE_COL, cfg.CITY_COL]).reset_index(drop=True)

    city_meta = (df.groupby(cfg.CITY_COL)[[cfg.LAT_COL, cfg.LON_COL]]
                   .mean().reset_index())
    cities = city_meta[cfg.CITY_COL].tolist()
    N = len(cities)
    city2idx = {c: i for i, c in enumerate(cities)}
    print(f"Cities ({N}): {cities}")

    exclude = [cfg.DATE_COL, cfg.CITY_COL, cfg.LAT_COL, cfg.LON_COL]
    feature_cols = [c for c in df.columns if c not in exclude]
    assert cfg.TARGET_COL in feature_cols, f"{cfg.TARGET_COL} not found in features"
    if feature_cols[0] != cfg.TARGET_COL:
        feature_cols.insert(0, feature_cols.pop(feature_cols.index(cfg.TARGET_COL)))
    target_idx = 0
    F_dim = len(feature_cols)
    print(f"Features ({F_dim}): {feature_cols}")

    dates = sorted(df[cfg.DATE_COL].unique())
    T = len(dates)
    date2t = {d: i for i, d in enumerate(dates)}

    data_arr = np.full((T, N, F_dim), np.nan)
    for _, row in df.iterrows():
        t = date2t[row[cfg.DATE_COL]]
        n = city2idx[row[cfg.CITY_COL]]
        data_arr[t, n, :] = row[feature_cols].values.astype(float)

    # Bidirectional linear interpolation of short data gaps.
    for n in range(N):
        for f in range(F_dim):
            s = pd.Series(data_arr[:, n, f])
            data_arr[:, n, f] = s.interpolate("linear", limit_direction="both").values

    if cfg.USE_DATE_FEATURES:
        dates_dt = pd.to_datetime(dates)
        doy = dates_dt.day_of_year.values.astype(float)
        dow = dates_dt.day_of_week.values.astype(float)
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        sin_dow = np.sin(2 * np.pi * dow / 7.0)
        cos_dow = np.cos(2 * np.pi * dow / 7.0)
        date_feats = np.stack([sin_doy, cos_doy, sin_dow, cos_dow], axis=1)   # [T, 4]
        date_feats = np.tile(date_feats[:, None, :], (1, N, 1))              # [T, N, 4]
        data_arr = np.concatenate([data_arr, date_feats], axis=-1)
        feature_cols = feature_cols + ["sin_doy", "cos_doy", "sin_dow", "cos_dow"]
        F_dim = data_arr.shape[-1]
        print(f"Date features added -> total features: {F_dim}")

    n_train = int(T * cfg.TRAIN_RATIO)
    n_val = int(T * cfg.VAL_RATIO)
    train_d = data_arr[:n_train]
    val_d = data_arr[n_train: n_train + n_val]
    test_d = data_arr[n_train + n_val:]
    print(f"Timesteps -- Train: {len(train_d)}, Val: {len(val_d)}, Test: {len(test_d)}")

    scaler = StandardScaler()
    scaler.fit(train_d.reshape(-1, F_dim))

    def sc(a):
        sh = a.shape
        return scaler.transform(a.reshape(-1, F_dim)).reshape(sh).astype(np.float32)

    return (sc(train_d), sc(val_d), sc(test_d), scaler, feature_cols, target_idx,
            city_meta, cities, N, dates)


def make_windows(data: np.ndarray, lookback: int, horizon: int):
    """Slides a [lookback]-day input / [horizon]-day target window over a
    scaled [T, N, F] array, returning X [S, lookback, N, F] and
    y [S, horizon, N] (target feature only)."""
    X, y = [], []
    for t in range(lookback, len(data) - horizon + 1):
        X.append(data[t - lookback: t])
        y.append(data[t: t + horizon, :, 0])
    return np.array(X, np.float32), np.array(y, np.float32)


def inverse_target(arr_sc: np.ndarray, scaler: StandardScaler, target_idx: int) -> np.ndarray:
    """Inverse-transforms a scaled [S, H, N] (or [S, N]) target-only array
    back to original units, using the scaler fit on all features."""
    orig_shape = arr_sc.shape
    n_feat = scaler.mean_.shape[0]
    dummy = np.zeros((arr_sc.size, n_feat))
    dummy[:, target_idx] = arr_sc.reshape(-1)
    return scaler.inverse_transform(dummy)[:, target_idx].reshape(orig_shape)


class STDataset(torch.utils.data.Dataset):
    """Thin Dataset wrapper around a pair of (X, y) numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def mixup_batch(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.3):
    """Standard mixup augmentation (Zhang et al., 2018) applied across the
    batch dimension."""
    if alpha <= 0:
        return X, y
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    return lam * X + (1 - lam) * X[idx], lam * y + (1 - lam) * y[idx]
