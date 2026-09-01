"""
Regression metrics and the statistical testing suite used for Table IV
(point estimates + bootstrap confidence intervals) and Table V (paired
significance tests between GD-GRU and every baseline).
"""

import numpy as np
from scipy import stats as sstats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from statsmodels.stats.multitest import multipletests
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# ------------------------------------------------------------------------
# Point-estimate metrics
# ------------------------------------------------------------------------

def mae_fn(t, p):
    return np.mean(np.abs(t - p))


def rmse_fn(t, p):
    return np.sqrt(np.mean((t - p) ** 2))


def r2_fn(t, p):
    return 1 - np.sum((t - p) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-12)


def mape_fn(t, p, floor=5.0):
    mask = t > floor
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100


def pcc_fn(t, p):
    return np.corrcoef(t.ravel(), p.ravel())[0, 1]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mape_floor: float = 5.0) -> dict:
    """Full metric set (MAE, RMSE, MAPE, R2, PCC) on the real (non-resampled)
    data. This is always the reported point estimate -- bootstrap resampling
    (below) is used only to derive a confidence interval around it, never to
    redefine the point itself."""
    yt = y_true.ravel()
    yp = y_pred.ravel()
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mask = yt > mape_floor
    mape = (np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100
            if mask.sum() > 0 else np.nan)
    r2 = r2_score(yt, yp)
    pcc, _ = sstats.pearsonr(yt, yp)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2, "PCC": pcc}


# ------------------------------------------------------------------------
# Block bootstrap (for temporally correlated test data)
# ------------------------------------------------------------------------

def _circular_block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator):
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, max(1, n), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n] % n


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, block_size=10, ci=95, seed=42):
    """Circular block-bootstrap CI for a metric. Returns (lo, hi) only -- the
    point estimate must be computed separately, directly on the full sample."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = yt.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circular_block_bootstrap_indices(n, block_size, rng)
        vals[b] = metric_fn(yt[idx], yp[idx])
    lo, hi = np.percentile(vals, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def bootstrap_diff_ci(err1, err2, n_boot=2000, block_size=10, ci=95, seed=42):
    """Block-bootstrap CI on the paired mean-absolute-error gap
    (MAE(model 1) - MAE(model 2))."""
    rng = np.random.default_rng(seed)
    e1 = np.abs(np.asarray(err1).ravel())
    e2 = np.abs(np.asarray(err2).ravel())
    n = len(e1)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circular_block_bootstrap_indices(n, block_size, rng)
        vals[b] = e1[idx].mean() - e2[idx].mean()
    lo, hi = np.percentile(vals, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(vals.mean()), float(lo), float(hi)


# ------------------------------------------------------------------------
# Paired significance tests
# ------------------------------------------------------------------------

def diebold_mariano_test(err1, err2, h: int = 1, loss: str = "squared"):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold (1997)
    small-sample correction (manuscript Sec. II-H). err1/err2 are paired
    forecast errors for the two models being compared; h is the forecast
    horizon (1 for the pooled 'ALL' comparison). A negative statistic means
    model 1 (err1) has the lower average loss."""
    err1 = np.asarray(err1).ravel()
    err2 = np.asarray(err2).ravel()
    assert len(err1) == len(err2), "paired series must be the same length"
    T = len(err1)

    if loss == "squared":
        d = err1 ** 2 - err2 ** 2
    elif loss == "absolute":
        d = np.abs(err1) - np.abs(err2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d_mean = d.mean()
    var_d = np.var(d, ddof=0)
    for k in range(1, h):
        if k < T:
            gamma_k = np.mean((d[:-k] - d_mean) * (d[k:] - d_mean))
            var_d += 2 * gamma_k
    var_d = max(var_d, 1e-12) / T

    dm_stat = d_mean / np.sqrt(var_d)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat_corr = dm_stat * hln
    p_value = 2 * (1 - sstats.t.cdf(np.abs(dm_stat_corr), df=T - 1))
    return dm_stat_corr, p_value, T


def wilcoxon_test(err1, err2, alternative: str = "less"):
    """One-sided Wilcoxon signed-rank test on |error|: H1 = model 1's
    absolute error is smaller than model 2's."""
    a1 = np.abs(np.asarray(err1).ravel())
    a2 = np.abs(np.asarray(err2).ravel())
    return sstats.wilcoxon(a1, a2, alternative=alternative, zero_method="wilcox")


def cohens_d_paired(err1, err2):
    """Paired Cohen's d on |error1| - |error2|. Negative d means model 1
    tends to have the smaller absolute error."""
    diff = np.abs(np.asarray(err1).ravel()) - np.abs(np.asarray(err2).ravel())
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def _bh_fdr_fallback(pvals):
    """Manual Benjamini-Hochberg FDR correction, used only if statsmodels
    is not installed."""
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(m)
    out[order] = q
    return out


def multiplicity_correction(p_raw, alpha: float = 0.05):
    """Bonferroni and Benjamini-Hochberg FDR correction across a family of
    p-values. Returns (p_bonferroni, sig_bonferroni, p_fdr_bh, sig_fdr_bh)."""
    p_raw = np.asarray(p_raw)
    if _HAS_STATSMODELS:
        rej_bonf, p_bonf, _, _ = multipletests(p_raw, method="bonferroni")
        rej_fdr, p_fdr, _, _ = multipletests(p_raw, method="fdr_bh")
    else:
        m = len(p_raw)
        p_bonf = np.clip(p_raw * m, 0, 1)
        p_fdr = _bh_fdr_fallback(p_raw)
        rej_bonf = p_bonf < alpha
        rej_fdr = p_fdr < alpha
    return p_bonf, rej_bonf, p_fdr, rej_fdr
