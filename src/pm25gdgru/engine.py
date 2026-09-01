"""
Shared training engine.

GDGRUNet, DCRNNNet, and PlainGRUNet all expose the same
``forward(x, graph, target, teacher_forcing_ratio)`` interface, so a single
training loop and ensemble runner can drive all three (and any future
model that follows the same interface). This replaces what used to be
several near-identical copies of the training loop, one per model /
per analysis script.

``graph`` is passed through unchanged to the model, so it may be a single
adjacency tensor (GD-GRU) or a ``(T_fwd, T_bwd)`` tuple (DCRNN), etc.
"""

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import Config, set_seed
from .data import STDataset, inverse_target
from .losses import build_horizon_weights, weighted_huber_loss
from .metrics import compute_metrics


class EarlyStopping:
    """Tracks an exponentially-smoothed validation loss (Eq. 27) and saves
    the best checkpoint seen so far."""

    def __init__(self, patience: int, min_delta: float, min_epochs: int,
                 smooth: float, path: str):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.smooth = smooth
        self.path = path
        self.counter = 0
        self.best = np.inf
        self.smoothed = None
        self.stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int):
        if self.smoothed is None:
            self.smoothed = val_loss
        else:
            self.smoothed = self.smooth * self.smoothed + (1 - self.smooth) * val_loss

        if epoch < self.min_epochs:
            if self.smoothed < self.best - self.min_delta:
                self.best = self.smoothed
                torch.save(model.state_dict(), self.path)
            return

        if self.smoothed < self.best - self.min_delta:
            self.best = self.smoothed
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_one_seed(model_ctor, model_kwargs: dict, X_tr, y_tr, X_va, y_va,
                    graph, cfg: Config, seed: int, horiz_w: torch.Tensor,
                    model_path: str, tf_initial: float = None,
                    tf_decay_epochs: int = None, verbose: bool = True):
    """Trains one ensemble member from scratch (or returns the cached
    checkpoint if ``model_path`` already exists) and returns
    ``(model, history)``.

    tf_initial / tf_decay_epochs override cfg.TF_INITIAL / cfg.TF_DECAY_EPOCHS
    for hyperparameter sensitivity sweeps; leave as None to use the values
    already set on cfg.
    """
    tf_initial = cfg.TF_INITIAL if tf_initial is None else tf_initial
    decay_epochs = cfg.TF_DECAY_EPOCHS if tf_decay_epochs is None else tf_decay_epochs
    decay_epochs = decay_epochs if decay_epochs is not None else cfg.EPOCHS

    set_seed(seed)

    if os.path.exists(model_path):
        model = model_ctor(**model_kwargs).to(cfg.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        model.eval()
        if verbose:
            print(f"  [cache hit] loaded existing checkpoint: {model_path}")
        return model, {"train": [], "val": []}

    tr_ld = torch.utils.data.DataLoader(
        STDataset(X_tr, y_tr), cfg.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(cfg.DEVICE == "cuda"))
    va_ld = torch.utils.data.DataLoader(
        STDataset(X_va, y_va), cfg.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(cfg.DEVICE == "cuda"))

    model = model_ctor(**model_kwargs).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=cfg.LR_FACTOR,
                                   patience=cfg.LR_PATIENCE, min_lr=cfg.MIN_LR)
    stopper = EarlyStopping(cfg.PATIENCE, cfg.MIN_DELTA, cfg.MIN_EPOCHS,
                             cfg.VAL_SMOOTH, path=model_path)

    history = {"train": [], "val": []}

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        tr_total = 0
        tf_ratio = max(0.0, tf_initial * (1.0 - epoch / decay_epochs))
        for X_b, y_b in tr_ld:
            X_b = X_b.to(cfg.DEVICE)
            y_b = y_b.to(cfg.DEVICE)
            if cfg.NOISE_STD > 0:
                X_b = X_b + torch.randn_like(X_b) * cfg.NOISE_STD
            optimizer.zero_grad()
            pred = model(X_b, graph, target=y_b, teacher_forcing_ratio=tf_ratio)
            loss = weighted_huber_loss(pred, y_b, horiz_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            tr_total += loss.item() * len(X_b)
        tr_loss = tr_total / len(tr_ld.dataset)

        model.eval()
        va_total = 0
        with torch.no_grad():
            for X_b, y_b in va_ld:
                X_b = X_b.to(cfg.DEVICE)
                y_b = y_b.to(cfg.DEVICE)
                pred = model(X_b, graph, target=None, teacher_forcing_ratio=0.0)
                va_total += F.huber_loss(pred, y_b, delta=1.0).item() * len(X_b)
        va_loss = va_total / len(va_ld.dataset)

        scheduler.step(va_loss)
        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        stopper(va_loss, model, epoch)

        if verbose and (epoch % 30 == 0 or stopper.stop):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    ep {epoch:4d}  train={tr_loss:.4f}  val={va_loss:.4f}  "
                  f"smooth={stopper.smoothed:.4f}  lr={lr_now:.1e}")
        if stopper.stop:
            if verbose:
                print(f"    Early stop at epoch {epoch}.")
            break

    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    history["stopped_epoch"] = epoch
    return model, history


def predict(model: torch.nn.Module, loader, graph, device: str):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            p = model(X_b.to(device), graph, target=None, teacher_forcing_ratio=0.0)
            preds.append(p.cpu().numpy())
            trues.append(y_b.numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0)


@dataclass
class EnsembleResult:
    tag: str
    pred_orig: np.ndarray
    true_orig: np.ndarray
    preds_orig_per_seed: list
    overall: dict
    per_horizon: list
    histories: list = field(default_factory=list)


def train_ensemble(model_ctor, model_kwargs: dict, X_tr, y_tr, X_va, y_va, X_te, y_te,
                    graph, cfg: Config, seeds, scaler, target_idx: int,
                    ckpt_dir: str, tag: str, tf_initial: float = None,
                    tf_decay_epochs: int = None, verbose: bool = False) -> EnsembleResult:
    """Trains a multi-seed ensemble, then averages each seed's *scaled*
    test predictions before inverse-transforming and scoring once
    (manuscript Eq. 24) -- never averages per-seed metrics.

    Each seed's checkpoint is cached to ``{ckpt_dir}/{tag}_seed{seed}.pt``,
    so re-running with the same tag skips already-trained seeds.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    horiz_w = build_horizon_weights(cfg)

    te_ld = torch.utils.data.DataLoader(
        STDataset(X_te, y_te), cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    preds_sc_per_seed, histories = [], []
    true_sc = None
    for seed in seeds:
        model_path = os.path.join(ckpt_dir, f"{tag}_seed{seed}.pt")
        model, history = train_one_seed(
            model_ctor, model_kwargs, X_tr, y_tr, X_va, y_va, graph, cfg, seed,
            horiz_w, model_path, tf_initial=tf_initial,
            tf_decay_epochs=tf_decay_epochs, verbose=verbose)
        pred_sc, true_sc = predict(model, te_ld, graph, cfg.DEVICE)
        preds_sc_per_seed.append(pred_sc)
        histories.append(history)

    pred_sc_ens = np.mean(preds_sc_per_seed, axis=0)
    pred_orig = np.clip(inverse_target(pred_sc_ens, scaler, target_idx), 0, None)
    true_orig = inverse_target(true_sc, scaler, target_idx)
    preds_orig_per_seed = [np.clip(inverse_target(p, scaler, target_idx), 0, None)
                            for p in preds_sc_per_seed]

    overall = compute_metrics(true_orig, pred_orig, cfg.MAPE_FLOOR)
    per_horizon = [compute_metrics(true_orig[:, h, :], pred_orig[:, h, :], cfg.MAPE_FLOOR)
                   for h in range(cfg.HORIZON)]

    return EnsembleResult(tag=tag, pred_orig=pred_orig, true_orig=true_orig,
                           preds_orig_per_seed=preds_orig_per_seed,
                           overall=overall, per_horizon=per_horizon, histories=histories)
