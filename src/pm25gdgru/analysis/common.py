"""Small shared helper for the sweep scripts in this package.

Each sweep appends one row per configuration to a CSV under
``RESULTS_DIR``. If the CSV already has a row for a given ``key``, that
configuration is skipped -- combined with ``engine.train_ensemble``'s own
per-seed checkpoint caching, this makes every sweep safely resumable after
an interrupted run (e.g. a disconnected Colab session).
"""

import os

import pandas as pd


def load_existing(csv_path: str, key_col: str = "key") -> set:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(df[key_col].astype(str))
    except Exception:
        return set()


def append_row(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)


def flatten_metrics(overall: dict, prefix: str = "") -> dict:
    return {f"{prefix}{k.replace(' ', '_').replace('(%)', 'pct')}": v
            for k, v in overall.items()}
