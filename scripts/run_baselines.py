#!/usr/bin/env python
"""
Run all classical and statistical baselines (Persistence, Climatology,
Random Forest, XGBoost, GRU no-graph).

Run AFTER train_gdgru.py so the same data split and scaler are used.

Usage
-----
    python scripts/run_baselines.py

Output
------
    results/predictions/baselines/  (one .npz per model)

Reproduces
----------
    Table 4 (Persistence, Climatology, RF, XGBoost, GRU no-graph rows).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pm25gdgru.baselines import run

if __name__ == "__main__":
    run()
