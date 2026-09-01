#!/usr/bin/env python
"""
Train the 5-seed DCRNN ensemble (manuscript baseline comparison).

DCRNN uses the same data split, scaler, and adjacency matrix as GD-GRU
so its predictions land on the same test windows and can be directly
paired in the Diebold-Mariano significance tests (significance.py).

Run AFTER train_gdgru.py (which produces the shared data split).

Usage
-----
    python scripts/train_dcrnn.py

Output
------
    results/predictions/dcrnn_final_predictions.npz
    results/checkpoints/dcrnn_seed*.pt

Reproduces
----------
    Table 4 (DCRNN row), Table 5 (GD-GRU vs DCRNN significance).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pm25gdgru.train_dcrnn import run

if __name__ == "__main__":
    run()
