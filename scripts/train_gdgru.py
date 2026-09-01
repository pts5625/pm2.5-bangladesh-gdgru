#!/usr/bin/env python
"""
Train the 5-seed GD-GRU ensemble.

This script trains the main Graph Diffusion GRU model described in the
manuscript, saves per-seed checkpoints, produces all publication figures
(Fig. 4-7), and writes the cached prediction arrays used by every
downstream analysis.

Usage
-----
    # From the repository root:
    python scripts/train_gdgru.py

    # With a custom results directory:
    PM25_RESULTS_DIR=/path/to/results python scripts/train_gdgru.py

Output
------
    results/
    ├── predictions/gdgru_final_predictions.npz  (used by evaluate / significance / explainability)
    ├── checkpoints/gdgru_seed*.pt
    ├── figures/                                  (Fig. 4-7 from the manuscript)
    └── test_predictions.csv                      (long-format sample × horizon × city)

Reproduces
----------
    Table 4 (overall performance), Table 6 (per-horizon metrics),
    Table S1 city-level R² heatmap (Fig. 5), Fig. 4, 6, 7.
"""

import sys
import os

# Ensure the package is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pm25gdgru.train_gdgru import run

if __name__ == "__main__":
    run()
