#!/usr/bin/env python
"""
Run the full explainability suite on the trained GD-GRU ensemble.

Applies five attribution methods to the 5-seed ensemble loaded from
checkpoints produced by train_gdgru.py:
  1. Gradient × Input  (Fig. 8a, 10, 11)
  2. Temporal Sensitivity (Fig. 11)
  3. Permutation Feature Importance  (Fig. 8b)
  4. KernelSHAP  (Fig. 8c, 9)
  5. Occlusion-based Spatial Influence  (Fig. 12, graph topology validation)

Run AFTER train_gdgru.py.

Usage
-----
    python scripts/run_explainability.py

Output
------
    results/figures/  (Fig. 8-12)
    results/feature_importance_grad_input.csv
    results/feature_importance_pfi.csv
    results/feature_importance_shap.csv
    results/spatial_influence_matrix.csv
    results/temporal_sensitivity.csv

Reproduces
----------
    Table 7, Fig. 8, 9, 10, 11, 12 and the graph topology validation
    (r = 0.928, p < 0.001, manuscript Sec. "Spatial Influence and Graph
    Topology Validation").
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pm25gdgru.explainability import run

if __name__ == "__main__":
    run()
