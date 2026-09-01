"""
analysis — supplementary sensitivity sweeps and ablation studies.

Each module in this sub-package reproduces one supplementary table from the
manuscript:

  common.py             — shared helpers (resumable CSV append, key tracking)
  lookback_ablation.py  — Table S5: lookback-window ablation (7/10/14/21 days)
  graph_comparison.py   — Table S3: alternative graph-construction variants
                          (geographic, adaptive, wind-informed, hybrid)

All sweeps re-use ``pm25gdgru.engine.train_ensemble``, so every variant is
trained and evaluated exactly the same way as the main GD-GRU model.

Usage
-----
    python -m pm25gdgru.analysis.lookback_ablation
    python -m pm25gdgru.analysis.graph_comparison
"""
