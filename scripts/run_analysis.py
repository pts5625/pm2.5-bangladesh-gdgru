#!/usr/bin/env python
"""
Run supplementary sensitivity analyses: lookback-window ablation and
alternative graph-construction comparison.

Run AFTER train_gdgru.py (shared data pipeline) and train_dcrnn.py.

Usage
-----
    # Lookback-window ablation only (Table S5):
    python scripts/run_analysis.py --lookback

    # Graph construction comparison only (Table S3):
    python scripts/run_analysis.py --graph

    # Both (default):
    python scripts/run_analysis.py

Output
------
    results/table_s3_graph_comparison.csv   (Table S3)
    results/table_s5_lookback_ablation.csv  (Table S5)

Reproduces
----------
    Supplementary Table S3 (graph construction variants):
        geographic, adaptive, wind-informed, hybrid adjacency matrices.
    Supplementary Table S5 (lookback ablation):
        GD-GRU retrained with 7 / 10 / 14 / 21-day lookback windows.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="Run supplementary analyses.")
    parser.add_argument("--lookback", action="store_true",
                        help="Run lookback-window ablation (Table S5).")
    parser.add_argument("--graph", action="store_true",
                        help="Run graph-construction comparison (Table S3).")
    args = parser.parse_args()

    # Default: run both
    run_all = not args.lookback and not args.graph

    if args.lookback or run_all:
        print("=" * 70)
        print("Running lookback-window ablation (Table S5)...")
        print("=" * 70)
        from pm25gdgru.analysis.lookback_ablation import run as run_lookback
        run_lookback()

    if args.graph or run_all:
        print("=" * 70)
        print("Running graph construction comparison (Table S3)...")
        print("=" * 70)
        from pm25gdgru.analysis.graph_comparison import run as run_graph
        run_graph()


if __name__ == "__main__":
    main()
