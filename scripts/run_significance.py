#!/usr/bin/env python
"""
Build Table IV (with 95% bootstrap CIs) and Table V (statistical
significance: GD-GRU vs every baseline).

Run AFTER all training scripts have completed and their .npz prediction
files exist under results/predictions/.

Usage
-----
    python scripts/run_significance.py

Output
------
    results/table_iv_with_ci.csv
    results/table_v_significance.csv

Reproduces
----------
    Table 4 (with CIs) and Table 5 of the manuscript.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pm25gdgru.significance import run

if __name__ == "__main__":
    run()
