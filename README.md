# Explainable Spatiotemporal GNNs for Multi-City PM₂.₅ Forecasting in Bangladesh

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official code repository for the manuscript:

> **Explainable Spatiotemporal Graph Neural Networks: Multi-City PM₂.₅ Forecasting in Bangladesh with Geographic Graph Topology Validation**
> Pritom Ray Nobin, Department of Urban and Regional Planning, KUET, Bangladesh.

---

## Overview

This repository provides a complete, reproducible implementation of the **Graph Diffusion GRU (GD-GRU)** model for 3-day-ahead multi-city PM₂.₅ forecasting across Bangladesh's 12-city monitoring network (2022–2025).


---

## Repository Structure

```
pm25-bangladesh-gdgru/
├── src/
│   └── pm25gdgru/                   # Main Python package
│       ├── __init__.py
│       ├── config.py                # All hyperparameters (Table 2)
│       ├── data.py                  # Data loading, preprocessing, window building
│       ├── engine.py                # Unified training loop (all models share this)
│       ├── losses.py                # Scaled Huber loss + horizon weighting
│       ├── metrics.py               # MAE, RMSE, MAPE, R², PCC, DM test, bootstrap CI
│       ├── evaluate.py              # Table IV builder (bootstrap CIs)
│       ├── significance.py          # Table V builder (DM + Wilcoxon + FDR correction)
│       ├── baselines.py             # Persistence, Climatology, RF, XGBoost, GRU-no-graph
│       ├── gru_baseline.py          # Standalone GRU (no graph) model definition
│       ├── plotting.py              # All manuscript figures (Fig. 4–7)
│       ├── explainability.py        # Gradient×Input, PFI, KernelSHAP, occlusion (Fig. 8–12, Table 7)
│       ├── train_gdgru.py           # GD-GRU ensemble training (main model)
│       ├── train_dcrnn.py           # DCRNN ensemble training (baseline)
│       ├── models/
│       │   ├── __init__.py          # Exports GDGRUNet, DCRNNNet
│       │   ├── gdgru.py             # GD-GRU: GDConv, GDGRUCell, GDGRUNet (Eq. 10–24)
│       │   └── dcrnn.py             # DCRNN: DiffusionConv, DCGRUCell, DCRNNNet
│       └── analysis/
│           ├── __init__.py
│           ├── common.py            # Resumable CSV helpers shared by sweep scripts
│           ├── lookback_ablation.py # Table S5: lookback window = 7/10/14/21 days
│           └── graph_comparison.py  # Table S3: geographic / adaptive / wind / hybrid graphs
├── scripts/
│   ├── train_gdgru.py               # ① Train GD-GRU ensemble
│   ├── train_dcrnn.py               # ② Train DCRNN ensemble
│   ├── run_baselines.py             # ③ Run all classical baselines
│   ├── run_significance.py          # ④ Build Table IV + V (significance)
│   ├── run_explainability.py        # ⑤ Generate explainability figures (Fig. 8–12)
│   └── run_analysis.py              # ⑥ Run supplementary ablations (Table S3, S5)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Manuscript–Code Correspondence

| Manuscript element | Script / module |
|---|---|
| **GD-GRU architecture** (Eq. 10–24, Fig. 2) | `src/pm25gdgru/models/gdgru.py` |
| **DCRNN architecture** (baseline) | `src/pm25gdgru/models/dcrnn.py` |
| **Graph construction** (Eq. 1–9, Fig. 3) | `data.py → build_adjacency()` |
| **Training loop** (Eq. 18–24) | `engine.py → train_ensemble()` |
| **Loss function** (Eq. 25) | `losses.py` |
| **Table 2** (hyperparameters) | `config.py` |
| **Table 3** (baseline hyperparameters) | `baselines.py` |
| **Table 4** (overall performance) | `evaluate.py` + `scripts/run_significance.py` |
| **Table 5** (statistical significance, DM+Wilcoxon+FDR) | `significance.py` + `metrics.py` |
| **Table 6** (per-horizon performance) | `train_gdgru.py → run()` |
| **Table 7** (feature importance convergence) | `explainability.py` |
| **Fig. 4** (scatter: observed vs predicted) | `plotting.py` |
| **Fig. 5** (R² heatmap: city × horizon) | `plotting.py` |
| **Fig. 6** (time-series: 4 representative cities) | `plotting.py` |
| **Fig. 7** (training/validation loss curves) | `plotting.py` |
| **Fig. 8** (feature importance: Grad×In, PFI, SHAP) | `explainability.py` |
| **Fig. 9** (KernelSHAP beeswarm) | `explainability.py` |
| **Fig. 10** (city-level Gradient×Input heatmap) | `explainability.py` |
| **Fig. 11** (temporal sensitivity / lookback) | `explainability.py` |
| **Fig. 12** (spatial influence occlusion matrix, r = 0.928) | `explainability.py` |
| **Table S3** (graph construction ablation) | `analysis/graph_comparison.py` |
| **Table S5** (lookback window ablation) | `analysis/lookback_ablation.py` |

---

## Data

The model uses daily observations from **seven sources** for **12 cities** (2022–2025). See Table 1 of the manuscript for the complete feature list and sources.

**Data Availability Statement (from manuscript):**

> All meteorological reanalysis (ERA5), satellite-derived (MODIS, Sentinel-5P, VIIRS), and static variables used in this study, are fully and freely available without restriction on Figshare at DOI: 10.6084/m9.figshare.33405397. Ground-level PM2.5 data were obtained from the Continuous Air Monitoring Station (CAMS) network operated by the Department of Environment (DoE), Bangladesh (https://doe.gov.bd). Because these data are owned by the DoE, they are subject to third-party restrictions, and the author do not hold redistribution rights to these station-level records. Interested researchers may request access to the exact same PM2.5 dataset used in this study directly from the DoE, either under Bangladesh's Right to Information Act, 2009, by submitting a written application to the DoE's Designated Officer, or by direct written request to the DoE's Air Quality Management Wing, Department of Environment, E-16, Agargaon, Sher-e-Bangla Nagar, Dhaka-1207, Bangladesh. The author did not have any special access privileges to these data that others would not have.

**Expected input format**: A single CSV file with one row per (city, date) pair. The data loader (`data.py`) automatically infers all non-metadata columns as features. If you wish to exclude specific columns or cities, do so in the CSV itself prior to loading.

Set the path in `config.py`:
```python
cfg.DATA_PATH = "path/to/your/pm25_data.csv"
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/pts5625/pm25-bangladesh-gdgru.git
cd pm25-bangladesh-gdgru

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.0 with CUDA 11.8 (GPU strongly recommended).

---

## Reproducing the Results

Run the scripts in order. Each script is safe to re-run; it skips already-completed steps.

```bash
# ① Train the main GD-GRU ensemble (≈45 min on a single GPU)
python scripts/train_gdgru.py

# ② Train the DCRNN ensemble for paired comparison (≈75 min)
python scripts/train_dcrnn.py

# ③ Run all classical baselines (Persistence, Climatology, RF, XGBoost, GRU no-graph)
python scripts/run_baselines.py

# ④ Build Table IV (bootstrap CIs) and Table V (DM + Wilcoxon + FDR significance)
python scripts/run_significance.py

# ⑤ Run full explainability suite → Fig. 8–12, Table 7
python scripts/run_explainability.py

# ⑥ Supplementary ablations: lookback window (Table S5) and graph variants (Table S3)
python scripts/run_analysis.py --lookback    # Table S5 only
python scripts/run_analysis.py --graph       # Table S3 only
python scripts/run_analysis.py               # both
```

All outputs are written to the `results/` directory (created automatically):

```
results/
├── predictions/
│   ├── gdgru_final_predictions.npz
│   ├── dcrnn_final_predictions.npz
│   └── baselines/                         (one .npz per baseline)
├── checkpoints/                            (model weights, per seed)
├── figures/                                (Fig. 4–12 as .png / .pdf)
├── table_iv_with_ci.csv                    (Table 4)
├── table_v_significance.csv                (Table 5)
├── table_s3_graph_comparison.csv           (Table S3)
├── table_s5_lookback_ablation.csv          (Table S5)
├── feature_importance_grad_input.csv
├── feature_importance_pfi.csv
├── feature_importance_shap.csv
├── spatial_influence_matrix.csv
├── temporal_sensitivity.csv
└── test_predictions.csv                    (long-format predictions)
```

Set a custom output directory:
```bash
PM25_RESULTS_DIR=/scratch/my_results python scripts/train_gdgru.py
```

---

## Model Architecture

The **GD-GRU** (Graph Diffusion GRU) is an encoder-decoder spatiotemporal GNN:

1. **Input projection**: Linear → LayerNorm → GELU → Dropout (per time step, per city)
2. **GD-GRU Encoder**: `L = 21` lookback steps through a `GDGRUCell` that replaces all three GRU linear gates with Chebyshev graph diffusion convolution (order *K* = 2)
3. **Autoregressive Decoder**: 3-step ahead prediction with optional teacher forcing (schedule described in Eq. 19–20)
4. **Per-city output heads**: Diagonal extraction from a shared linear projection
5. **Ensemble**: 5 independently seeded models; predictions are averaged (Eq. 24)

Key hyperparameters (see `config.py`):

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Contact

**Pritom Ray Nobin** · nobin5625@gmail.com · ORCID: [0009-0000-3981-2127](https://orcid.org/0009-0000-3981-2127)

Department of Urban and Regional Planning, Khulna University of Engineering & Technology (KUET), Bangladesh.
