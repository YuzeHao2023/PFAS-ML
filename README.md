# PFAS-ML

## Overview ✅

PFAS-ML is a small reproducible demonstration project for training an XGBoost classifier on the PFAS dataset provided in `data/science.ado6638_tables_s2_to_s5_and s9_to_s13.xlsx`. The repository contains a training and evaluation pipeline that:

- Loads Table S9 as observations (observed PFAS detection) and selects a small set of features (modeled thresholds, zip, state) as a simple baseline feature set
- Trains an XGBoost classifier
- Computes precision / recall / F1 across a uniform grid of probability thresholds
- Saves plots and a CSV summarizing per-threshold metrics and the trained model

This README documents how to install dependencies and run training and evaluation end-to-end.

---

## Requirements 🔧

- Python 3.10+ (the dev container in this workspace uses Ubuntu 24.04)
- The dependencies are listed in `requirements.txt` and can be installed with pip

Install with:

```bash
python3 -m pip install -r requirements.txt
```

(If you run into Excel read errors, ensure `openpyxl` is installed — it is already included in the `requirements.txt`.)

---

## Files & structure 📁

- `src/main2.py` — Main pipeline: loads data, builds features, trains XGBoost, computes threshold metrics, saves `outputs/xgb_model.json`, `outputs/threshold_results.csv` and `outputs/threshold_analysis.png`.
- `src/main.py` — Lightweight entrypoint that calls the pipeline (convenience wrapper).
- `data/` — Input data (Excel workbook with tables S2–S13). The pipeline uses **Table S9** by default.
- `outputs/` — Directory where results are saved after running the pipeline.

---

## Quick start — train and evaluate (recommended) ▶️

Train the model and run threshold analysis with the defaults:

```bash
python3 src/main2.py
```

This will:
- Read `Table S9` from `data/science.ado6638_tables_s2_to_s5_and s9_to_s13.xlsx` (skipping header rows)
- Construct features and labels
- Train an XGBoost model (saved to `outputs/xgb_model.json`)
- Evaluate over 101 thresholds in [0, 1]
- Save `outputs/threshold_results.csv` and `outputs/threshold_analysis.png`

You can also run the convenience wrapper (does the same by default):

```bash
python3 src/main.py
```

---

## CLI options (for reproducible runs) 🛠️

`src/main2.py` accepts optional command-line arguments:

- `--data-path` : Path to the Excel file (default: `data/science.ado6638_tables_s2_to_s5_and s9_to_s13.xlsx`)
- `--sheet-name` : Excel sheet name to use (default: `Table S9`)
- `--output-dir` : Directory to save outputs (default: `outputs`)
- `--test-size` : Test split proportion (default: `0.2`)
- `--seed` : Random seed for reproducibility (default: `42`)
- `--thresholds` : Number of thresholds to sample between 0 and 1 (default: `101`)

Example (custom run):

```bash
python3 src/main2.py --data-path "data/science.ado6638_tables_s2_to_s5_and s9_to_s13.xlsx" --sheet-name Table\ S9 --output-dir outputs/custom --thresholds 201
```

---

## Outputs ✨

- `outputs/xgb_model.json` — saved XGBoost model
- `outputs/threshold_results.csv` — per-threshold precision/recall/f1 table
- `outputs/threshold_analysis.png` — plot of Precision / Recall / F1 vs threshold

---

## Notes & next steps 💡

- The current feature set is intentionally small (modeled thresholds, zip code, and one-hot encoded states) to keep the example simple. For production or research purposes, augment features from `Table S5` or external rasters as described in the paper.
- For a more robust experiment, add cross-validation and hyperparameter search (GridSearchCV or randomized search) and a better feature engineering pipeline.

---

If you want, I can:
1) annotate and improve plots (mark best F1, 0.75 threshold, ROC AUC),
2) add cross-validation and hyperparameter tuning, or
3) expand feature extraction to include fields from `Table S11` and other sheets.

Reply with the number of the task you want next. ✅