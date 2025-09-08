# Environment‑Specific Thermal Sensation Prediction

Predicts **thermal sensation** across multiple **environments** (e.g., *Classroom*, *Hostel*, *Workshop or laboratory*).  
The system trains per‑environment models and blends them with a **stacked ensemble (main model)**. It also benchmarks a suite of **baseline models** and exports a **ranked comparison**.

---

## 1) Architecture

```
Raw data ─┐
          ├─▶ Feature Engineering & Cleaning
          │      • numeric & categorical handling
          │      • per‑environment splits
          │
          ├─▶ Feature Selection (per base model or global)
          │      • univariate scores (R² / η²)
          │      • permutation importance (ΔR²)
          │      • hybrid rank = 0.4*univariate + 0.6*permutation (configurable)
          │      • stability filters (TOP_M_PER_FOLD, STABILITY_TAU, K ranges)
          │
          ├─▶ Base Learners (per environment)
          │      • CatBoost, XGBoost, LightGBM, ElasticNet, SVR (ensemble bases)
          │      • tuned via environment presets in `utils/config.py`
          │      • K‑fold OOF predictions
          │
          ├─▶ Stacked Ensemble (MAIN MODEL)
          │      • Meta‑learner options: Ridge | NNLS | Averaging | LightGBM
          │      • Repeated CV with optional stratification by target bins
          │      • Auto‑select best meta by CV R² (optional)
          │      • Optional calibration
          │
          └─▶ Evaluation & Reporting
                 • per‑environment metrics (R², RMSE, MAE, Accuracy, MBE, Residual STD)
                 • predictions CSV per environment
                 • unified comparison across models
                 • ranked leaderboard per environment
```

### Key design choices
- **Environment‑specific modeling**: each environment gets its own selected features and tuned learners.
- **Diverse bases, simple meta**: blends strong tree‑based learners (CatBoost/XGB/LGBM) with a linear/NNLS meta to avoid overfitting when meta features are few.
- **Stable CV**: repeated K‑fold and optional **stratification** (by quantized target) stabilize the meta‑learner.
- **Config‑first**: all behavior (selection mode/scope, CV, bases, meta, baselines, per‑environment params) is controlled in `utils/config.py`.

---

## 2) Main Model: Stacked Ensemble

**Base learners** (default):
- `catboost`, `xgboost`, `lightgbm`, `elasticnet`, `svr_rbf`

**Meta‑learners** (choose one or auto‑select):
- **Ridge** regression with α search
- **NNLS** (non‑negative linear blend); cleanly upweights dominant bases
- **Avg** (uniform averaging); strong when bases are similarly good
- **LightGBM** (tree meta); use when many meta‑features or nonlinearity helps

**Cross‑validation for meta**:
- `FOLDS`: K folds (default 5)
- `CV_REPEATS`: repeats of CV (default 3)
- `STRATIFY_BINS`: quantile bins for stratified CV (default 7; set 0 to disable)

**Feature selection** (per base or global):
- `SELECTION_MODE ∈ {none, univariate_only, permutation_only, hybrid}`
- Hybrid rank uses `UNIV_WEIGHT` and `PERM_WEIGHT` (defaults 0.4/0.6)
- Stability via `TOP_M_PER_FOLD`, `STABILITY_TAU`, `K_VALUES`/`KMIN`/`KMAX`

> Optional production policy: if a single base outperforms the stack by > ε (e.g., 0.01 R²), deploy the single model for that environment; else deploy the stack.

---

## 3) Baseline Models (for comparison)

Configurable in `utils/config.py` (`BASELINE_MODELS`). Defaults include:
- **SVR (RBF)** — `svr_rbf`
- **Random Forest**
- **Gradient Boosting (sklearn)**
- **Decision Tree**
- **Linear Regression**
- **Ridge**
- **Lasso**
- **KNN (distance)**
- **MLP (Neural Network)** with early stopping
- **XGBoost (single)**
- **LightGBM (single)**
- **AdaBoost**

> You can remove any baseline (e.g., `catboost_single`) by deleting it from `BASELINE_MODELS`.  
> Ensemble bases are controlled separately by `ENSEMBLE_BASE_MODELS`.

---

## 4) Installation

```bash
# Create & activate virtual environment (Windows PowerShell)
python -m venv venv
.env\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
---

## 5) Execution

**A) Main ensemble only**
```bash
python run_main_pipeline.py
```

**Force a specific stacker (overrides config just for this run):**
```bash
# ridge | nnls | avg | lgbm
# Windows PowerShell:
$env:FORCE_STACKER_METHOD="ridge"; python run_main_pipeline.py; Remove-Item Env:FORCE_STACKER_METHOD

# macOS/Linux:
export FORCE_STACKER_METHOD=nnls; python run_main_pipeline.py; unset FORCE_STACKER_METHOD
```

**B) Full comparison (ensemble + baselines + ranked leaderboard)**
```bash
python scripts/run_full_comparison.py
```
This will:
1) run the ensemble,
2) run all baselines,
3) save `output/comparison_all_models.csv` (all metrics),
4) save `output/comparison_all_models_ranked.csv` with a **Rank** column  
   (ranking: higher **R²**, then lower **RMSE**, then lower **MAE**).

**C) Baselines only**
```bash
python scripts/run_baselines.py
```

**D) Meta‑learner sweep (optional)**
```bash
python scripts/run_stacker_sweep.py
```

**E) Additional Analysis Scripts**
```bash
# Generate SCIE figures
python scripts/make_scie_figures.py --env all

# Run statistical tests and generate plots
python scripts/run_stats_and_plots.py

# Generate predicted vs observed plots
python scripts/plot_pred_vs_obs.py

# Run ablation studies
python scripts/run_ablations.py

# Run leave-one-environment-out generalizability tests
python scripts/run_loeo_generalizability.py
```

> **Run from the project root** (the folder that contains `run_main_pipeline.py`).

---

## 6) Configuration Essentials

Edit **`utils/config.py`**:
- `SEED`, `FOLDS`, selection knobs, calibration: `CALIBRATION_ON`
- **Ensemble bases**: `ENSEMBLE_BASE_MODELS = ["catboost","xgboost","lightgbm","elasticnet","svr_rbf"]`
- **Meta**: `STACKING_METHOD`, `AUTO_SELECT_STACKER`, `STACKER_CANDIDATES`, `META_RIDGE_ALPHAS`, `CV_REPEATS`, `STRATIFY_BINS`
- **Baselines**: `BASELINE_MODELS` (add/remove to control comparison set)
- **Environment presets**: `get_environment_params(env_name)` adjusts per‑environment hyper‑params for CatBoost/LGBM/etc.

---

## 7) Inputs & Outputs

**Inputs**
- The loader in your repository reads per‑environment sheets / files (no changes required if your previous runs worked). If you moved data, update your loader paths accordingly.

**Per‑environment outputs**
- `output/<ENV>_selected_features.csv` (selected features per base)
- `output/<ENV>_predictions.csv` (row‑wise predictions & targets)
- Console + `output/predicted_tsv_results.csv` (summary metrics)

**Cross‑model comparison**
- `output/comparison_all_models.csv`
- `output/comparison_all_models_ranked.csv` (adds `Rank` per environment)
