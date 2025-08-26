# Environment Specific Thermal Sensation Prediction

One fixed ensemble (CatBoost, ExtraTrees, ElasticNet, XGBoost → LightGBM meta), with per-environment, per-model
feature selection via **Hybrid Stability-Selected Permutation (HSSP)**. Outputs feature rankings, predictions,
intervals, and metrics. Includes an **ablation runner**.

## Quick start
1. `pip install -r requirements.txt`
2. Put your Excel at `dataset/input_dataset.xlsx`
3. Run the main pipeline: `python run_main_pipeline.py`
4. Run ablations: `python scripts/run_ablations.py`
5. Run baselines: `python scripts/run_baselines.py`

## Switches (utils/config.py)
- `SELECTION_MODE`: 'none' | 'univariate_only' | 'permutation_only' | 'hybrid'
- `SELECTION_SCOPE`: 'per_model' | 'global'
- `USE_STACKING`: True | False
- `CALIBRATION_ON`: True | False
- `KMIN`, `KMAX`: bounds for stability-selected set size

### Ablations
`scripts/run_ablations.py` sweeps the grid and writes results under `output/ablations/` and an aggregated
`ablations_summary.csv`.
