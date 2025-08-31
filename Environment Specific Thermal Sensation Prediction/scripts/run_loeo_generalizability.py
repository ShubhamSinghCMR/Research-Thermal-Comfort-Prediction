
# scripts/run_loeo_generalizability.py  (right-edge padding fix v5; colors unchanged)
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
from typing import Dict
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import evaluate_predictions
from utils.preprocess import build_fold_preprocessor

OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
DATA_XLSX = PROJECT_ROOT / "dataset" / "input_dataset.xlsx"
SEED = 42

METRICS = ["RMSE", "MAE", "R2", "Accuracy"]
LOWER_BETTER = {"RMSE": True, "MAE": True, "R2": False, "Accuracy": False}

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def collect_envs() -> list:
    xls = pd.ExcelFile(DATA_XLSX); return xls.sheet_names

def load_env_dataframe(env_name: str) -> pd.DataFrame:
    return pd.read_excel(DATA_XLSX, sheet_name=env_name)

def detect_target(df: pd.DataFrame) -> str:
    cand = [c for c in df.columns if "Given" in c and "TSV" in c]
    return cand[0] if cand else "Given Final TSV"


def _compute_xlim(values, metric):
    if not values:
        return (0, 1)
    vmin = min(values); vmax = max(values)
    # default xmin
    if LOWER_BETTER.get(metric, True):
        xmin = 0.0
    else:
        span0 = vmax - vmin
        xmin = min(0.0, vmin - 0.05 * (span0 if span0>0 else (abs(vmax) if vmax!=0 else 1.0)))
    # span if axis starts at xmin
    span = max(1e-8, vmax - xmin)
    # robust right padding: works even when bars are very close together (small span)
    # ensures at least ~8% of scale + constant 0.04 units, or 12% of span — whichever is larger
    pad_abs = 0.08 * max(1.0, abs(vmax)) + 0.04
    pad_rel = 0.12 * span
    pad = max(pad_abs, pad_rel)
    xmax = vmax + pad
    return xmin, xmax


def _nice_barh(ax, labels, values, metric, title=None, xshare=None):
    palette = cm.get_cmap("Pastel1")
    colors = [palette(i % palette.N) for i in range(len(labels))]
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    better = "lower" if LOWER_BETTER.get(metric, True) else "higher"
    ax.set_xlabel(f"{metric} ({better} is better)")
    if values:
        xmin, xmax = _compute_xlim(values, metric)
        if xshare is not None: xmax = max(xmax, xshare)
        ax.set_xlim(xmin, xmax)
        delta = (xmax - xmin)
        for b, v in zip(bars, values):
            ax.text(b.get_width() + 0.02*delta, b.get_y()+b.get_height()/2, f"{v:.3f}",
                    va="center", ha="left", fontsize=10, color="black", clip_on=False)
    if title: ax.set_title(title)
    ax.margins(x=0.03, y=0.03)

def _plot_ranked(values_dict, title, metric, out_png: Path):
    items = sorted(values_dict.items(), key=lambda kv: kv[1], reverse=not LOWER_BETTER[metric])
    labels = [k.replace("_"," ").title() for k,_ in items]; values = [v for _,v in items]
    fig, ax = plt.subplots(figsize=(12, max(4.2, 0.38*len(labels)+1))); fig.patch.set_facecolor("white")
    _nice_barh(ax, labels, values, metric, title)
    fig.tight_layout(pad=2.0); plt.subplots_adjust(left=0.35, right=0.95, top=0.90, bottom=0.15)
    fig.savefig(out_png, dpi=260); plt.close(fig)

def get_models() -> Dict[str, object]:
    M = {}
    M["linear_regression"] = LinearRegression()
    M["ridge"] = Ridge()
    M["lasso"] = Lasso(random_state=SEED)
    M["svr_rbf"] = SVR()
    M["knn_distance"] = KNeighborsRegressor(weights="distance")
    M["random_forest"] = RandomForestRegressor(n_estimators=400, random_state=SEED)
    M["extra_trees"] = ExtraTreesRegressor(n_estimators=500, random_state=SEED)
    M["gradient_boosting"] = GradientBoostingRegressor(random_state=SEED)
    M["adaboost"] = AdaBoostRegressor(random_state=SEED)
    M["mlp"] = MLPRegressor(hidden_layer_sizes=(128,64), random_state=SEED, max_iter=500)
    if HAS_XGB: M["xgboost_single"] = XGBRegressor(random_state=SEED, n_estimators=500, max_depth=6, learning_rate=0.05,
                                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist")
    if HAS_LGBM: M["lightgbm_single"] = LGBMRegressor(random_state=SEED, n_estimators=600, verbosity=-1)
    base_keys = [k for k in ["linear_regression","ridge","svr_rbf","random_forest","extra_trees","gradient_boosting"] if k in M]
    estimators = [(k, M[k]) for k in base_keys]
    M["stacked_ensemble"] = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    return M

def _align_like_train(df: pd.DataFrame, train_cols: list) -> pd.DataFrame:
    aligned = df.copy()
    for c in train_cols:
        if c not in aligned.columns: aligned[c] = np.nan
    return aligned[train_cols]

def main():
    _ensure_dir(PLOTS_DIR)
    envs = collect_envs()

    env_frames = {}
    for env in envs:
        df = load_env_dataframe(env).copy()
        target = detect_target(df)
        X_cols = [c for c in df.columns if c != target]
        env_frames[env] = {"X": df[X_cols], "y": df[target].values, "target": target, "X_cols": X_cols}

    loeo_rows = []; models = get_models()

    for test_env in envs:
        train_envs = [e for e in envs if e != test_env]
        X_train = pd.concat([env_frames[e]["X"] for e in train_envs], axis=0, ignore_index=True)
        y_train = np.concatenate([env_frames[e]["y"] for e in train_envs], axis=0)
        X_test  = env_frames[test_env]["X"]; y_test  = env_frames[test_env]["y"]

        pre, transform_df, num_cols, cat_cols, _ = build_fold_preprocessor(X_train)
        train_cols = list(X_train.columns)
        X_test_aligned = _align_like_train(X_test, train_cols)

        Xtr = transform_df(X_train); Xte = transform_df(X_test_aligned)

        per_env_maps = {m:{} for m in METRICS}
        for name, model in models.items():
            model.fit(Xtr, y_train); y_pred = model.predict(Xte)
            met = evaluate_predictions(y_test, y_pred, tolerance=0.5)
            for m in METRICS: per_env_maps[m][name] = float(met[m])
            loeo_rows.append({"TestEnv": test_env, "Model": name, **{k: float(met[k]) for k in METRICS}})

        for m in METRICS:
            _plot_ranked(per_env_maps[m], f"LOEO ranking — test on {test_env}", m,
                         PLOTS_DIR / f"loeo_ranking_{test_env.replace(' ','_')}_{m}.png")

    loeo_df = pd.DataFrame(loeo_rows); loeo_df.to_csv(OUTPUT_DIR / "loeo_metrics_all_metrics.csv", index=False)

    for m in METRICS:
        avg = loeo_df.groupby("Model")[m].mean().sort_values(ascending=LOWER_BETTER[m])
        _plot_ranked(dict(avg), f"Model ranking for LOEO generalizability — {m}", m,
                     PLOTS_DIR / f"loeo_global_ranking_{m}.png")

if __name__ == "__main__":
    main()
