
# scripts/run_stats_and_plots.py  (right-edge padding fix v5; colors unchanged)
import os, re
from pathlib import Path
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from features.feature_engineering import get_all_sheet_names, load_environment_sheet
except Exception:
    get_all_sheet_names = None
    load_environment_sheet = None

from utils.stat_tests import regression_metrics, wilcoxon_main_vs_baseline, holm_bonferroni, friedman_test

warnings.filterwarnings("ignore")

OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_ROOT = OUTPUT_DIR / "stats and plots"
DATA_XLSX = PROJECT_ROOT / "dataset" / "input_dataset.xlsx"

STACKED_TAG = "stacked_ensemble"
METRICS = ["RMSE", "MAE", "R2", "Accuracy"]
LOWER_BETTER = {"RMSE": True, "MAE": True, "R2": False, "Accuracy": False}

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _autodiscover_models(env_name: str) -> List[str]:
    env_tag = env_name.replace(" ", "_")
    keys = []
    for p in OUTPUT_DIR.glob(f"{env_tag}_*_predictions.csv"):
        m = re.match(rf"{re.escape(env_tag)}_(.+?)_predictions\.csv$", p.name)
        if m: keys.append(m.group(1))
    return sorted(set(keys))

def _format_model_label(key: str) -> str: return key.replace("_", " ").title()

def _classify_tsv(tsv_array):
    return np.digitize(tsv_array, [-1e9, -0.5, 0.5, 1e9]) - 1


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


def _nice_barh(ax, labels, values, metric, add_title=None, xshare=None):
    # Keep the same color scheme (Pastel1)
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
            # Place label just outside the bar but guaranteed inside axes due to robust padding.
            ax.text(b.get_width() + 0.02*delta, b.get_y()+b.get_height()/2, f"{v:.3f}",
                    va="center", ha="left", fontsize=10, color="black", clip_on=False)

    if add_title: ax.set_title(add_title)
    ax.margins(x=0.03, y=0.03)

def _plot_ranked(values_dict, title, metric, out_png: Path):
    items = sorted(values_dict.items(), key=lambda kv: kv[1], reverse=not LOWER_BETTER[metric])
    labels = [_format_model_label(k) for k,_ in items]; values = [v for _,v in items]
    fig, ax = plt.subplots(figsize=(12, max(4.2, 0.38*len(labels)+1))); fig.patch.set_facecolor("white")
    _nice_barh(ax, labels, values, metric, add_title=title)
    fig.tight_layout(pad=2.0); plt.subplots_adjust(left=0.35, right=0.95, top=0.90, bottom=0.15)
    fig.savefig(out_png, dpi=260); plt.close(fig)

def plot_correlation(df_num: pd.DataFrame, out_png: Path, out_r: Path, out_p: Path):
    cols = df_num.columns; n = len(cols)
    r = np.zeros((n, n)); p = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            rr, pp = stats.pearsonr(df_num[cols[i]].dropna(), df_num[cols[j]].dropna())
            r[i,j] = r[j,i] = rr; p[i,j] = p[j,i] = pp
    flat = p.flatten(); order = np.argsort(flat); alpha = 0.05
    ranked = np.empty_like(flat); ranked[order] = (np.arange(1, len(flat)+1) / len(flat)) * alpha
    sig_mask = (flat <= ranked).reshape(p.shape)
    fig, ax = plt.subplots(figsize=(10,8)); fig.patch.set_facecolor("white")
    im = ax.imshow(r, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(cols); ax.set_yticklabels(cols)
    for i in range(n):
        for j in range(n):
            val = r[i,j]; intensity = im.norm(val)
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=("white" if intensity>0.6 else "black"))
    for (i,j), flag in np.ndenumerate(~sig_mask):
        if flag: ax.text(j, i, "×", ha="center", va="center", fontsize=9, color="red")
    ax.set_title("Pearson correlation (× = not significant at FDR 5%)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("r")
    fig.tight_layout(pad=2.0); plt.subplots_adjust(left=0.18, right=0.92, top=0.92, bottom=0.18)
    fig.savefig(out_png, dpi=260); plt.close(fig)

def load_env_dataframe(env_name: str) -> pd.DataFrame:
    if get_all_sheet_names and load_environment_sheet: return load_environment_sheet(env_name)
    xls = pd.ExcelFile(DATA_XLSX); return pd.read_excel(xls, sheet_name=env_name)

def collect_envs() -> list:
    if get_all_sheet_names: return get_all_sheet_names(str(DATA_XLSX))
    xls = pd.ExcelFile(DATA_XLSX); return xls.sheet_names

def find_prediction_file(env_name: str, key: str) -> Path:
    env_tag = env_name.replace(" ", "_")
    pattern = f"{env_tag}_predictions.csv" if key==STACKED_TAG else f"{env_tag}_{key}_predictions.csv"
    hits = list((OUTPUT_DIR).glob(pattern)); return hits[0] if hits else None

def _plot_confusion_matrix(y_true_cls, y_pred_cls, labels, title, out_png: Path):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    cmx = confusion_matrix(y_true_cls, y_pred_cls, labels=range(len(labels)))
    cmx_norm = cmx / cmx.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.8, 6.8)); fig.patch.set_facecolor("white")
    im = ax.imshow(cmx_norm, cmap="YlGnBu")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Proportion")
    for i in range(len(labels)):
        for j in range(len(labels)):
            frac = cmx_norm[i,j]; tcolor = "white" if im.norm(frac)>0.6 else "black"
            ax.text(j, i, f"{cmx[i,j]:d}\n({frac*100:.1f}%)", ha="center", va="center", fontsize=9, color=tcolor)
    acc = accuracy_score(y_true_cls, y_pred_cls)
    prec = precision_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)
    rec = recall_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)
    f1 = f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)
    ax.set_title(f"{title}\nAcc={acc:.3f} | Prec={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}")
    fig.tight_layout(pad=2.0); plt.subplots_adjust(left=0.16, right=0.92, top=0.90, bottom=0.12)
    fig.savefig(out_png, dpi=260); plt.close(fig)

def main():
    _ensure_dir(PLOTS_ROOT)
    envs = collect_envs()

    # Read comparison_all_models_ranked.csv first as it's our main data source
    ranked_csv = OUTPUT_DIR / "comparison_all_models_ranked.csv"
    if not ranked_csv.exists():
        print(f"[ERROR] Missing ranked comparison file: {ranked_csv}")
        return
        
    df_ranked = pd.read_csv(ranked_csv)
    
    # correlation per environment - using the same prediction files as ranked comparison
    for env in envs:
        env_tag = env.replace(" ", "_")
        f_main = find_prediction_file(env, STACKED_TAG)
        if not f_main or not f_main.exists():
            print(f"[WARN] Missing prediction file for correlation analysis: {env}")
            continue
            
        df = pd.read_csv(f_main)
        base_numeric = ['RATemp', 'MRT', 'Top', 'Air Velo', 'RH']
        base_cats = ['Season', 'Clothing', 'Activity']
        feats = [c for c in (base_numeric if env == "Classroom" else base_numeric + base_cats) if c in df.columns]
        X = df[feats].copy()
        Xnum = X.select_dtypes(include=[np.number])
        if Xnum.shape[1] >= 2:
            plot_correlation(Xnum, PLOTS_ROOT / f"{env_tag}_correlation_heatmap_01.png",
                             PLOTS_ROOT / f"{env_tag}_correlation_matrix.csv",
                             PLOTS_ROOT / f"{env_tag}_correlation_pvalues.csv")

    # Already read ranked_csv above, continue with metric maps
    env_metric_maps = {m:{} for m in METRICS}

    # Populate metric maps from ranked comparison data
    for env in envs:
        env_data = df_ranked[df_ranked['Environment'] == env]
        if env_data.empty:
            # Try with normalized environment name
            env_data = df_ranked[df_ranked['Environment'] == env.replace('_', ' ')]
        if env_data.empty:
            print(f"[WARN] No data found for environment: {env}")
            continue
            
        for metric in METRICS:
            if metric not in env_data.columns:
                print(f"[WARN] Metric {metric} not found in ranked comparison data")
                continue
            env_metric_maps[metric][env] = dict(zip(env_data['Model'], env_data[metric]))

    # Generate confusion matrices using the same prediction files as ranked comparison
    for env in envs:
        env_tag = env.replace(" ", "_")
        f_main = find_prediction_file(env, STACKED_TAG)
        if not f_main or not f_main.exists():
            print(f"[WARN] Missing prediction file for confusion matrix: {env}")
            continue
            
        try:
            # Use the same file that was used for the ranked comparison
            df_main = pd.read_csv(f_main)
            y_true = df_main.filter(regex="Given Final TSV|Given_Final_TSV|Given.*TSV", axis=1).iloc[:,0].values
            y_pred_main = df_main["TSV_Predicted"].values if "TSV_Predicted" in df_main.columns else \
                        df_main["oof_pred"].values if "oof_pred" in df_main.columns else \
                        df_main[[c for c in df_main.columns if "Predicted" in c][0]].values
            
            y_true_cls = _classify_tsv(y_true)
            y_pred_cls = _classify_tsv(y_pred_main)
            _plot_confusion_matrix(y_true_cls, y_pred_cls, ["Cool","Neutral","Warm"],
                                   f"Confusion Matrix (3-class TSV) — {env}",
                                PLOTS_ROOT / f"{env_tag}_confusion_matrix_STACKED_01.png")
        except Exception as e:
            print(f"[WARN] Confusion matrix skipped for {env}: {e}")

    # per-env rankings and 3-in-1
    for metric in METRICS:
        envs_present = [e for e in envs if e in env_metric_maps[metric]]
        for env in envs_present:
            env_tag = env.replace(" ", "_")
            _plot_ranked(env_metric_maps[metric][env], f"Model ranking by {metric} — {env}", metric,
                         PLOTS_ROOT / f"{env_tag}_ranking_{metric}.png")
        if len(envs_present) == 3:
            # compute shared xmax using robust padding across all values
            all_vals = []
            for e in envs_present: all_vals.extend(list(env_metric_maps[metric][e].values()))
            _, xshare = _compute_xlim(all_vals, metric)
            fig, axes = plt.subplots(1, 3, figsize=(21, 6)); fig.patch.set_facecolor("white")
            for i, env in enumerate(envs_present):
                items = sorted(env_metric_maps[metric][env].items(), key=lambda kv: kv[1], reverse=not LOWER_BETTER[metric])
                labels = [_format_model_label(k) for k,_ in items]; values = [v for _,v in items]
                _nice_barh(axes[i], labels, values, metric, add_title=env, xshare=xshare)
            # fig.suptitle(f"Per‑environment model rankings — {metric}", fontsize=14)
            fig.tight_layout(pad=2.0); plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.10, wspace=0.35)
            fig.savefig(PLOTS_ROOT / f"cv_summary_3in1_env_rankings_{metric}_02.png", dpi=260); plt.close(fig)

if __name__ == "__main__":
    main()
