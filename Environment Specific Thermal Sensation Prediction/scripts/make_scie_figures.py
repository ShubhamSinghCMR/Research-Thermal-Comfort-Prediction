#!/usr/bin/env python3
"""
Generate additional SCIE-strengthening figures from the existing project outputs.

Figures:
1) Residual boxplots per TSV class (stacked ensemble OOF) -> output/plots/{env}_residual_boxplots.png
2) Feature stability (selection frequency across base models) -> output/plots/{env}_feature_stability.png
3) Friedman-style rank bar (uses provided Rank per model) -> output/plots/{env}_average_ranks.png
4) Wilcoxon p-values vs. Stacked Ensemble with Holm line -> output/plots/{env}_wilcoxon_pvalues.png

Assumptions about existing outputs:
- Stacked ensemble OOF predictions per environment:
  output/{Env}_predictions.csv
  Must contain 'Given Final TSV' and 'TSV_Predicted' columns.

- Selected features per model per environment:
  output/{Env}_{model}_selected_features.csv
  Must contain columns: 'Feature' and 'Stability_Freq'. If not present and the
  file has only 'Feature', we treat presence as freq=1 and sum across models.

- Model ranks:
  output/comparison_all_models_ranked.csv
  Must contain columns: Environment, Model, Rank (lower is better).

- Wilcoxon vs. main (stacked ensemble):
  output/{Env}_stats_main_vs_baselines_all_metrics.csv
  Must contain columns: Baseline, p_raw. (We plot p_raw per baseline.)

Usage:
  python scripts/make_scie_figures.py --env Classroom
  python scripts/make_scie_figures.py --env Hostel
  python scripts/make_scie_figures.py --env "Workshop_or_laboratory"
  python scripts/make_scie_figures.py --env all

Note: Uses matplotlib only, no explicit colors, one figure per plot.
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUT_DIR = os.path.join(PROJECT, "output")
PLOTS_DIR = os.path.join(OUT_DIR, "scie_figures/")

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def env_file_prefix(env):
    return env

def plot_residual_boxplots(env):
    fn = os.path.join(OUT_DIR, f"{env_file_prefix(env)}_predictions.csv")
    if not os.path.exists(fn):
        print(f"[skip] No predictions file for {env}: {fn}")
        return
    df = pd.read_csv(fn)
    if not {"Given Final TSV", "TSV_Predicted"}.issubset(df.columns):
        print(f"[skip] Missing required columns in {fn}")
        return
    y_true = df["Given Final TSV"].astype(float).values
    y_pred = df["TSV_Predicted"].astype(float).values
    residuals = y_pred - y_true
    tsv_bins = np.clip(np.rint(y_true), -3, 3).astype(int)
    groups, labels = [], []
    for cls in range(-3, 4):
        r = residuals[tsv_bins == cls]
        if r.size == 0:
            r = np.array([np.nan])
        groups.append(r)
        labels.append(str(cls))
    plt.figure(figsize=(8,5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("True TSV (rounded classes)")
    plt.ylabel("Residual (prediction − truth)")
    plt.title(f"Residuals per TSV Class — {env}")
    _ensure_dir(PLOTS_DIR)
    out = os.path.join(PLOTS_DIR, f"{env}_residual_boxplots_02.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[ok] Saved {out}")

def plot_feature_stability(env, top_k=20):
    pattern = os.path.join(OUT_DIR, f"{env}_*_selected_features.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"[skip] No selected_features files for {env}")
        return
    rows = []
    for f in files:
        try:
            raw = pd.read_csv(f)
            cols_lower = [c.lower() for c in raw.columns]
            if 'feature' in cols_lower and 'stability_freq' in cols_lower:
                # normalize column names
                col_map = {}
                for c in raw.columns:
                    cl = c.lower()
                    if cl == 'feature': col_map[c] = 'Feature'
                    if cl == 'stability_freq': col_map[c] = 'Stability_Freq'
                tmp = raw.rename(columns=col_map)[['Feature','Stability_Freq']].copy()
                tmp = tmp.rename(columns={'Stability_Freq':'freq'})
            elif 'feature' in cols_lower and len(raw.columns) == 1:
                tmp = raw.copy()
                tmp.columns = ['Feature']
                tmp['freq'] = 1.0
            else:
                raise ValueError(f"Unexpected columns in {os.path.basename(f)}: {list(raw.columns)}")
            rows.append(tmp)
        except Exception as e:
            print(f"[warn] {f} could not be read: {e}")
    if not rows:
        print(f"[skip] No usable feature files for {env}")
        return
    df = pd.concat(rows, ignore_index=True)
    agg = df.groupby("Feature", as_index=False)["freq"].sum().sort_values("freq", ascending=False)
    top = agg.head(top_k)
    plt.figure(figsize=(10, max(4, 0.35*len(top))))
    plt.barh(top["Feature"][::-1], top["freq"][::-1])
    plt.xlabel("Selection Frequency (sum across models/folds)")
    plt.title(f"Feature Stability — {env}")
    _ensure_dir(PLOTS_DIR)
    out = os.path.join(PLOTS_DIR, f"{env}_feature_stability.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[ok] Saved {out}")

def plot_average_ranks(env):
    fn = os.path.join(OUT_DIR, "comparison_all_models_ranked.csv")
    if not os.path.exists(fn):
        print(f"[skip] No comparison_all_models_ranked.csv")
        return
    df = pd.read_csv(fn)
    # Try exact match; else normalize spaces/slashes as in repo
    sub = df[df["Environment"] == (env if env != "Workshop_or_laboratory" else "Workshop/Laboratory")].copy()
    if sub.empty:
        alt = df["Environment"].str.replace(" ", "_").str.replace("/", "_").str.lower()
        sub = df[alt.eq(env.lower())].copy()
    if sub.empty:
        print(f"[skip] No rows in comparison_all_models_ranked for {env}")
        return
    sub = sub[["Model","Rank"]].sort_values("Rank", ascending=True)
    plt.figure(figsize=(9, max(4, 0.35*len(sub))))
    plt.barh(sub["Model"], sub["Rank"])
    plt.xlabel("Average Rank (lower is better)")
    plt.title(f"Model Ranks — {env}")
    _ensure_dir(PLOTS_DIR)
    out = os.path.join(PLOTS_DIR, f"{env}_average_ranks.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[ok] Saved {out}")

def plot_wilcoxon(env, alpha=0.05):
    fn = os.path.join(OUT_DIR, f"{env}_stats_main_vs_baselines_all_metrics.csv")
    if not os.path.exists(fn):
        print(f"[skip] No stats file for {env}: {fn}")
        return
    df = pd.read_csv(fn)
    if not {"Baseline","p_raw"}.issubset(df.columns):
        print(f"[skip] Missing columns in {fn}")
        return
    sub = df[["Baseline","p_raw"]].sort_values("p_raw", ascending=True).reset_index(drop=True)
    m = len(sub)
    # Holm thresholds for sorted p-values: alpha/(m-i)
    holm = np.array([alpha/(m - i) for i in range(m)], dtype=float)
    idx = np.where(sub["p_raw"].values <= holm)[0]
    ref = holm[idx.max()] if idx.size > 0 else 0.0

    plt.figure(figsize=(10, 5 + 0.2*m))
    plt.bar(range(m), sub["p_raw"].values)
    plt.xticks(range(m), sub["Baseline"].values, rotation=45, ha="right")
    plt.axhline(ref, linestyle="--", linewidth=1)
    plt.ylabel("Wilcoxon p-value (vs. Stacked Ensemble)")
    plt.title(f"Wilcoxon Tests (Holm reference) — {env}")
    _ensure_dir(PLOTS_DIR)
    out = os.path.join(PLOTS_DIR, f"{env}_wilcoxon_pvalues_02.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[ok] Saved {out}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,
                        help='Environment name: Classroom | Hostel | Workshop_or_laboratory | all')
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    envs = [args.env]
    if args.env.lower() == "all":
        envs = ["Classroom", "Hostel", "Workshop_or_laboratory"]

    for env in envs:
        print(f"=== {env} ===")
        plot_residual_boxplots(env)
        plot_feature_stability(env, top_k=args.top_k)
        plot_average_ranks(env)
        plot_wilcoxon(env, alpha=args.alpha)

if __name__ == "__main__":
    main()

