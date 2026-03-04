import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `utils` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stat_tests import wilcoxon_main_vs_baseline  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "output"


def _find_main_predictions(env_name: str) -> Path | None:
    """
    Return the stacked-ensemble prediction file for a given environment.
    Assumes naming convention: {Env}_predictions.csv as used in run_main_pipeline.
    """
    env_tag = env_name.replace(" ", "_")
    fn = OUTPUT_DIR / f"{env_tag}_predictions.csv"
    return fn if fn.exists() else None


def _find_baseline_predictions(env_name: str) -> list[tuple[str, Path]]:
    """
    Return list of (baseline_key, path) for all baseline predictions of an environment.
    Baseline naming convention: {Env}_{model}_predictions.csv, excluding the stacked ensemble.
    """
    env_tag = env_name.replace(" ", "_")
    preds = []
    for p in OUTPUT_DIR.glob(f"{env_tag}_*_predictions.csv"):
        name = p.name.removeprefix(f"{env_tag}_").removesuffix("_predictions.csv")
        if name.lower() in {"stacked", "stacked_ensemble"}:
            continue
        if name == "":
            continue
        preds.append((name, p))
    return sorted(preds, key=lambda kv: kv[0])


def _load_tsv_and_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load TSV ground truth and predictions from a prediction CSV.
    Tries several column name variants used in the project.
    """
    df = pd.read_csv(path)
    # True TSV
    true_col_candidates = [
        "Given Final TSV",
        "Given_Final_TSV",
    ]
    y_true = None
    for c in true_col_candidates:
        if c in df.columns:
            y_true = df[c].values
            break
    if y_true is None:
        # try regex-like search
        hits = [c for c in df.columns if "Given" in c and "TSV" in c]
        if not hits:
            raise ValueError(f"No TSV ground-truth column found in {path}")
        y_true = df[hits[0]].values

    # Predictions
    pred_col_candidates = [
        "TSV_Predicted",
        "TSV_Predicted_raw",
        "oof_pred",
    ]
    y_pred = None
    for c in pred_col_candidates:
        if c in df.columns:
            y_pred = df[c].values
            break
    if y_pred is None:
        hits = [c for c in df.columns if "Predicted" in c or "prediction" in c.lower()]
        if not hits:
            raise ValueError(f"No prediction column found in {path}")
        y_pred = df[hits[0]].values

    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def compute_env_stats(env_name: str) -> pd.DataFrame | None:
    """
    For a single environment:
    - Load stacked ensemble predictions as 'main'.
    - For each baseline model's predictions, compute paired Wilcoxon test on absolute errors.
    - Return a DataFrame with columns: Baseline, p_raw.
    """
    main_path = _find_main_predictions(env_name)
    if not main_path:
        print(f"[WARN] No main stacked-ensemble predictions for {env_name}")
        return None

    baselines = _find_baseline_predictions(env_name)
    if not baselines:
        print(f"[WARN] No baseline prediction files for {env_name}")
        return None

    y_true_main, y_pred_main = _load_tsv_and_pred(main_path)
    abs_err_main = np.abs(y_true_main - y_pred_main)

    rows = []
    for key, path in baselines:
        try:
            y_true_b, y_pred_b = _load_tsv_and_pred(path)
            if y_true_b.shape[0] != y_true_main.shape[0]:
                print(f"[WARN] Skipping {key} for {env_name}: different number of samples")
                continue
            abs_err_base = np.abs(y_true_b - y_pred_b)
            stats_dict = wilcoxon_main_vs_baseline(abs_err_main, abs_err_base)
            rows.append(
                {
                    "Baseline": key,
                    "p_raw": stats_dict["pvalue"],
                    "wilcoxon_stat": stats_dict["wilcoxon_stat"],
                }
            )
        except Exception as e:
            print(f"[WARN] Failed Wilcoxon for {env_name} / {key}: {e}")

    if not rows:
        print(f"[WARN] No usable baseline comparisons for {env_name}")
        return None

    return pd.DataFrame(rows)


def main():
    envs = ["Classroom", "Hostel", "Workshop_or_laboratory"]
    out_dir = OUTPUT_DIR

    for env in envs:
        df = compute_env_stats(env)
        if df is None:
            continue
        out_path = out_dir / f"{env}_stats_main_vs_baselines_all_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"[ok] Saved stats for {env} -> {out_path}")


if __name__ == "__main__":
    main()

