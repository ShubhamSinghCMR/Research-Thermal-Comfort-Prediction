#!/usr/bin/env python3
"""
Compute Friedman test (chi2, p-value) per environment for manuscript Section 5.10.1.
Uses OOF prediction files: matrix (n_samples x n_models) of absolute error; each sample
is a block. Output: output/friedman_test_results.csv (Environment, friedman_chi2, friedman_p).
These CSV values are the correct reference for the manuscript.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stat_tests import friedman_test

OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_XLSX = PROJECT_ROOT / "dataset" / "input_dataset.xlsx"

try:
    from features.feature_engineering import get_all_sheet_names, load_environment_sheet
except Exception:
    get_all_sheet_names = None
    load_environment_sheet = None


def _get_y_true_y_pred(df):
    for c in ["Given Final TSV", "Given_Final_TSV"]:
        if c in df.columns:
            y_true_col = c
            break
    else:
        hits = [c for c in df.columns if "Given" in c and "TSV" in c]
        y_true_col = hits[0] if hits else None
    if y_true_col is None:
        raise ValueError("No ground-truth TSV column")
    for c in ["TSV_Predicted", "oof_pred", "TSV_Predicted_raw"]:
        if c in df.columns:
            y_pred_col = c
            break
    else:
        hits = [c for c in df.columns if "Predicted" in c or "pred" in c.lower()]
        y_pred_col = hits[0] if hits else None
    if y_pred_col is None:
        raise ValueError("No prediction column")
    y_true = np.asarray(df[y_true_col], dtype=float)
    y_pred = np.asarray(df["TSV_Predicted"] if "TSV_Predicted" in df.columns else df[y_pred_col], dtype=float)
    return y_true, np.clip(y_pred, -3.0, 3.0)


def get_prediction_files(env_name):
    env_tag = env_name.replace(" ", "_")
    out = []
    p = OUTPUT_DIR / f"{env_tag}_predictions.csv"
    if p.exists():
        out.append(p)
    for p in OUTPUT_DIR.glob(f"{env_tag}_*_predictions.csv"):
        out.append(p)
    return sorted(out)


def load_env_data(env_name):
    if load_environment_sheet is None:
        return pd.read_excel(DATA_XLSX, sheet_name=env_name)
    return load_environment_sheet(env_name, str(DATA_XLSX))


def main():
    if not DATA_XLSX.exists():
        print("Dataset not found:", DATA_XLSX)
        sys.exit(1)
    sheets = get_all_sheet_names(str(DATA_XLSX)) if get_all_sheet_names else pd.ExcelFile(DATA_XLSX).sheet_names
    envs = [s for s in sheets if s in ("Classroom", "Hostel", "Workshop or laboratory")]
    if not envs:
        envs = [s for s in sheets if any(x in s for x in ["Classroom", "Hostel", "Workshop", "Laboratory"])]
    if not envs:
        envs = list(sheets)

    rows = []
    for env_name in envs:
        df = load_env_data(env_name)
        n_rows = len(df)
        paths = get_prediction_files(env_name)
        if len(paths) < 2:
            continue
        abs_errors = []
        for path in paths:
            d = pd.read_csv(path)
            if len(d) != n_rows:
                continue
            y_true, y_pred = _get_y_true_y_pred(d)
            abs_errors.append(np.abs(y_true - y_pred))
        if len(abs_errors) < 2:
            continue
        matrix = np.column_stack(abs_errors)
        res = friedman_test(matrix)
        rows.append({
            "Environment": env_name,
            "friedman_chi2": round(res["friedman_stat"], 2),
            "friedman_p": res["friedman_p"],
        })
        print(f"  {env_name}: chi2 = {rows[-1]['friedman_chi2']}, p = {res['friedman_p']:.2e}")

    if not rows:
        sys.exit(1)
    out_path = OUTPUT_DIR / "friedman_test_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
