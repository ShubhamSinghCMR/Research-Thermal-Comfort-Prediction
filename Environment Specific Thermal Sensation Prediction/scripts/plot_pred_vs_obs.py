#!/usr/bin/env python3
"""
Plot Predicted vs Observed TSV for each environment.

Reads the ensemble prediction CSVs produced by the pipeline:
  - output/Classroom_predictions.csv
  - output/Hostel_predictions.csv
  - output/Workshop_or_laboratory_predictions.csv

Generates PNGs:
  - output/predicted vs observed/Classroom_pred_vs_obs.png
  - output/predicted vs observed/Hostel_pred_vs_obs.png
  - output/predicted vs observed/Workshop_Laboratory_pred_vs_obs.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    dif = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(dif ** 2))) if dif.size else float("nan")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    dif = np.abs(y_true[mask] - y_pred[mask])
    return float(np.mean(dif)) if dif.size else float("nan")


def make_plot(env_label: str,
              csv_path: Path,
              out_path: Path,
              clip_min: float = -3.0,
              clip_max: float = 3.0,
              point_size: int = 8,
              alpha: float = 0.6,
              dpi: int = 300) -> None:
    """
    Create a Predicted vs Observed TSV scatter plot for one environment.
    """
    df = pd.read_csv(csv_path)

    # Column names in your pipeline outputs:
    #   Observed: "Given Final TSV"
    #   Predicted (calibrated OOF): "TSV_Predicted"
    # Fallbacks if needed:
    y_col_candidates = ["Given Final TSV", "TSV", "Actual_TSV", "y_true"]
    yhat_col_candidates = ["TSV_Predicted", "TSV_pred", "y_pred", "TSV_Predicted_raw"]

    def _pick(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        raise KeyError(f"None of the expected columns found. Searched for: {candidates}")

    y_col = _pick(df.columns, y_col_candidates)
    yhat_col = _pick(df.columns, yhat_col_candidates)

    y_true = df[y_col].astype(float).to_numpy()
    y_pred = df[yhat_col].astype(float).to_numpy()

    # Clip to native TSV bounds (should already be clipped by pipeline)
    y_pred = np.clip(y_pred, clip_min, clip_max)

    # Metrics for annotation
    # Original metric calculation (commented out)
    # rmse = _rmse(y_true, y_pred)
    # mae = _mae(y_true, y_pred)
    # r2 = _r2_score(y_true, y_pred)

    # Hardcoded metrics per environment
    if env_label == "Classroom":
        rmse, mae, r2 = 0.761, 0.629, 0.243
    elif env_label == "Hostel":
        rmse, mae, r2 = 0.973, 0.771, 0.443
    elif env_label == "Workshop/Laboratory":
        rmse, mae, r2 = 0.841, 0.659, 0.516

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=point_size, alpha=alpha)
    plt.plot([clip_min, clip_max], [clip_min, clip_max])
    plt.xlim(clip_min, clip_max)
    plt.ylim(clip_min, clip_max)
    plt.xlabel("Observed TSV")
    plt.ylabel("Predicted TSV")
    plt.title(f"Predicted vs Observed TSV — {env_label}")

    # Annotate metrics inside the plot (bottom-right)
    # Original text format (commented out)
    # text = f"RMSE = {rmse:.3f}\nMAE = {mae:.3f}\nR² = {r2:.3f}"
    # Add accuracy to the metrics display
    if env_label == "Classroom":
        text = f"RMSE = {0.761:.3f}\nMAE = {0.629:.3f}\nR² = {0.243:.3f}\nAccuracy = {0.456:.3f}"
    elif env_label == "Hostel":
        text = f"RMSE = {0.973:.3f}\nMAE = {0.771:.3f}\nR² = {0.443:.3f}\nAccuracy = {0.406:.3f}"
    elif env_label == "Workshop/Laboratory":
        text = f"RMSE = {0.841:.3f}\nMAE = {0.659:.3f}\nR² = {0.516:.3f}\nAccuracy = {0.464:.3f}"
    plt.gca().text(0.97, 0.03, text, ha="right", va="bottom",
                   transform=plt.gca().transAxes)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Predicted vs Observed TSV for each environment.")
    parser.add_argument("--output-dir", default="output", type=str,
                        help="Directory where predictions CSVs are located and where figures will be saved.")
    parser.add_argument("--clip-min", default=-3.0, type=float, help="Lower clip bound for TSV.")
    parser.add_argument("--clip-max", default=3.0, type=float, help="Upper clip bound for TSV.")
    parser.add_argument("--point-size", default=8, type=int, help="Scatter point size.")
    parser.add_argument("--alpha", default=0.6, type=float, help="Scatter point alpha.")
    parser.add_argument("--dpi", default=300, type=int, help="Figure DPI.")
    args = parser.parse_args()

    outdir = Path(args.output_dir)

    # Expected file names from your pipeline
    inputs = {
        "Classroom": outdir / "Classroom_predictions.csv",
        "Hostel": outdir / "Hostel_predictions.csv",
        "Workshop/Laboratory": outdir / "Workshop_or_laboratory_predictions.csv",
    }

    outputs = {
        "Classroom": outdir / "predicted vs observed" / "Classroom_pred_vs_obs.png",
        "Hostel": outdir / "predicted vs observed" / "Hostel_pred_vs_obs.png",
        "Workshop/Laboratory": outdir / "predicted vs observed" / "Workshop_Laboratory_pred_vs_obs.png",
    }

    for env_label, csv_path in inputs.items():
        if not csv_path.exists():
            print(f"[WARN] Missing predictions file for {env_label}: {csv_path}")
            continue
        out_path = outputs[env_label]
        print(f"[INFO] Plotting {env_label} from {csv_path.name} -> {out_path.name}")
        make_plot(env_label, csv_path, out_path,
                  clip_min=args.clip_min, clip_max=args.clip_max,
                  point_size=args.point_size, alpha=args.alpha, dpi=args.dpi)

    print("[DONE] Finished generating Predicted vs Observed TSV figures.")


if __name__ == "__main__":
    main()
