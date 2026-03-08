from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from scipy.stats import kendalltau
import numpy as np

def calculate_accuracy(y_true, y_pred, tolerance=0.5):
    absolute_errors = np.abs(np.array(y_true) - np.array(y_pred))
    return (absolute_errors <= tolerance).mean() if len(y_true) > 0 else 0.0


def _ordinal_metrics_from_continuous(y_true, y_pred):
    """Compute ordinal-aware metrics from continuous predictions (clipped to [-3,3], then rounded to class)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {
            "Accuracy_exact": np.nan,
            "F1_macro": np.nan,
            "Kappa": np.nan,
            "Kappa_quadratic": np.nan,
            "Kendall_tau_b": np.nan,
        }
    y_pred_clipped = np.clip(y_pred, -3.0, 3.0)
    y_pred_class = np.round(y_pred_clipped).astype(int)
    y_true_class = np.clip(np.round(y_true), -3, 3).astype(int)
    accuracy_exact = accuracy_score(y_true_class, y_pred_class)
    f1_macro = f1_score(y_true_class, y_pred_class, average="macro", zero_division=0)
    kappa_unweighted = cohen_kappa_score(y_true_class, y_pred_class)
    kappa_quadratic = cohen_kappa_score(y_true_class, y_pred_class, weights="quadratic")
    tau_b, _ = kendalltau(y_true_class, y_pred_class)
    return {
        "Accuracy_exact": accuracy_exact,
        "F1_macro": f1_macro,
        "Kappa": kappa_unweighted,
        "Kappa_quadratic": kappa_quadratic,
        "Kendall_tau_b": float(tau_b) if not np.isnan(tau_b) else np.nan,
    }


def evaluate_predictions(y_true, y_pred, tolerance=0.5):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    acc = calculate_accuracy(y_true, y_pred, tolerance)
    resid = np.array(y_true) - np.array(y_pred)
    resid_std = np.std(resid)
    mbe = np.mean(resid)
    out = {"RMSE": rmse, "MAE": mae, "R2": r2, "Accuracy": acc, "Residual_STD": resid_std, "MBE": mbe}
    out.update(_ordinal_metrics_from_continuous(y_true, y_pred))
    return out
