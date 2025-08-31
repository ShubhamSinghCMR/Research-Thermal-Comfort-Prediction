
# utils/stat_tests.py

import numpy as np
import pandas as pd
from scipy import stats

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    mae = np.mean(np.abs(resid))
    mse = np.mean(resid**2)
    rmse = float(np.sqrt(mse))
    if np.var(y_true) < 1e-12:
        r2 = float("nan")
    else:
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        r2 = 1.0 - (ss_res / ss_tot)
    acc = float(np.mean(np.abs(resid) <= 0.5))  # tolerance-accuracy used in repo
    return {"RMSE": rmse, "MAE": float(mae), "R2": float(r2), "Accuracy": acc}

def holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    for i, idx in enumerate(order):
        adj[idx] = min((m - i) * pvals[idx], 1.0)
    for i in range(1, m):
        if adj[order[i]] < adj[order[i-1]]:
            adj[order[i]] = adj[order[i-1]]
    return adj

def wilcoxon_main_vs_baseline(abs_err_main, abs_err_base):
    d = np.asarray(abs_err_base, dtype=float) - np.asarray(abs_err_main, dtype=float)
    if np.allclose(d, 0.0):
        return {"wilcoxon_stat": 0.0, "pvalue": 1.0}
    try:
        stat, p = stats.wilcoxon(d, zero_method="wilcox", alternative="greater")
    except Exception:
        stat, p = stats.wilcoxon(d)
    return {"wilcoxon_stat": float(stat), "pvalue": float(p)}

def friedman_test(matrix):
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 3:
        return {"friedman_stat": float("nan"), "friedman_p": float("nan")}
    args = [matrix[:, j] for j in range(matrix.shape[1])]
    stat, p = stats.friedmanchisquare(*args)
    return {"friedman_stat": float(stat), "friedman_p": float(p)}
