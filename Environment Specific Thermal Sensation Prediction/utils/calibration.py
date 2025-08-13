import numpy as np

def compute_bias_and_qhat(y_true, y_pred, alpha=0.10):
    resid = np.array(y_true) - np.array(y_pred)
    bias = float(np.mean(resid))
    qhat = float(np.quantile(np.abs(resid - bias), 1.0 - alpha))
    return bias, qhat

def apply_calibration(pred, bias, clip_range=(-3,3)):
    val = float(pred + bias)
    return min(max(val, clip_range[0]), clip_range[1])
