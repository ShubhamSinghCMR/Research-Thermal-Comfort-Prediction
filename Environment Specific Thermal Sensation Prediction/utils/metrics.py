from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def calculate_accuracy(y_true, y_pred, tolerance=0.5):
    absolute_errors = np.abs(np.array(y_true) - np.array(y_pred))
    return (absolute_errors <= tolerance).mean() if len(y_true) > 0 else 0.0

def evaluate_predictions(y_true, y_pred, tolerance=0.5):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    acc = calculate_accuracy(y_true, y_pred, tolerance)
    resid = np.array(y_true) - np.array(y_pred)
    resid_std = np.std(resid)
    mbe = np.mean(resid)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Accuracy": acc, "Residual_STD": resid_std, "MBE": mbe}
