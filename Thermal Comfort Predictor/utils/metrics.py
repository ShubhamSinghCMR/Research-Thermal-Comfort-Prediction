from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def calculate_accuracy(y_true, y_pred, tolerance=0.5):
    """
    Calculate prediction accuracy within a tolerance range.
    
    Parameters
    ----------
    y_true : array-like
        True TSV values
    y_pred : array-like
        Predicted TSV values
    tolerance : float, default=0.5
        Tolerance range for correct prediction.
    
    Returns
    -------
    float
        Accuracy score (fraction of predictions within tolerance)
    """
    absolute_errors = np.abs(np.array(y_true) - np.array(y_pred))
    correct_predictions = (absolute_errors <= tolerance).sum()
    total_predictions = len(y_true)
    
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0


def evaluate_predictions(y_true, y_pred, tolerance=0.5):
    """
    Evaluate predictions using multiple metrics.
    
    Returns
    -------
    dict
        RMSE, MAE, R2, Accuracy (within tolerance), Residual_STD, and MBE
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    accuracy = calculate_accuracy(y_true, y_pred, tolerance)
    
    residuals = np.array(y_true) - np.array(y_pred)
    residual_std = np.std(residuals)
    mbe = np.mean(residuals)  # Mean Bias Error

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Accuracy": accuracy,
        "Residual_STD": residual_std,
        "MBE": mbe
    }
