"""
Metrics Module for Thermal Comfort Prediction
=============================================

This module provides evaluation metrics for assessing thermal comfort prediction performance.
It includes both standard regression metrics and custom accuracy calculations that account
for the specific requirements of thermal comfort prediction.

Key Metrics:
------------
1. Standard Regression Metrics
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² Score (Coefficient of Determination)
   - MBE (Mean Bias Error)

2. Custom Metrics
   - Accuracy within tolerance
   - Residual standard deviation

Dependencies:
-------------
- scikit-learn: Standard regression metrics
- numpy: Numerical operations

Main Functions:
---------------
- calculate_accuracy: Custom accuracy within tolerance
- evaluate_predictions: Comprehensive evaluation with multiple metrics
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def calculate_accuracy(y_true, y_pred, tolerance=0.5):
    """
    Calculate prediction accuracy within a specified tolerance range.
    
    A prediction is considered correct if it falls within ±tolerance
    of the true value.

    Parameters
    ----------
    y_true : array-like
        True TSV (Thermal Sensation Vote) values
    y_pred : array-like
        Predicted TSV values
    tolerance : float, default=0.5
        Acceptable deviation range for predictions
        
    Returns
    -------
    float
        Accuracy score between 0 and 1
    """
    absolute_errors = np.abs(np.array(y_true) - np.array(y_pred))
    correct_predictions = (absolute_errors <= tolerance).sum()
    total_predictions = len(y_true)
    
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0


def evaluate_predictions(y_true, y_pred, tolerance=0.5):
    """
    Evaluate predictions using multiple performance metrics.

    Parameters
    ----------
    y_true : array-like
        True TSV values
    y_pred : array-like
        Predicted TSV values
    tolerance : float, default=0.5
        Tolerance for accuracy calculation
        
    Returns
    -------
    dict
        Dictionary containing multiple metrics:
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - R2: R-squared score
        - Accuracy: Fraction of predictions within tolerance
        - Residual_STD: Standard deviation of residuals
        - MBE: Mean Bias Error
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    accuracy = calculate_accuracy(y_true, y_pred, tolerance)
    
    residuals = np.array(y_true) - np.array(y_pred)
    residual_std = np.std(residuals)
    mbe = np.mean(residuals)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Accuracy": accuracy,
        "Residual_STD": residual_std,
        "MBE": mbe
    }
