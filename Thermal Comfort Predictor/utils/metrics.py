"""
Metrics Module for Thermal Comfort Prediction
========================================

This module provides evaluation metrics for assessing thermal comfort prediction performance.
It includes both standard regression metrics and custom accuracy calculations that account
for the specific requirements of thermal comfort prediction.

Key Metrics:
----------
1. Standard Regression Metrics
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² Score (Coefficient of Determination)
   - MBE (Mean Bias Error)

2. Custom Metrics
   - Accuracy within tolerance
   - Residual standard deviation

Dependencies:
-----------
- scikit-learn: Standard regression metrics
- numpy: Numerical operations

Main Functions:
------------
- calculate_accuracy: Custom accuracy within tolerance
- evaluate_predictions: Comprehensive evaluation with multiple metrics
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def calculate_accuracy(y_true, y_pred, tolerance=0.5):
    """
    Calculate prediction accuracy within a specified tolerance range.
    
    This function considers a prediction correct if it falls within
    ±tolerance of the true value. This is particularly important for
    thermal comfort prediction where small deviations are acceptable.
    
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
        
    Notes
    -----
    - A tolerance of 0.5 is standard in thermal comfort research
    - Returns 0.0 if there are no predictions (empty input)
    - Handles numpy arrays and pandas series
    """
    absolute_errors = np.abs(np.array(y_true) - np.array(y_pred))
    correct_predictions = (absolute_errors <= tolerance).sum()
    total_predictions = len(y_true)
    
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0


def evaluate_predictions(y_true, y_pred, tolerance=0.5):
    """
    Evaluate predictions using multiple performance metrics.
    
    This function provides a comprehensive evaluation of thermal comfort
    predictions using both standard regression metrics and custom metrics
    specific to thermal comfort assessment.
    
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
        - RMSE: Root Mean Squared Error (overall error magnitude)
        - MAE: Mean Absolute Error (average absolute deviation)
        - R2: R-squared score (proportion of variance explained)
        - Accuracy: Fraction of predictions within tolerance
        - Residual_STD: Standard deviation of residuals (error consistency)
        - MBE: Mean Bias Error (systematic bias)
        
    Notes
    -----
    Metric Interpretations:
    - RMSE: Lower is better, penalizes large errors more
    - MAE: Lower is better, more robust to outliers
    - R2: Higher is better (max 1.0), indicates prediction quality
    - Accuracy: Higher is better, practical measure of usability
    - Residual_STD: Lower is better, indicates prediction consistency
    - MBE: Closer to 0 is better, indicates systematic bias
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
