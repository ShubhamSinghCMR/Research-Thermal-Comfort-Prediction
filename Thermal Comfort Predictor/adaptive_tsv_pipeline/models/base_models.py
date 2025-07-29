"""
Adaptive Base Models Module for Thermal Comfort Prediction
===========================================================

This module implements adaptive base models for the thermal comfort prediction system.
It is designed to handle dynamic feature sets from the adaptive pipeline and 
ensures robust model training using K-Fold cross-validation.

Key Features:
-------------
- Multiple base models: CatBoost, ExtraTrees, ElasticNet, XGBoost
- Handles dynamic feature spaces (adaptive pipeline)
- K-Fold cross-validation for reliable performance estimation
- Out-of-fold predictions for meta-model training
- Overfitting detection through train vs validation accuracy tracking
- Comprehensive evaluation metrics

Dependencies:
-------------
- scikit-learn: Model training and evaluation
- catboost: Gradient boosting implementation
- xgboost: Gradient boosting implementation
- numpy: Numerical operations
- pandas: Data manipulation

Main Functions:
---------------
- train_base_models: Main entry point for training all base models
- get_base_models: Get configured base model instances
- evaluate_predictions: Calculate multiple performance metrics
"""

import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import utilities from adaptive pipeline
from ..utils.metrics import calculate_accuracy
from ..utils.config import SHOW_WARNINGS


# ==========================
# Warning Config
# ==========================
if not SHOW_WARNINGS:
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)


def print_status(message, status=""):
    """
    Print formatted status updates with emoji indicators.
    
    Parameters
    ----------
    message : str
        Status message to display
    status : str, optional
        Status type ('started', 'completed', or '')
    """
    if status == "started":
        print(f"\n[🔄 Started] {message}")
    elif status == "completed":
        print(f"[✅ Completed] {message}")
    else:
        print(f"[ℹ️] {message}")


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using multiple performance metrics.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns
    -------
    dict
        Dictionary containing:
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - R2: R-squared score
        - Accuracy: Custom accuracy within tolerance
        - Residual_STD: Standard deviation of residuals
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    accuracy = calculate_accuracy(y_true, y_pred)
    residual_std = np.std(y_true - y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Accuracy": accuracy,
        "Residual_STD": residual_std,
    }


def get_base_models():
    """
    Get dictionary of configured base models with tuned hyperparameters.
    
    Returns
    -------
    dict
        Dictionary of model instances:
        - catboost: CatBoostRegressor
        - extratrees: ExtraTreesRegressor
        - elasticnet: ElasticNet
        - xgboost: XGBRegressor
    """
    return {
        "catboost": CatBoostRegressor(
            depth=7,
            iterations=1000,
            learning_rate=0.05,
            l2_leaf_reg=5,
            random_seed=42,
            early_stopping_rounds=50,
            verbose=False,
        ),
        "extratrees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "elasticnet": ElasticNet(
            alpha=0.3,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42,
        ),
        "xgboost": XGBRegressor(
            n_estimators=700,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
            random_state=42,
            early_stopping_rounds=50,
            verbosity=0,
        ),
    }


def train_base_models(X, y, n_splits=5):
    """
    Train base models using K-Fold cross-validation and return out-of-fold predictions.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Input features (dynamic/adaptive)
    y : pandas.Series
        Target values
    n_splits : int, default=5
        Number of K-Fold splits
        
    Returns
    -------
    tuple
        oof_preds : pandas.DataFrame
            Out-of-fold predictions from each base model
        model_results : dict
            Performance metrics for each model (averaged across folds)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    base_models = get_base_models()

    # Storage for predictions and metrics
    oof_preds = pd.DataFrame(np.zeros((len(X), len(base_models))), columns=base_models.keys())
    model_results = {}

    for idx, (name, model) in enumerate(base_models.items(), 1):
        if SHOW_WARNINGS:
            print_status(f"Training base model [{idx}/{len(base_models)}] - {name}", "started")

        fold_metrics = []
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            # Fit models with early stopping for boosting models
            if name == "catboost":
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
            elif name == "xgboost":
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_valid_pred = model.predict(X_valid)

            # Metrics
            train_score = evaluate_predictions(y_train, y_train_pred)
            valid_score = evaluate_predictions(y_valid, y_valid_pred)

            # Gap analysis
            valid_score["Train_Accuracy"] = train_score["Accuracy"]

            # Out-of-fold predictions
            oof_preds.loc[valid_idx, name] = y_valid_pred

            fold_metrics.append(valid_score)

        # Average metrics
        avg_metrics = {m: np.mean([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
        model_results[name] = avg_metrics

        if SHOW_WARNINGS:
            print_status(
                f"Completed {name}: "
                f"RMSE={avg_metrics['RMSE']:.3f}, MAE={avg_metrics['MAE']:.3f}, "
                f"R2={avg_metrics['R2']:.3f}, Accuracy={avg_metrics['Accuracy']:.2%}",
                "completed",
            )

    return oof_preds, model_results
