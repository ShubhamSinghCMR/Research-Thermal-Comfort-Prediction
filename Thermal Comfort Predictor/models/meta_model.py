"""
Meta-Model Module for Thermal Comfort Prediction
============================================

This module implements the meta-model (stacking) layer of the thermal comfort prediction system.
It uses LightGBM as the meta-learner to combine predictions from base models.

Key Features:
------------
- LightGBM meta-learner with environment-specific parameters
- K-Fold cross-validation for robust performance estimation
- Feature importance tracking for model interpretability
- Early stopping to prevent overfitting
- Comprehensive evaluation metrics

Dependencies:
------------
- lightgbm: Meta-model implementation
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Model evaluation and cross-validation

Main Functions:
-------------
- train_meta_model_kfold: Train meta-model using K-Fold cross-validation
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from models.base_models import evaluate_predictions
from utils.config import SHOW_WARNINGS

def train_meta_model_kfold(oof_preds, y, env_params, n_splits=5):
    """
    Train LightGBM meta-model using K-Fold cross-validation.
    
    Parameters
    ----------
    oof_preds : pandas.DataFrame
        Out-of-fold predictions from base models
    y : pandas.Series
        Target values
    env_params : dict
        Environment-specific parameters including LightGBM settings
    n_splits : int, default=5
        Number of K-Fold splits
        
    Returns
    -------
    dict
        Dictionary containing:
        - oof_predictions: Out-of-fold meta-model predictions
        - cv_metrics: Cross-validation performance metrics
        - feature_importance: Feature importance DataFrame
        - model: Trained meta-model instance
        
    Notes
    -----
    Training Process:
    1. Configure LightGBM parameters for environment
    2. Initialize K-Fold cross-validation
    3. For each fold:
        - Train on K-1 folds
        - Predict on held-out fold
        - Store out-of-fold predictions
        - Calculate performance metrics
        - Track feature importance
    4. Average metrics and feature importance across folds
    
    Early Stopping:
    - Uses 50 rounds patience
    - Monitors validation set performance
    - Prevents overfitting
    """
    lgb_params = env_params["lightgbm"].copy()
    
    # Ensure verbose parameter is set correctly
    if not SHOW_WARNINGS:
        lgb_params["verbose"] = -1
        lgb_params["min_gain_to_split"] = 0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta_model = lgb.LGBMRegressor(**lgb_params)

    oof_meta_preds = np.zeros(len(oof_preds))
    fold_metrics = []
    feature_importances = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(oof_preds, y), 1):
        X_train, X_valid = oof_preds.iloc[train_idx], oof_preds.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        callbacks = [lgb.early_stopping(50)]
        if SHOW_WARNINGS:
            callbacks.append(lgb.log_evaluation())

        meta_model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=callbacks
        )

        y_pred = meta_model.predict(X_valid)
        oof_meta_preds[valid_idx] = y_pred
        metrics = evaluate_predictions(y_valid, y_pred)
        fold_metrics.append(metrics)

        feature_importances.append(meta_model.feature_importances_)

        if SHOW_WARNINGS:
            print(f"[Meta-model Fold {fold}] RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}, R2={metrics['R2']:.3f}")

    avg_metrics = {m: np.mean([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
    feature_importance_df = pd.DataFrame({
        "feature": oof_preds.columns,
        "importance": np.mean(feature_importances, axis=0),
    }).sort_values("importance", ascending=False)

    if SHOW_WARNINGS:
        print(f"\n[Meta-model] Final CV Metrics: RMSE={avg_metrics['RMSE']:.3f}, MAE={avg_metrics['MAE']:.3f}, R2={avg_metrics['R2']:.3f}")

    return {
        "oof_predictions": oof_meta_preds,
        "cv_metrics": avg_metrics,
        "feature_importance": feature_importance_df,
        "model": meta_model,
    }
