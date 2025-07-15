import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
import logging

def prepare_meta_features(base_preds, X_base):
    """
    Creates meta-feature DataFrame from base model predictions and perception/context.
    """
    meta_features = pd.DataFrame()

    meta_features['tsv_catboost'] = base_preds['TSV']['catboost']
    meta_features['tsv_rf'] = base_preds['TSV']['rf']
    meta_features['tsv_tabnet'] = base_preds['TSV']['tabnet']
    meta_features['tsv_bayes'] = base_preds['TSV']['bayes']
    meta_features['tsv_spread'] = (
        meta_features[['tsv_catboost', 'tsv_rf', 'tsv_tabnet', 'tsv_bayes']].max(axis=1) -
        meta_features[['tsv_catboost', 'tsv_rf', 'tsv_tabnet', 'tsv_bayes']].min(axis=1)
    )
    meta_features['tsv_qrf_lower'] = base_preds['TSV']['qrf_lower']
    meta_features['tsv_qrf_upper'] = base_preds['TSV']['qrf_upper']

    meta_features['temp_catboost'] = base_preds['Temp']['catboost']
    meta_features['temp_rf'] = base_preds['Temp']['rf']
    meta_features['temp_tabnet'] = base_preds['Temp']['tabnet']
    meta_features['temp_bayes'] = base_preds['Temp']['bayes']
    meta_features['temp_spread'] = (
        meta_features[['temp_catboost', 'temp_rf', 'temp_tabnet', 'temp_bayes']].max(axis=1) -
        meta_features[['temp_catboost', 'temp_rf', 'temp_tabnet', 'temp_bayes']].min(axis=1)
    )

    if isinstance(X_base, pd.DataFrame):
        for col in ['HumidityPerception', 'AirMovement', 'LightPerception', 'ThermalControlIndex']:
            if col in X_base.columns:
                meta_features[col] = X_base[col].reset_index(drop=True)

    return meta_features


def train_meta_model(meta_X, y_tsv, y_temp, model_dir="models/saved/"):
    """
    Trains LightGBM meta learners for TSV and Temperature.
    Returns predictions and trained models, and test indices.
    """
    os.makedirs(model_dir, exist_ok=True)
    logging.info('Meta model training started.')

    # Store original indices for later use
    original_indices = meta_X.index if hasattr(meta_X, 'index') else range(len(meta_X))

    X_train, X_test, y_tsv_train, y_tsv_test, y_temp_train, y_temp_test = train_test_split(
        meta_X, y_tsv, y_temp, test_size=0.2, random_state=42)

    # Calculate test indices based on the split
    test_size = len(X_test)
    total_size = len(meta_X)
    train_size = total_size - test_size
    test_indices = original_indices[train_size:]

    logging.info('Training LightGBM meta model for TSV...')
    # Configure LightGBM for small datasets to reduce warnings
    tsv_lgb = lgb.LGBMRegressor(
        random_state=42,
        verbose=-1,  # Suppress all output
        min_child_samples=1,  # Allow single samples in leaves
        min_split_gain=0,  # Allow splits with no gain
        num_leaves=31,  # Limit tree complexity
        max_depth=6,  # Limit tree depth
        n_estimators=100
    )
    tsv_lgb.fit(X_train, y_tsv_train)
    tsv_pred = tsv_lgb.predict(X_test)
    logging.info('TSV meta model trained.')
    joblib.dump(tsv_lgb, os.path.join(model_dir, "meta_tsv.pkl"))

    logging.info('Training LightGBM meta model for Temperature...')
    temp_lgb = lgb.LGBMRegressor(
        random_state=42,
        verbose=-1,  # Suppress all output
        min_child_samples=1,  # Allow single samples in leaves
        min_split_gain=0,  # Allow splits with no gain
        num_leaves=31,  # Limit tree complexity
        max_depth=6,  # Limit tree depth
        n_estimators=100
    )
    temp_lgb.fit(X_train, y_temp_train)
    temp_pred = temp_lgb.predict(X_test)
    logging.info('Temperature meta model trained.')
    joblib.dump(temp_lgb, os.path.join(model_dir, "meta_temp.pkl"))

    importance_df = pd.DataFrame({
        'Feature': meta_X.columns,
        'TSV_Importance': tsv_lgb.feature_importances_,
        'Temp_Importance': temp_lgb.feature_importances_
    })
    importance_df.to_csv(os.path.join(model_dir, "meta_feature_importance.csv"), index=False)
    logging.info('Meta model training completed.')

    return {
        "TSV_meta": tsv_pred,
        "Temp_meta": temp_pred,
        "y_tsv_true": y_tsv_test,
        "y_temp_true": y_temp_test,
        "tsv_model": tsv_lgb,
        "temp_model": temp_lgb,
        "test_indices": test_indices
    }
