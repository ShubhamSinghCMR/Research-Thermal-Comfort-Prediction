import pandas as pd
import numpy as np
import joblib
import os
import logging
import lightgbm as lgb
from utils.config import LIGHTGBM_PARAMS

def prepare_meta_features(base_preds, test_features):
    """
    Prepares meta features by combining base model predictions and test features.
    """
    meta_features = pd.DataFrame()
    
    # Base model predictions
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
    
    # Add some context features if available
    if 'HumidityPerception' in test_features.columns:
        meta_features['HumidityPerception'] = test_features['HumidityPerception'].values
    if 'AirMovement' in test_features.columns:
        meta_features['AirMovement'] = test_features['AirMovement'].values
    if 'BMI' in test_features.columns:
        meta_features['BMI'] = test_features['BMI'].values
    if 'CLO_score' in test_features.columns:
        meta_features['CLO_score'] = test_features['CLO_score'].values

    return meta_features


def train_meta_model(meta_X, y_tsv, y_temp, model_dir="models/saved/"):
    """
    Trains LightGBM meta learners for TSV and Temperature.
    Uses all provided data for training and prediction (no additional split).
    Returns predictions for the entire meta_X dataset.
    """
    os.makedirs(model_dir, exist_ok=True)
    logging.info('Meta model training started.')

    # Use all data for training and prediction (no additional split)
    X_train = meta_X
    y_tsv_train = y_tsv
    y_temp_train = y_temp
    
    # For evaluation, we'll use the same data (since this is already the test set from base models)
    X_test = meta_X
    y_tsv_test = y_tsv
    y_temp_test = y_temp

    # Test indices are all indices (since we're using all data)
    test_indices = range(len(meta_X))

    logging.info('Training LightGBM meta model for TSV...')
    # Use configurable LightGBM parameters
    tsv_lgb = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
    tsv_lgb.fit(X_train, y_tsv_train)
    tsv_pred = tsv_lgb.predict(X_test)
    logging.info('TSV meta model trained.')
    joblib.dump(tsv_lgb, os.path.join(model_dir, "meta_tsv.pkl"))

    logging.info('Training LightGBM meta model for Temperature...')
    # Use configurable LightGBM parameters
    temp_lgb = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
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
