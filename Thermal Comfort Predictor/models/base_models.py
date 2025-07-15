import pandas as pd
import numpy as np
import os
import joblib
import torch
import logging
import shutil

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from utils.config import (SEED, TEST_SIZE_PERCENT, CATBOOST_PARAMS, 
                         RANDOM_FOREST_PARAMS, BAYESIAN_RIDGE_PARAMS, QUANTILE_RF_PARAMS)
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
except ImportError:
    print("Warning: pytorch-tabnet not available. TabNet will be skipped.")
    TabNetRegressor = None

def train_base_models(X, y_tsv, y_temp, model_dir="models/saved/"):
    os.makedirs(model_dir, exist_ok=True)
    logging.info('Base model training started.')

    # Convert to numpy arrays to avoid feature names warnings for model training
    X_array = X.values if hasattr(X, 'values') else np.array(X)
    y_tsv_array = y_tsv.values if hasattr(y_tsv, 'values') else np.array(y_tsv)
    y_temp_array = y_temp.values if hasattr(y_temp, 'values') else np.array(y_temp)

    # Use configurable test size from config
    test_size = TEST_SIZE_PERCENT / 100.0  # Convert percentage to decimal
    X_train, X_test, y_tsv_train, y_tsv_test, y_temp_train, y_temp_test = train_test_split(
        X, y_tsv, y_temp, test_size=test_size, random_state=SEED)  # Use DataFrame for X to preserve indices

    # For model training, use numpy arrays
    X_train_array = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
    X_test_array = X_test.values if hasattr(X_test, 'values') else np.array(X_test)
    y_tsv_train_array = y_tsv_train.values if hasattr(y_tsv_train, 'values') else np.array(y_tsv_train)
    y_temp_train_array = y_temp_train.values if hasattr(y_temp_train, 'values') else np.array(y_temp_train)

    predictions = {
        "TSV": {},
        "Temp": {}
    }

    logging.info('Training CatBoost models...')
    # Use configurable CatBoost parameters
    cat_tsv = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_temp = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_tsv.fit(X_train_array, y_tsv_train_array)
    cat_temp.fit(X_train_array, y_temp_train_array)
    logging.info('CatBoost models trained.')

    # Clean up catboost_info folder if it was created
    if os.path.exists('catboost_info'):
        try:
            shutil.rmtree('catboost_info')
            logging.info('Cleaned up catboost_info folder.')
        except Exception as e:
            logging.warning(f'Could not remove catboost_info folder: {e}')

    predictions["TSV"]["catboost"] = cat_tsv.predict(X_test_array)
    predictions["Temp"]["catboost"] = cat_temp.predict(X_test_array)

    joblib.dump(cat_tsv, f"{model_dir}/catboost_tsv.pkl")
    joblib.dump(cat_temp, f"{model_dir}/catboost_temp.pkl")

    logging.info('Training Random Forest models...')
    # Use configurable Random Forest parameters
    rf_tsv = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    rf_temp = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    rf_tsv.fit(X_train_array, y_tsv_train_array)
    rf_temp.fit(X_train_array, y_temp_train_array)
    logging.info('Random Forest models trained.')

    predictions["TSV"]["rf"] = rf_tsv.predict(X_test_array)
    predictions["Temp"]["rf"] = rf_temp.predict(X_test_array)

    joblib.dump(rf_tsv, f"{model_dir}/rf_tsv.pkl")
    joblib.dump(rf_temp, f"{model_dir}/rf_temp.pkl")

    logging.info('Training Bayesian Ridge models...')
    # Use configurable Bayesian Ridge parameters
    br_tsv = BayesianRidge(**BAYESIAN_RIDGE_PARAMS)
    br_temp = BayesianRidge(**BAYESIAN_RIDGE_PARAMS)
    br_tsv.fit(X_train_array, y_tsv_train_array)
    br_temp.fit(X_train_array, y_temp_train_array)
    logging.info('Bayesian Ridge models trained.')

    predictions["TSV"]["bayes"] = br_tsv.predict(X_test_array)
    predictions["Temp"]["bayes"] = br_temp.predict(X_test_array)

    joblib.dump(br_tsv, f"{model_dir}/bayes_tsv.pkl")
    joblib.dump(br_temp, f"{model_dir}/bayes_temp.pkl")

    logging.info('Training fallback TabNet (Random Forest) model...')
    rf_fallback = RandomForestRegressor(random_state=SEED)
    rf_fallback.fit(X_train_array, y_tsv_train_array)
    predictions["TSV"]["tabnet"] = rf_fallback.predict(X_test_array)
    predictions["Temp"]["tabnet"] = rf_fallback.predict(X_test_array)
    logging.info('Fallback TabNet model trained.')

    logging.info('Training Quantile Random Forest for uncertainty...')
    # Use configurable Quantile RF parameters  
    qrf = RandomForestRegressor(**QUANTILE_RF_PARAMS)
    qrf.fit(X_train_array, y_tsv_train_array)

    all_preds = np.array([tree.predict(X_test_array) for tree in qrf.estimators_])
    lower = np.percentile(all_preds, 5, axis=0)
    upper = np.percentile(all_preds, 95, axis=0)

    predictions["TSV"]["qrf_lower"] = lower
    predictions["TSV"]["qrf_upper"] = upper
    logging.info('Quantile Random Forest trained.')

    joblib.dump(qrf, f"{model_dir}/qrf_tsv.pkl")

    logging.info('Base model training completed.')
    return predictions, X_test, y_tsv_test, y_temp_test
