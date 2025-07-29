"""
Adaptive Configuration Module for Thermal Comfort Prediction
==============================================================

This module centralizes all configuration settings and hyperparameters for the 
adaptive thermal comfort prediction system. It allows environment-specific tuning 
and provides a single source of truth for parameters.

Configuration Categories:
-------------------------
1. Global Settings
   - Logging and warning controls
   - Random seeds for reproducibility
   - Data splitting ratios

2. Base Model Parameters
   - CatBoost
   - ExtraTrees
   - ElasticNet
   - XGBoost
   - LightGBM

3. Environment-Specific Parameters
   - Classroom
   - Hostel
   - Winter Environments
   - Workshop/Laboratory

Usage:
------
    from adaptive_tsv_pipeline.utils.config import SHOW_WARNINGS, get_environment_params

    params = get_environment_params("Class room")
"""

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

SHOW_WARNINGS = False    # Master switch for showing warnings/logs
SEED = 42               # Global seed for reproducibility

# =============================================================================
# DATA SPLITTING CONFIGURATION
# =============================================================================

TRAIN_SIZE_PERCENT = 80   # Training set size percentage
TEST_SIZE_PERCENT = 20    # Testing set size percentage

# =============================================================================
# BASE MODEL PARAMETERS
# =============================================================================

# CatBoost Parameters
BASE_CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 5,
    "random_seed": SEED,
    "early_stopping_rounds": 50,
    "verbose": False,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
}

# ExtraTrees Parameters
BASE_EXTRATREES_PARAMS = {
    "n_estimators": 500,
    "max_depth": 18,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": SEED,
    "n_jobs": -1,
}

# ElasticNet Parameters
BASE_ELASTICNET_PARAMS = {
    "alpha": 0.3,
    "l1_ratio": 0.5,
    "max_iter": 5000,
    "random_state": SEED,
}

# XGBoost Parameters
BASE_XGBOOST_PARAMS = {
    "n_estimators": 700,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1,
    "random_state": SEED,
    "early_stopping_rounds": 50,
    "verbosity": 0,
}

# LightGBM Parameters
BASE_LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "random_state": SEED,
    "n_jobs": -1,
    "boosting_type": "gbdt",
    "verbose": -1,
    "min_gain_to_split": 0,
}

# =============================================================================
# ENVIRONMENT-SPECIFIC PARAMETERS
# =============================================================================

def get_environment_params(environment_name: str):
    """
    Get tuned parameters for a specific environment.
    
    Parameters
    ----------
    environment_name : str
        Environment name (e.g., 'Class room', 'Hostel', 'Workshop or laboratory')
    
    Returns
    -------
    dict
        Dictionary of tuned parameters for all models
    """
    catboost_params = BASE_CATBOOST_PARAMS.copy()
    extratrees_params = BASE_EXTRATREES_PARAMS.copy()
    elasticnet_params = BASE_ELASTICNET_PARAMS.copy()
    xgboost_params = BASE_XGBOOST_PARAMS.copy()
    lightgbm_params = BASE_LIGHTGBM_PARAMS.copy()

    # Adjustments based on environment
    if environment_name == "Class room":
        catboost_params.update({"depth": 6, "learning_rate": 0.04})
        lightgbm_params.update({"num_leaves": 25, "max_depth": 6})
    elif environment_name == "Hostel":
        catboost_params.update({"depth": 7})
        lightgbm_params.update({"num_leaves": 31, "max_depth": -1})
    elif "Winter" in environment_name:
        catboost_params.update({"iterations": 700, "depth": 6, "learning_rate": 0.04})
        lightgbm_params.update({"n_estimators": 250, "num_leaves": 25})
    elif environment_name == "Workshop or laboratory":
        catboost_params.update({"depth": 6})
        lightgbm_params.update({"num_leaves": 28})

    return {
        "catboost": catboost_params,
        "extratrees": extratrees_params,
        "elasticnet": elasticnet_params,
        "xgboost": xgboost_params,
        "lightgbm": lightgbm_params,
    }
