"""
Configuration Module for Thermal Comfort Prediction
=============================================

This module contains all configuration settings and hyperparameters for the
thermal comfort prediction system. It provides centralized parameter management
and environment-specific tuning.

Configuration Categories:
----------------------
1. Global Settings
   - Warning/logging controls
   - Random seed
   - Data split ratios

2. Base Model Parameters
   - CatBoost
   - ExtraTrees
   - ElasticNet
   - XGBoost
   - LightGBM

3. Environment-Specific Parameters
   - Class room
   - Hostel
   - Winter environments
   - Workshop/laboratory

Usage:
-----
Import specific parameters:
    from utils.config import SHOW_WARNINGS, SEED

Get environment parameters:
    from utils.config import get_environment_params
    params = get_environment_params("Class room")
"""

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

SHOW_WARNINGS = False    # Master switch to show/hide all warnings and logs
SEED = 42               # Global random seed for reproducibility

# =============================================================================
# DATA SPLITTING CONFIGURATION
# =============================================================================

TRAIN_SIZE_PERCENT = 80  # Percentage of data for training
TEST_SIZE_PERCENT = 20   # Percentage of data for testing

# =============================================================================
# BASE MODEL PARAMETERS
# =============================================================================

# CatBoost Configuration
# ---------------------
# - Uses Bernoulli bootstrapping for better handling of small datasets
# - Early stopping to prevent overfitting
# - L2 regularization for stability
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

# ExtraTrees Configuration
# ----------------------
# - Large number of trees for stable predictions
# - Controlled tree depth to prevent overfitting
# - Minimum samples requirements for splits and leaves
BASE_EXTRATREES_PARAMS = {
    "n_estimators": 500,
    "max_depth": 18,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": SEED,
    "n_jobs": -1,
}

# ElasticNet Configuration
# ----------------------
# - Balanced L1/L2 regularization
# - Increased max iterations for convergence
BASE_ELASTICNET_PARAMS = {
    "alpha": 0.3,
    "l1_ratio": 0.5,
    "max_iter": 5000,
    "random_state": SEED,
}

# XGBoost Configuration
# -------------------
# - Feature and sample subsampling for robustness
# - L2 regularization
# - Early stopping enabled
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

# LightGBM Configuration
# --------------------
# - Leaf-wise growth with controlled depth
# - Feature and sample subsampling
# - L1/L2 regularization
# - Early stopping enabled
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
# ENVIRONMENT-SPECIFIC PARAMETER FUNCTION
# =============================================================================

def get_environment_params(environment_name):
    """
    Get tuned model parameters for specific environments.
    
    Parameters
    ----------
    environment_name : str
        Name of the environment to get parameters for
        
    Returns
    -------
    dict
        Dictionary containing tuned parameters for all models
        
    Notes
    -----
    Environment-Specific Tuning:
    - Class room: Reduced tree depth, slower learning
    - Hostel: Standard parameters with adjusted tree depth
    - Winter environments: Fewer iterations, reduced depth
    - Workshop/laboratory: Adjusted tree structure
    
    The function modifies base parameters based on empirical
    performance in each environment.
    """
    catboost_params = BASE_CATBOOST_PARAMS.copy()
    extratrees_params = BASE_EXTRATREES_PARAMS.copy()
    elasticnet_params = BASE_ELASTICNET_PARAMS.copy()
    xgboost_params = BASE_XGBOOST_PARAMS.copy()
    lightgbm_params = BASE_LIGHTGBM_PARAMS.copy()

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

# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate_split_config():
    """
    Validate that train/test split percentages sum to 100.
    
    Raises
    ------
    ValueError
        If train and test percentages don't sum to 100
    """
    total = TRAIN_SIZE_PERCENT + TEST_SIZE_PERCENT
    if total != 100:
        raise ValueError(
            f"Train and test percentages must add up to 100. Current: {TRAIN_SIZE_PERCENT} + {TEST_SIZE_PERCENT} = {total}"
        )

validate_split_config()
