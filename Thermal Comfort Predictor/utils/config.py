# =============================================================================
# CONFIGURATION FILE FOR THERMAL COMFORT PREDICTOR
# =============================================================================

SHOW_WARNINGS = False    # Master switch to show/hide all warnings and logs
SEED = 42

# =============================================================================
# DATA SPLITTING CONFIGURATION
# =============================================================================
TRAIN_SIZE_PERCENT = 80
TEST_SIZE_PERCENT = 20

# =============================================================================
# BASE MODEL PARAMETERS (final ensemble)
# =============================================================================

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

BASE_EXTRATREES_PARAMS = {
    "n_estimators": 500,
    "max_depth": 18,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": SEED,
    "n_jobs": -1,
}

BASE_ELASTICNET_PARAMS = {
    "alpha": 0.3,
    "l1_ratio": 0.5,
    "max_iter": 5000,
    "random_state": SEED,
}

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
    "verbose": -1,  # Suppress all warnings
    "min_gain_to_split": 0,  # Allow splits with zero gain
}

# =============================================================================
# ENVIRONMENT-SPECIFIC PARAMETER FUNCTION
# =============================================================================
def get_environment_params(environment_name):
    """Return tuned parameters for each environment"""
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
    total = TRAIN_SIZE_PERCENT + TEST_SIZE_PERCENT
    if total != 100:
        raise ValueError(
            f"Train and test percentages must add up to 100. Current: {TRAIN_SIZE_PERCENT} + {TEST_SIZE_PERCENT} = {total}"
        )

validate_split_config()
