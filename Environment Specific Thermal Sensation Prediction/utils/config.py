SEED = 42
SHOW_WARNINGS = False

# ===================== CV / Selection =====================
FOLDS = 5
TOP_M_PER_FOLD = 7
STABILITY_TAU = 0.6
K_VALUES = [3,4,5,6,7]
KMIN = 3
KMAX = 7
PERM_REPEATS = 3

# Selection modes:
#   'none'            -> keep all original features
#   'univariate_only' -> rank by univariate (r^2 / eta^2) only
#   'permutation_only'-> rank by permutation (ΔR2 rank) only
#   'hybrid'          -> 0.4 * univariate_rank + 0.6 * permutation_rank
SELECTION_MODE = 'hybrid'

# Selection scope:
#   'per_model' -> each base model gets its own selected set (default)
#   'global'    -> one common set for all base models per environment
SELECTION_SCOPE = 'per_model'

# Meta / stack
USE_HUBER_META = True
USE_STACKING = True      # if False -> pick the best single base model (by R2)

# Calibration
CALIBRATION_ON = True

# ===================== Base model hyperparameters =====================
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
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 1.2,
    "random_state": SEED,
    "n_jobs": -1,
    "boosting_type": "gbdt",
    "verbose": -1,
    "min_gain_to_split": 0,
}

def get_environment_params(environment_name):
    catboost = BASE_CATBOOST_PARAMS.copy()
    extratrees = BASE_EXTRATREES_PARAMS.copy()
    elasticnet = BASE_ELASTICNET_PARAMS.copy()
    xgboost = BASE_XGBOOST_PARAMS.copy()
    lightgbm = BASE_LIGHTGBM_PARAMS.copy()

    if environment_name == "Classroom":
        catboost.update({"depth": 6, "learning_rate": 0.04})
        lightgbm.update({"num_leaves": 25, "max_depth": 6})
    elif environment_name == "Hostel":
        catboost.update({"depth": 7})
        lightgbm.update({"num_leaves": 31, "max_depth": -1})
    elif "Winter" in environment_name:
        catboost.update({"iterations": 700, "depth": 6, "learning_rate": 0.04})
        lightgbm.update({"n_estimators": 500, "num_leaves": 25})
    elif environment_name == "Workshop or laboratory":
        catboost.update({"depth": 6})
        lightgbm.update({"num_leaves": 28})

    return {"catboost": catboost, "extratrees": extratrees, "elasticnet": elasticnet,
            "xgboost": xgboost, "lightgbm": lightgbm}
