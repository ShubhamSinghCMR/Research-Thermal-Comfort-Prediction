
# ===================== Global =====================
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
#   'hybrid'          -> UNIV_WEIGHT * univariate + PERM_WEIGHT * permutation
SELECTION_MODE = 'hybrid'
UNIV_WEIGHT = 0.4
PERM_WEIGHT = 0.6

# Selection scope:
#   'per_model' -> each base model gets its own selected set (default)
#   'global'    -> one common set for all base models per environment
SELECTION_SCOPE = 'per_model'

# Meta / stack
USE_HUBER_META = True
USE_STACKING = True      # if False -> pick the best single base model (by R2)

# Calibration
CALIBRATION_ON = True

# ===================== Ensemble / Stacking options =====================
# Keep base models diverse; names must match keys returned by get_environment_params()
ENSEMBLE_BASE_MODELS = ["catboost", "xgboost", "lightgbm", "elasticnet", "svr_rbf"]

# Meta-learner for stacking: "ridge" | "nnls" | "avg" | "lgbm"
STACKING_METHOD = "nnls"

# Optionally auto-select best stacker by CV R2 among candidates
AUTO_SELECT_STACKER = False
STACKER_CANDIDATES = ["ridge", "nnls", "avg", "lgbm"]

# Ridge search grid for meta
META_RIDGE_ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0]

# Repeated CV & stratified target bins (for meta training stability)
CV_REPEATS = 3
STRATIFY_BINS = 7

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

def _get_baseline_defaults(seed=SEED):
    return {
        "svr_rbf": {"kernel": "rbf", "C": 10.0, "gamma": "scale", "epsilon": 0.1},
        "random_forest": {"n_estimators": 300, "max_depth": None, "n_jobs": -1, "random_state": seed},
        "gradient_boosting": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0, "random_state": seed},
        "decision_tree": {"max_depth": None, "random_state": seed},
        "linear_regression": {},  # sklearn defaults
        "ridge": {"alpha": 1.0, "random_state": seed},
        "lasso": {"alpha": 0.001, "max_iter": 10000, "random_state": seed},
        "knn_distance": {"n_neighbors": 15, "weights": "distance"},
        "mlp": {"hidden_layer_sizes": (100,), "activation": "relu", "alpha": 0.0001, "learning_rate_init": 0.001,
                "max_iter": 1000, "early_stopping": True, "n_iter_no_change": 20, "random_state": seed},
        "xgboost_single": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": seed, "n_jobs": -1},
        "lightgbm_single": {"n_estimators": 500, "num_leaves": 31, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": seed},
        "catboost_single": {"iterations": 600, "depth": 6, "learning_rate": 0.05, "random_state": seed, "verbose": False},
        "adaboost": {"n_estimators": 400, "learning_rate": 0.05, "random_state": seed},
    }

# Environments: tune a bit per environment
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

    params = {
        "catboost": catboost,
        "extratrees": extratrees,
        "elasticnet": elasticnet,
        "xgboost": xgboost,
        "lightgbm": lightgbm,
    }

    # Single-model counterparts inherit unless overridden
    params["xgboost_single"]   = params["xgboost"].copy()
    params["lightgbm_single"]  = params["lightgbm"].copy()
    params["catboost_single"]  = params["catboost"].copy()

    # Fill the rest of baselines with defaults if missing
    for k, v in _get_baseline_defaults(SEED).items():
        params.setdefault(k, v.copy())

    return params

# Models to include in baseline comparison
BASELINE_MODELS = [
    "svr_rbf",
    "random_forest",
    "gradient_boosting",
    "decision_tree",
    "linear_regression",
    "ridge",
    "lasso",
    "knn_distance",
    "mlp",
    "xgboost_single",
    "lightgbm_single",
    # "catboost_single",
    "adaboost",
]
