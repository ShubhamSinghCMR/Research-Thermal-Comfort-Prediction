# =============================================================================
# CONFIGURATION FILE FOR THERMAL COMFORT PREDICTOR
# =============================================================================
# This file contains all the settings and parameters for the machine learning models.
# You can adjust these values to improve performance or change how the system works.

# =============================================================================
# REPRODUCIBILITY SETTING
# =============================================================================

# SEED: A magic number that makes sure you get the same results every time you run the program
# Think of it like setting a specific starting point for randomness - this way, if you run 
# the program multiple times, you'll get identical results. This is crucial for:
# - Comparing different model configurations fairly
# - Debugging and troubleshooting (same results = easier to find problems)
# - Scientific reproducibility (others can replicate your exact results)
# - Consistent testing and validation
# The number 42 is commonly used in programming (from "The Hitchhiker's Guide to the Galaxy")
SEED = 42

# =============================================================================
# DATA SPLITTING CONFIGURATION
# =============================================================================

# How to split your data between training and testing
# Training data: Used to teach the models how to make predictions
# Testing data: Used to see how well the models perform on new, unseen data

TRAIN_SIZE_PERCENT = 80  # Use 80% of data for training the models - RECOMMENDED
TEST_SIZE_PERCENT = 20   # Use 20% of data for testing the models - RECOMMENDED

# NOTE: These two numbers must add up to 100 (we use ALL the data, just split it up)
# Common ways to split data:
# - 80/20: Standard split, good balance between learning and testing - RECOMMENDED for most cases
# - 70/30: More testing data, better for checking if models really work - Good for smaller datasets
# - 90/10: More training data, helps complex models learn better - Use only with large datasets (>1000 samples)
# - 50/50: Half and half, maximum testing but less learning - Not recommended for production models

# =============================================================================
# MACHINE LEARNING MODEL SETTINGS
# =============================================================================
# Each model has different "knobs and dials" you can adjust to make it work better.
# Think of these like settings on a camera - you can adjust them for different situations.

# -----------------------------------------------------------------------------
# CatBoost Model Settings (Gradient Boosting - good for structured data)
# -----------------------------------------------------------------------------
CATBOOST_PARAMS = {
    'iterations': 1000,           # How many learning rounds (more = learns more, but slower) - RECOMMENDED: 500-2000
    'learning_rate': 0.1,         # How fast it learns (smaller = more careful learning) - RECOMMENDED: 0.01-0.3
    'depth': 6,                   # How deep each decision tree can go - RECOMMENDED: 4-10
    'l2_leaf_reg': 3,            # Prevents overfitting (higher = more conservative) - RECOMMENDED: 1-10
    'bagging_temperature': 1,     # Controls randomness in training - RECOMMENDED: 0.5-2
    'random_seed': SEED,          # Uses our reproducibility number
    'allow_writing_files': False, # Don't create extra files during training
    'verbose': False              # Don't print lots of training messages
}

# -----------------------------------------------------------------------------
# Random Forest Model Settings (Many decision trees working together)
# -----------------------------------------------------------------------------
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,          # Number of decision trees to create - RECOMMENDED: 50-300
    'max_depth': None,            # How deep each tree can grow (None = no limit) - RECOMMENDED: 10-20 or None
    'min_samples_split': 2,       # Minimum data points needed to split a branch - RECOMMENDED: 2-10
    'min_samples_leaf': 1,        # Minimum data points that must be in each leaf - RECOMMENDED: 1-5
    'max_features': 'sqrt',       # How many features to consider for each split - RECOMMENDED: 'sqrt' or 'log2'
    'random_state': SEED          # Uses our reproducibility number
}

# -----------------------------------------------------------------------------
# Bayesian Ridge Model Settings (Statistical regression with uncertainty)
# -----------------------------------------------------------------------------
BAYESIAN_RIDGE_PARAMS = {
    'alpha_1': 1e-6,             # Prior belief about noise level (technical parameter) - RECOMMENDED: 1e-6 to 1e-3
    'alpha_2': 1e-6,             # Prior belief about noise level (technical parameter) - RECOMMENDED: 1e-6 to 1e-3
    'lambda_1': 1e-6,            # Prior belief about feature importance (technical parameter) - RECOMMENDED: 1e-6 to 1e-3
    'lambda_2': 1e-6,            # Prior belief about feature importance (technical parameter) - RECOMMENDED: 1e-6 to 1e-3
    'compute_score': False,       # Whether to calculate extra statistics (False = faster)
    'fit_intercept': True        # Whether to add a bias term (almost always True)
}

# -----------------------------------------------------------------------------
# LightGBM Meta Model Settings (Combines predictions from other models)
# -----------------------------------------------------------------------------
LIGHTGBM_PARAMS = {
    'n_estimators': 100,          # Number of boosting rounds - RECOMMENDED: 50-500
    'learning_rate': 0.1,         # How fast it learns (smaller = more careful) - RECOMMENDED: 0.01-0.3
    'num_leaves': 31,            # Maximum leaves per tree (controls complexity) - RECOMMENDED: 10-100
    'max_depth': 6,              # Maximum tree depth - RECOMMENDED: 3-12
    'min_child_samples': 1,       # Minimum samples needed in each leaf - RECOMMENDED: 1-20
    'min_split_gain': 0,         # Minimum improvement needed to split - RECOMMENDED: 0-1
    'subsample': 1.0,            # Fraction of data to use for each tree (1.0 = all data) - RECOMMENDED: 0.7-1.0
    'colsample_bytree': 1.0,     # Fraction of features to use for each tree (1.0 = all features) - RECOMMENDED: 0.7-1.0
    'reg_alpha': 0,              # L1 regularization (prevents overfitting) - RECOMMENDED: 0-10
    'reg_lambda': 0,             # L2 regularization (prevents overfitting) - RECOMMENDED: 0-10
    'random_state': 42,          # Uses our reproducibility number
    'verbose': -1                # Don't print training messages
}

# -----------------------------------------------------------------------------
# Quantile Random Forest Settings (Provides uncertainty estimates)
# -----------------------------------------------------------------------------
QUANTILE_RF_PARAMS = {
    'n_estimators': 100,          # Number of trees for uncertainty estimation - RECOMMENDED: 50-200
    'max_depth': None,            # Maximum tree depth (None = no limit) - RECOMMENDED: 10-20 or None
    'min_samples_split': 2,       # Minimum samples needed to split - RECOMMENDED: 2-10
    'min_samples_leaf': 1,        # Minimum samples per leaf - RECOMMENDED: 1-5
    'max_features': 'sqrt',       # Features to consider per split - RECOMMENDED: 'sqrt'
    'random_state': SEED          # Uses our reproducibility number
}

# =============================================================================
# PERFORMANCE TUNING GUIDELINES
# =============================================================================
# How to adjust the settings above based on what problems you're seeing:

# If your model is OVERFITTING (works great on training data but poorly on test data):
# - Make numbers SMALLER: n_estimators, iterations, max_depth, num_leaves
# - Make numbers BIGGER: min_samples_split, min_samples_leaf, min_child_samples
# - Add more restrictions: increase l2_leaf_reg, reg_alpha, reg_lambda

# If your model is UNDERFITTING (works poorly on both training and test data):
# - Make numbers BIGGER: n_estimators, iterations, max_depth, num_leaves
# - Make numbers SMALLER: min_samples_split, min_samples_leaf, min_child_samples
# - Remove restrictions: decrease l2_leaf_reg, reg_alpha, reg_lambda

# If you want FASTER TRAINING:
# - Make smaller: n_estimators, iterations, max_depth
# - Make bigger: learning_rate (but then reduce iterations to compensate)

# If you want BETTER ACCURACY:
# - Make bigger: n_estimators, iterations (but use smaller learning_rate)
# - Fine-tune: max_depth, num_leaves based on your dataset size

# =============================================================================
# CLOTHING INSULATION VALUES (CLO scores)
# =============================================================================
# How much insulation different clothing provides (used for thermal comfort calculations)
CLO_MAPPING = {
    'T-Shirt': 0.2,
    'Short sleeves shirt (Poly/cotton)': 0.25,
    'Long sleeves shirt (Poly/cotton)': 0.3,
    'Jacket/wwoolen jacket': 0.4,
    'Pullover/Sweater/upcoller': 0.3,
    'Thermal tops': 0.4,
    'Suit': 0.6,
    'Tights': 0.2,
    'Pyjamas': 0.3,
    'Lower (thermal inner)': 0.3,
    'Dhoti': 0.2,
    'Jeans': 0.35,
    'Trousers/long skirt (Poly/cotton)': 0.35,
    'Shorts/short skirt (Poly/cotton)': 0.2
}

# =============================================================================
# ACTIVITY LEVELS (MET scores)
# =============================================================================
# How much energy different activities use (used for thermal comfort calculations)
MET_MAPPING = {
    'Sleeping hrs': 0.9,
    'Sitting (passive work) hrs': 1.0,
    'Sitting (Active work) hrs': 1.3,
    'Standing (relaxed )hrs': 1.5,
    'Standing (working)': 1.8,
    'Walking Indoors (hrs)': 2.0,
    'Walking (Outdoor) hrs': 2.5,
    'Others hrs': 1.2
}

# =============================================================================
# PERCEPTION SCALES
# =============================================================================
# How to convert text descriptions to numbers for machine learning

# Humidity perception scale (how dry/humid it feels)
HUMIDITY_SCALE = {
    "very dry": -3,
    "moderately dry": -2,
    "slightly dry": -1,
    "neutral": 0,
    "slightly humid": 1,
    "moderately humid": 2,
    "very humid": 3
}

# Air movement perception scale (how still/moving the air feels)
AIR_SCALE = {
    "very still": -2,
    "moderately still": -1,
    "slightly still": 0,
    "acceptable": 0,
    "slightly moving": 1,
    "moderately moving": 2,
    "much moving": 3
}

# Lighting perception scale (how dim/bright it feels)
LIGHT_SCALE = {
    "very dim": -3,
    "dim": -2,
    "slightly dim": -1,
    "neither bright nor neither dim": 0,
    "slightly bright": 1,
    "bright": 2,
    "very bright": 3
}

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================
# Other important settings for the thermal comfort prediction system

MODEL_DIR = "models/saved/"                    # Where to save trained models
TSV_COMFORT_RANGE = (-1, 1)                   # Range considered "comfortable" for TSV predictions
ENABLE_RULE_CORRECTION = True                 # Whether to apply rule-based corrections to predictions
TSV_CLIP_RANGE = (-3.0, 3.0)                 # Limit TSV predictions to this range (prevents unrealistic values)

# =============================================================================
# VALIDATION FUNCTION
# =============================================================================
# This function checks that your train/test split settings make sense

def validate_split_config():
    """
    Checks that train and test percentages add up to 100 and are reasonable.
    This prevents common configuration mistakes.
    """
    total = TRAIN_SIZE_PERCENT + TEST_SIZE_PERCENT
    if total != 100:
        raise ValueError(f"Train and test percentages must add up to 100. Current: {TRAIN_SIZE_PERCENT} + {TEST_SIZE_PERCENT} = {total}")
    
    if TRAIN_SIZE_PERCENT <= 0 or TEST_SIZE_PERCENT <= 0:
        raise ValueError("Train and test percentages must be positive numbers")
    
    if TRAIN_SIZE_PERCENT >= 100 or TEST_SIZE_PERCENT >= 100:
        raise ValueError("Train and test percentages must be less than 100")

# Automatically check the configuration when this file is loaded
validate_split_config()
