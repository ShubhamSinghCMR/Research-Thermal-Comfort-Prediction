"""
Improved feature engineering module for Thermal Comfort Prediction.
- Uses only specified input features for each environment
- Adds smarter missing value handling
- Adds outlier capping
- Uses RobustScaler for stable scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def clean_numeric_string(x):
    """Clean numeric strings and handle missing indicators."""
    if isinstance(x, str):
        x = x.strip().replace(' ', '')
        if x in ['', 'nan', 'NA', 'N/A']:
            return np.nan
    return x

def cap_outliers(X, factor=1.5):
    """
    Cap outliers using IQR method for numeric columns.
    """
    X_capped = X.copy()
    numeric_cols = X_capped.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = X_capped[col].quantile(0.25)
        Q3 = X_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        X_capped[col] = np.clip(X_capped[col], lower, upper)
    
    return X_capped

def handle_missing_values(X):
    """
    Handle missing values:
    - Numeric: median imputation
    - Categorical: mode imputation
    """
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])
    return X

def scale_features(X):
    """
    Scale numeric columns using RobustScaler.
    Encode categorical columns (Clothing, Activity) as category codes.
    """
    X_scaled = X.copy()
    categorical_cols = ['Clothing', 'Activity']
    numeric_cols = [col for col in X_scaled.columns if col not in categorical_cols]

    scaler = RobustScaler()
    X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])

    for col in categorical_cols:
        if col in X_scaled.columns:
            X_scaled[col] = X_scaled[col].astype('category').cat.codes

    return X_scaled

def load_and_preprocess_data(sheet_name):
    """
    Load and preprocess data for a specific environment.
    Returns:
        X_scaled: scaled/encoded features used for model
        y: target values
        X_original: original unscaled features for reporting
    """
    df = pd.read_excel('dataset/input_dataset.xlsx', sheet_name=sheet_name)
    df.columns = [col.strip() for col in df.columns]

    # Required columns
    if sheet_name == 'Classroom':
        required_columns = ['RATemp', 'MRT', 'Top', 'Air Velo', 'RH']
    else:
        required_columns = ['RATemp', 'MRT', 'Top', 'Air Velo', 'RH', 'Clothing', 'Activity']

    # Build feature dataframe
    X = pd.DataFrame()
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in {sheet_name}")
        X[col] = df[col]

    # Clean and convert numeric values
    for col in X.columns:
        X[col] = X[col].apply(clean_numeric_string)
        X[col] = pd.to_numeric(X[col], errors='ignore')

    # Save original copy before scaling
    X_original = X.copy()

    # Handle missing values
    X = handle_missing_values(X)

    # Cap outliers
    X = cap_outliers(X)

    # Scale features
    X_scaled = scale_features(X)

    # Target
    target_col = 'Given Final TSV'
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} not found in {sheet_name}")
    y = pd.to_numeric(df[target_col].apply(clean_numeric_string), errors='coerce')

    valid_mask = ~y.isna()
    X_scaled = X_scaled[valid_mask]
    y = y[valid_mask]
    X_original = X_original[valid_mask]

    return X_scaled, y, X_original

def get_all_sheet_names():
    """Return all environment sheet names."""
    excel_file = pd.ExcelFile('dataset/input_dataset.xlsx')
    return excel_file.sheet_names
