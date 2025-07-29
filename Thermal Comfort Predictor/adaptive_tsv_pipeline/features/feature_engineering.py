"""
Adaptive Feature Engineering Module for Thermal Comfort Prediction
==================================================================

This module handles data preprocessing and feature engineering for the adaptive pipeline. 
It dynamically adapts to new or extra columns and ensures robust preprocessing.

Key Features:
-------------
- Automatically detects relevant numeric and categorical features
- Ignores unknown or ID-like columns
- Handles missing values using median/mode imputation
- Caps outliers using IQR method for numeric features
- Robustly scales numeric features and encodes categories dynamically

Dependencies:
-------------
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- sklearn.preprocessing: Robust feature scaling and encoding

Main Functions:
---------------
- load_and_preprocess_data: Main entry point for data preprocessing
- clean_numeric_string: Cleans and standardizes numeric inputs
- cap_outliers: Handles outliers using IQR method
- handle_missing_values: Imputes missing values
- scale_features: Scales numeric features and encodes categories
- get_all_sheet_names: Returns sheet names for adaptive environments
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'input_dataset.xlsx')

# ==========================
# Utility Functions
# ==========================

def clean_numeric_string(x):
    """
    Clean and standardize numeric string inputs, handling various missing value indicators.
    
    Parameters
    ----------
    x : str or numeric
        Input value to clean
        
    Returns
    -------
    float or np.nan
        Cleaned numeric value or np.nan for missing values
    """
    if isinstance(x, str):
        x = x.strip().replace(' ', '')
        if x in ['', 'nan', 'NA', 'N/A']:
            return np.nan
    return x


def cap_outliers(X, factor=1.5):
    """
    Cap outliers in numeric columns using the Interquartile Range (IQR) method.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Input features
    factor : float, default=1.5
        IQR multiplier for determining outlier boundaries
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with capped outliers
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
    Handle missing values in the dataset using appropriate imputation strategies.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Input features with missing values
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with imputed values
    """
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])
    return X


def scale_features(X):
    """
    Scale numeric features and encode categorical variables dynamically.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Input features to scale
        
    Returns
    -------
    pandas.DataFrame
        Scaled and encoded features
    """
    X_scaled = X.copy()

    # Detect categorical & numeric features dynamically
    categorical_cols = X_scaled.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns.tolist()

    # Scale numeric columns
    scaler = RobustScaler()
    if numeric_cols:
        X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])

    # Encode categorical columns
    for col in categorical_cols:
        X_scaled[col] = X_scaled[col].astype('category').cat.codes

    return X_scaled


def load_and_preprocess_data(sheet_name):
    """
    Load and preprocess data for a specific environment from the Excel dataset.
    
    Parameters
    ----------
    sheet_name : str
        Name of the environment sheet in the Excel file
        
    Returns
    -------
    tuple
        X_scaled : pandas.DataFrame
            Scaled features ready for model training
        y : pandas.Series
            Target values (Given Final TSV)
        X_original : pandas.DataFrame
            Original unscaled features for visualization and output
    """
    df = pd.read_excel(DATASET_PATH, sheet_name=sheet_name)
    df.columns = [col.strip() for col in df.columns]

    target_col = 'Given Final TSV'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {sheet_name}")

    # Drop target from features
    X = df.drop(columns=[target_col])

    # Drop irrelevant or ID-like columns
    drop_cols = [col for col in X.columns if "id" in col.lower() or X[col].isna().all()]
    X = X.drop(columns=drop_cols)

    # Clean and convert numeric values
    for col in X.columns:
        X[col] = X[col].apply(clean_numeric_string)
        X[col] = pd.to_numeric(X[col], errors='ignore')

    # Handle missing values
    X = handle_missing_values(X)

    # Cap outliers in numeric columns
    X = cap_outliers(X)

    # Keep original before scaling
    X_original = X.copy()

    # Scale and encode features
    X_scaled = scale_features(X)

    # Clean target column
    y = pd.to_numeric(df[target_col].apply(clean_numeric_string), errors='coerce')

    # Remove invalid target rows
    valid_mask = ~y.isna()
    X_scaled = X_scaled[valid_mask]
    X_original = X_original[valid_mask]
    y = y[valid_mask]

    return X_scaled, y, X_original


def get_all_sheet_names():
    """
    Get all environment sheet names from the input dataset.
    
    Returns
    -------
    list
        List of sheet names representing different environments
    """
    excel_file = pd.ExcelFile(DATASET_PATH)
    return excel_file.sheet_names
