"""
Feature Engineering Module for Thermal Comfort Prediction
======================================================

This module handles data preprocessing and feature engineering for thermal comfort prediction.
It includes functions for data cleaning, outlier handling, missing value imputation, and feature scaling.

Key Features:
------------
- Environment-specific feature selection
- Smart missing value handling with median/mode imputation
- Outlier capping using IQR method
- Robust scaling of numeric features
- Categorical encoding for clothing and activity features

Dependencies:
------------
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- sklearn.preprocessing: Feature scaling

Main Functions:
-------------
- load_and_preprocess_data: Main entry point for data preprocessing
- clean_numeric_string: Cleans and standardizes numeric inputs
- cap_outliers: Handles outliers using IQR method
- handle_missing_values: Imputes missing values
- scale_features: Scales numeric features and encodes categories
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

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
    
    Notes
    -----
    - Uses Q1 - factor*IQR for lower bound
    - Uses Q3 + factor*IQR for upper bound
    - Only processes numeric columns
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
        
    Notes
    -----
    Imputation Strategy:
    - Numeric columns: Median imputation
    - Categorical columns: Mode imputation
    """
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])
    return X

def scale_features(X):
    """
    Scale numeric features and encode categorical variables.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Input features to scale
        
    Returns
    -------
    pandas.DataFrame
        Scaled and encoded features
        
    Notes
    -----
    - Uses RobustScaler for numeric features to handle outliers
    - Converts categorical variables to numeric codes
    - Categorical columns: 'Clothing', 'Activity'
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
            
    Notes
    -----
    Processing Steps:
    1. Load data from Excel
    2. Select required features based on environment
    3. Clean numeric values
    4. Handle missing values
    5. Cap outliers
    6. Scale features
    7. Filter invalid target values
    
    Required Features:
    - Classroom: RATemp, MRT, Top, Air Velo, RH
    - Other environments: Above + Clothing, Activity
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

    # Handle missing values
    X = handle_missing_values(X)

    # Cap outliers
    X = cap_outliers(X)
    
    # Keep original version before scaling
    X_original = X.copy()

    # Scale features
    X_scaled = scale_features(X)

    # Target
    target_col = 'Given Final TSV'
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} not found in {sheet_name}")
    y = pd.to_numeric(df[target_col].apply(clean_numeric_string), errors='coerce')

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
    excel_file = pd.ExcelFile('dataset/input_dataset.xlsx')
    return excel_file.sheet_names
