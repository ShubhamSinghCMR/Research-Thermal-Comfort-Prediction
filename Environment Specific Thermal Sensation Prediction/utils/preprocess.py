import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

class CapIQRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)

def detect_cols(X_df):
    cat_cols = [c for c in ["Season","Clothing","Activity"] if c in X_df.columns]
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    return num_cols, cat_cols

def build_fold_preprocessor(X_train_df):
    X = X_train_df.copy()
    num_cols, cat_cols = detect_cols(X)

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    for c in cat_cols:
        X[c] = X[c].astype('category')

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("cap", CapIQRTransformer()),
        ("scale", RobustScaler())
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=True,
        sparse_threshold=0.0
    )

    pre.fit(X)

    feat_names = pre.get_feature_names_out()
    orig_to_idx = {c: [] for c in num_cols + cat_cols}
    for idx, name in enumerate(feat_names):
        if name.startswith("num__"):
            orig = name.split("num__")[1]
            orig_to_idx[orig].append(idx)
        elif name.startswith("cat__"):
            tail = name.split("cat__")[1]
            orig = tail.split("_")[0]
            orig_to_idx[orig].append(idx)

    def transform_df(df):
        df2 = df.copy()
        for c in num_cols:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')
        for c in cat_cols:
            df2[c] = df2[c].astype('category')
        return pre.transform(df2)

    return pre, transform_df, num_cols, cat_cols, orig_to_idx
