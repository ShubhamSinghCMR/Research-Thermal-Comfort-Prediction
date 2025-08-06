# selective_features_pipeline/features/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet


def load_and_preprocess_data(df):
    """Scales features and separates target."""
    X = df.drop(columns=["Thermal Sensation"])  # Replace with actual target column name if different
    y = df["Thermal Sensation"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y


def get_model_feature_importance(model, X, y, model_name):
    """Fits model and returns feature importances as sorted Series."""
    model.fit(X, y)

    if model_name == "ElasticNet":
        importances = abs(model.coef_)
    else:
        importances = model.feature_importances_

    return pd.Series(importances, index=X.columns).sort_values(ascending=False)


def compute_all_feature_importances(X, y, top_k=10):
    """
    Train all models and extract top-K feature importances.
    Prints:
      - all available features
      - top-k selected features for each model
    """
    models = {
        "CatBoost": CatBoostRegressor(verbose=0),
        "XGBoost": XGBRegressor(verbosity=0),
        "ExtraTrees": ExtraTreesRegressor(),
        "ElasticNet": ElasticNet()
    }

    print("\n✅ All Available Features:")
    print(list(X.columns))
    print("-" * 60)

    feature_rankings = {}
    top_features_by_model = {}

    for name, model in models.items():
        print(f"\n📌 Training and extracting features for: {name}")
        try:
            ranking = get_model_feature_importance(model, X, y, name)
            feature_rankings[name] = ranking

            top_features = list(ranking.head(top_k).index)
            top_features_by_model[name] = top_features

            print(f"🔹 Top {top_k} Features Used by {name}:")
            for i, feat in enumerate(top_features, 1):
                print(f"{i}. {feat}")
        except Exception as e:
            print(f"[{name}] Failed: {e}")

    return feature_rankings, top_features_by_model


def select_top_features(feature_importance: pd.Series, top_k=10):
    """Returns top_k feature names from a single model’s feature importance ranking."""
    return list(feature_importance.head(top_k).index)
