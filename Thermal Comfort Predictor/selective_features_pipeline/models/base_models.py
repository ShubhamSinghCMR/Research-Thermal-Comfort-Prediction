# selective_features_pipeline/models/base_models.py

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from features.feature_engineering import compute_all_feature_importances


def get_catboost_model(X, y, top_k=10):
    model = CatBoostRegressor(verbose=0)
    feature_importances, top_features_by_model = compute_all_feature_importances(X, y, top_k)
    top_features = top_features_by_model['CatBoost']
    model.fit(X[top_features], y)
    return model, top_features


def get_xgboost_model(X, y, top_k=10):
    model = XGBRegressor(verbosity=0)
    feature_importances, top_features_by_model = compute_all_feature_importances(X, y, top_k)
    top_features = top_features_by_model['XGBoost']
    model.fit(X[top_features], y)
    return model, top_features


def get_extratrees_model(X, y, top_k=10):
    model = ExtraTreesRegressor()
    feature_importances, top_features_by_model = compute_all_feature_importances(X, y, top_k)
    top_features = top_features_by_model['ExtraTrees']
    model.fit(X[top_features], y)
    return model, top_features


def get_elasticnet_model(X, y, top_k=10):
    model = ElasticNet()
    feature_importances, top_features_by_model = compute_all_feature_importances(X, y, top_k)
    top_features = top_features_by_model['ElasticNet']
    model.fit(X[top_features], y)
    return model, top_features
