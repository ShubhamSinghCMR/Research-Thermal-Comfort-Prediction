# selective_features_pipeline/models/meta_model.py

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from models.base_models import (
    get_catboost_model,
    get_xgboost_model,
    get_extratrees_model,
    get_elasticnet_model
)


def train_meta_model(X, y, top_k=10, n_splits=5):
    """
    Trains base models on top-K features and uses their predictions as inputs
    to train a stacking meta-model.
    Returns the trained meta-model and validation metrics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    r2_scores = []
    rmse_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"\n🔁 Fold {fold + 1}/{n_splits}")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train each base model with top-K features
        cb_model, cb_feats = get_catboost_model(X_train, y_train, top_k)
        xgb_model, xgb_feats = get_xgboost_model(X_train, y_train, top_k)
        et_model, et_feats = get_extratrees_model(X_train, y_train, top_k)
        en_model, en_feats = get_elasticnet_model(X_train, y_train, top_k)

        # Get base model predictions on test set
        cb_pred = cb_model.predict(X_test[cb_feats])
        xgb_pred = xgb_model.predict(X_test[xgb_feats])
        et_pred = et_model.predict(X_test[et_feats])
        en_pred = en_model.predict(X_test[en_feats])

        # Stack base model predictions
        stacked_preds_train = np.vstack([cb_pred, xgb_pred, et_pred, en_pred]).T

        # Train meta-model on stacked base predictions
        meta_model = LinearRegression()
        meta_model.fit(stacked_preds_train, y_test)

        # Predict with meta-model
        final_pred = meta_model.predict(stacked_preds_train)

        # Evaluation
        r2 = r2_score(y_test, final_pred)
        rmse = mean_squared_error(y_test, final_pred, squared=False)

        r2_scores.append(r2)
        rmse_scores.append(rmse)

        print(f"✅ R2 Score: {r2:.4f} | RMSE: {rmse:.4f}")

    print("\n📊 Final Cross-Validation Results")
    print(f"Avg R2 Score: {np.mean(r2_scores):.4f}")
    print(f"Avg RMSE: {np.mean(rmse_scores):.4f}")

    return meta_model
