import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from models.base_models import evaluate_predictions
from utils.config import SHOW_WARNINGS

def train_meta_model_kfold(oof_preds, y, env_params, n_splits=5):
    lgb_params = env_params["lightgbm"].copy()
    
    # Ensure verbose parameter is set correctly
    if not SHOW_WARNINGS:
        lgb_params["verbose"] = -1
        lgb_params["min_gain_to_split"] = 0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta_model = lgb.LGBMRegressor(**lgb_params)

    oof_meta_preds = np.zeros(len(oof_preds))
    fold_metrics = []
    feature_importances = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(oof_preds, y), 1):
        X_train, X_valid = oof_preds.iloc[train_idx], oof_preds.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        callbacks = [lgb.early_stopping(50)]
        if SHOW_WARNINGS:
            callbacks.append(lgb.log_evaluation())

        meta_model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=callbacks
        )

        y_pred = meta_model.predict(X_valid)
        oof_meta_preds[valid_idx] = y_pred
        metrics = evaluate_predictions(y_valid, y_pred)
        fold_metrics.append(metrics)

        feature_importances.append(meta_model.feature_importances_)

        if SHOW_WARNINGS:
            print(f"[Meta-model Fold {fold}] RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}, R2={metrics['R2']:.3f}")

    avg_metrics = {m: np.mean([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
    feature_importance_df = pd.DataFrame({
        "feature": oof_preds.columns,
        "importance": np.mean(feature_importances, axis=0),
    }).sort_values("importance", ascending=False)

    if SHOW_WARNINGS:
        print(f"\n[Meta-model] Final CV Metrics: RMSE={avg_metrics['RMSE']:.3f}, MAE={avg_metrics['MAE']:.3f}, R2={avg_metrics['R2']:.3f}")

    return {
        "oof_predictions": oof_meta_preds,
        "cv_metrics": avg_metrics,
        "feature_importance": feature_importance_df,
        "model": meta_model,
    }
