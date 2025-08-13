import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from utils.metrics import evaluate_predictions

def train_meta_model_kfold(oof_preds, y, env_params, n_splits=5):
    lgb_params = env_params['lightgbm'].copy()
    if 'objective' not in lgb_params:
        lgb_params['objective'] = 'huber'
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=lgb_params.get('random_state',42))
    meta_model = lgb.LGBMRegressor(**lgb_params)
    oof_meta = np.zeros(len(oof_preds))
    fold_metrics = []
    importances = []
    for fold, (tr, va) in enumerate(kf.split(oof_preds, y), 1):
        Xtr, Xva = oof_preds.iloc[tr], oof_preds.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        meta_model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)]
        )
        yp = meta_model.predict(Xva)
        oof_meta[va] = yp
        fold_metrics.append(evaluate_predictions(yva, yp))
        importances.append(meta_model.feature_importances_)
    avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    fi = pd.DataFrame({"feature": oof_preds.columns, "importance": np.mean(importances, axis=0)}).sort_values("importance", ascending=False)
    return {"oof_predictions": oof_meta, "cv_metrics": avg_metrics, "feature_importance": fi, "model": meta_model}
