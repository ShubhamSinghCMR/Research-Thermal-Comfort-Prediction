
import numpy as np
import pandas as pd

def _bin_stratify_targets(y: np.ndarray, n_bins: int = 7):
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y, q))
    if len(edges) <= 2:
        edges = np.linspace(y.min(), y.max(), n_bins + 1)
    return np.digitize(y, edges[1:-1], right=True)

def _fit_ridge_blender(P, y, alphas):
    from sklearn.linear_model import RidgeCV
    mu, sig = P.mean(axis=0), P.std(axis=0) + 1e-9
    Pz = (P - mu) / sig
    ridge = RidgeCV(alphas=alphas, fit_intercept=True, cv=5)
    ridge.fit(Pz, y)
    return {"type": "ridge", "model": ridge, "mu": mu, "sig": sig}

def _predict_ridge_blender(model_dict, P):
    ridge, mu, sig = model_dict["model"], model_dict["mu"], model_dict["sig"]
    return ridge.predict((P - mu) / sig)

def _fit_nnls_blender(P, y):
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression(positive=True)
    lr.fit(P, y)
    w = lr.coef_.clip(min=0)
    s = w.sum() if w.sum() > 0 else 1.0
    return {"type": "nnls", "w": (w / s), "b": lr.intercept_}

def _predict_nnls_blender(model_dict, P):
    return P @ model_dict["w"] + model_dict["b"]

def _fit_avg_blender(P, y):
    k = P.shape[1]
    w = np.ones(k) / max(1, k)
    return {"type": "avg", "w": w}

def _predict_avg_blender(model_dict, P):
    return P @ model_dict["w"]

def _fit_lgbm_meta(P, y, env_params):
    import lightgbm as lgb
    lgb_params = env_params['lightgbm'].copy()
    lgb_params.setdefault('objective', 'huber')
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(P, y, callbacks=[lgb.log_evaluation(period=0)])
    return {"type": "lgbm", "model": model}

def _predict_lgbm_meta(model_dict, P):
    return model_dict["model"].predict(P)

def _cv_splits(y, n_splits=5, n_repeats=1, random_state=42, stratify_bins=0):
    from sklearn.model_selection import KFold, StratifiedKFold
    y = np.asarray(y)
    splits = []
    if stratify_bins and stratify_bins > 1:
        yb = _bin_stratify_targets(y, stratify_bins)
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + rep)
            splits += list(skf.split(y, yb))
    else:
        for rep in range(n_repeats):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state + rep)
            splits += list(kf.split(y))
    return splits

def _evaluate_cv(P, y, fit_fn, pred_fn, splits, eval_fn):
    oof = np.zeros(len(y))
    fold_metrics = []
    for tr, va in splits:
        mdl = fit_fn(P[tr], y[tr])
        yp = pred_fn(mdl, P[va])
        oof[va] = yp
        fold_metrics.append(eval_fn(pd.Series(y[va]), pd.Series(yp)))
    return oof, fold_metrics

def train_meta_model_kfold(oof_preds, y, env_params, n_splits=5):
    """Train a meta-learner on OOF base predictions.

    Supports STACKING_METHOD in config: 'ridge', 'nnls', 'avg', 'lgbm'.
    If AUTO_SELECT_STACKER=True, evaluates STACKER_CANDIDATES and selects the best by CV R2.
    """
    from utils.metrics import evaluate_predictions
    try:
        from utils.config import (
            STACKING_METHOD, AUTO_SELECT_STACKER, STACKER_CANDIDATES,
            META_RIDGE_ALPHAS, CV_REPEATS, STRATIFY_BINS
        )
    except Exception:
        STACKING_METHOD = "ridge"
        AUTO_SELECT_STACKER = False
        STACKER_CANDIDATES = ["ridge","nnls","avg","lgbm"]
        META_RIDGE_ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0]
        CV_REPEATS = 1
        STRATIFY_BINS = 0

    P = oof_preds.to_numpy()
    y_arr = y.to_numpy()
    splits = _cv_splits(y_arr, n_splits=n_splits, n_repeats=CV_REPEATS, random_state=42, stratify_bins=STRATIFY_BINS)

    def make_runner(method):
        if method == "ridge":
            return (lambda X, yy: _fit_ridge_blender(X, yy, META_RIDGE_ALPHAS), _predict_ridge_blender)
        if method == "nnls":
            return (_fit_nnls_blender, _predict_nnls_blender)
        if method == "avg":
            return (_fit_avg_blender, _predict_avg_blender)
        if method == "lgbm":
            return (lambda X, yy: _fit_lgbm_meta(X, yy, env_params), _predict_lgbm_meta)
        raise ValueError(f"Unknown STACKING method: {method}")

    methods_to_try = STACKER_CANDIDATES if AUTO_SELECT_STACKER else [STACKING_METHOD]

    best = None
    results = {}
    for m in methods_to_try:
        fit_fn, pred_fn = make_runner(m)
        oof, fold_metrics = _evaluate_cv(P, y_arr, fit_fn, pred_fn, splits, eval_fn=evaluate_predictions)
        avg_metrics = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0]}
        results[m] = {"oof": oof, "metrics": avg_metrics}
        if (best is None) or (avg_metrics.get("R2", -1e9) > best["metrics"].get("R2", -1e9)):
            best = {"method": m, "oof": oof, "metrics": avg_metrics, "fit_fn": fit_fn, "pred_fn": pred_fn}

    final_model = best["fit_fn"](P, y_arr)

    # Feature importances for linear methods are absolute weights; for others, fallback
    try:
        if best["method"] == "ridge":
            importances = np.abs(final_model["model"].coef_)
        elif best["method"] == "nnls":
            importances = np.abs(final_model["w"])
        elif best["method"] == "lgbm":
            importances = final_model["model"].feature_importances_
        else:
            importances = np.ones(P.shape[1])
    except Exception:
        importances = np.ones(P.shape[1])

    fi = pd.DataFrame({"feature": oof_preds.columns, "importance": importances}).sort_values("importance", ascending=False)

    return {
        "oof_predictions": best["oof"],
        "cv_metrics": best["metrics"],
        "feature_importance": fi,
        "model": final_model,
        "stacking_method": best["method"],
        "all_methods": {k: v["metrics"] for k, v in results.items()},
    }
