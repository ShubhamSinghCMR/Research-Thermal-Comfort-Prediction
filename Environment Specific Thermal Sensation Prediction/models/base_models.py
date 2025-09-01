import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet

from utils.metrics import evaluate_predictions
from utils.preprocess import build_fold_preprocessor
from utils.feature_ranking import univariate_scores, permutation_rank
from utils.selection import stability_selection
from utils.config import (
    FOLDS, TOP_M_PER_FOLD, STABILITY_TAU, PERM_REPEATS,
    SELECTION_MODE, SELECTION_SCOPE, KMIN, KMAX
)
from lightgbm import LGBMRegressor
from sklearn.svm import SVR

def get_base_models_from_params(env_params):
    return {
        "catboost":   CatBoostRegressor(**env_params["catboost"]),
        "lightgbm":   LGBMRegressor(**env_params["lightgbm"]),
        "svr_rbf":    SVR(**env_params["svr_rbf"]),
        "elasticnet": ElasticNet(**env_params["elasticnet"]),
        "xgboost":    XGBRegressor(**env_params["xgboost"]),
    }

def _combine_scores(u_rank, dR2_rank):
    if SELECTION_MODE == 'none':
        return None
    if SELECTION_MODE == 'univariate_only':
        return {f: u_rank.get(f, 0.0) for f in set(u_rank.keys()) | set(dR2_rank.keys())}
    if SELECTION_MODE == 'permutation_only':
        return {f: dR2_rank.get(f, 0.0) for f in set(u_rank.keys()) | set(dR2_rank.keys())}
    return {f: 0.4*u_rank.get(f, 0.0) + 0.6*dR2_rank.get(f, 0.0) for f in set(u_rank.keys()) | set(dR2_rank.keys())}

def train_base_models(X_orig, y, env_params, n_splits=FOLDS, per_model_selection=True):
    X_df = X_orig.copy()
    y = pd.Series(y).reset_index(drop=True)
    kf = KFold(n_splits=n_splits, shuffle=True,
               random_state=env_params["lightgbm"].get("random_state", 42))

    from utils.config import ENSEMBLE_BASE_MODELS
    model_names = ENSEMBLE_BASE_MODELS[:]  # ["catboost","xgboost","lightgbm","elasticnet","svr_rbf"]
    per_model_fold_picks  = {m: [] for m in model_names}
    per_model_fold_scores = {m: [] for m in model_names}
    per_model_stats       = {m: {} for m in model_names}

    for fold, (tr, va) in enumerate(kf.split(X_df, y), 1):
        Xtr_df = X_df.iloc[tr].reset_index(drop=True)
        Xva_df = X_df.iloc[va].reset_index(drop=True)
        ytr = y.iloc[tr].reset_index(drop=True)
        yva = y.iloc[va].reset_index(drop=True)

        pre, transform_fn, num_cols, cat_cols, orig_to_idx = build_fold_preprocessor(Xtr_df)
        Xtr_t = transform_fn(Xtr_df); Xva_t = transform_fn(Xva_df)

        u_scores, u_rank = univariate_scores(Xtr_df, ytr, num_cols, cat_cols)

        fold_combined_by_model = {}
        for m in model_names:
            model = get_base_models_from_params(env_params)[m]
            if m == "xgboost":
                model.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)], verbose=False)
            elif m == "catboost":
                model.fit(Xtr_t, ytr, eval_set=(Xva_t, yva), verbose=False)
            elif m == "lightgbm":
                model.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)])  # uses LGBMRegressor
            else:
                model.fit(Xtr_t, ytr)

            def predict_fn(Xmat, mdl=model): return mdl.predict(Xmat)
            dR2, dMAE, dACC, dR2_rank = permutation_rank(predict_fn, Xva_t, yva, orig_to_idx, repeats=PERM_REPEATS)
            combined = _combine_scores(u_rank, dR2_rank)

            if combined is None:
                picks = list(Xtr_df.columns)
                per_model_fold_picks[m].append(picks)
                per_model_fold_scores[m].append({f:1.0 for f in picks})
            else:
                topM = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:TOP_M_PER_FOLD]
                per_model_fold_picks[m].append([k for k,_ in topM])
                per_model_fold_scores[m].append(combined)

            stats = per_model_stats[m]
            for f in Xtr_df.columns:
                s = stats.setdefault(f, {"u": [], "dR2": [], "dMAE": [], "dACC": [], "comb": []})
                s["u"].append(float(u_scores.get(f, 0.0)))
                s["dR2"].append(float(dR2.get(f, 0.0)))
                s["dMAE"].append(float(dMAE.get(f, 0.0)))
                s["dACC"].append(float(dACC.get(f, 0.0)))
                s["comb"].append(1.0 if combined is None else float(combined.get(f, 0.0)))
            fold_combined_by_model[m] = combined

        if SELECTION_SCOPE == 'global':
            agg = {}
            for m in model_names:
                cmb = fold_combined_by_model[m]
                if cmb is None:
                    agg = {f:1.0 for f in Xtr_df.columns}
                    break
                for f, v in cmb.items():
                    agg[f] = agg.get(f, 0.0) + v
            if agg:
                for f in agg:
                    agg[f] /= float(len(model_names))
                topM = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:TOP_M_PER_FOLD]
                common = [k for k,_ in topM]
                for m in model_names:
                    per_model_fold_picks[m][-1] = common

    selected_features = {}
    selection_reports = {}
    cat_set = set([c for c in ["Season","Clothing","Activity"] if c in X_df.columns])

    for m in model_names:
        pool, freq_map, mean_scores = stability_selection(
            per_model_fold_picks[m], per_model_fold_scores[m],
            tau=STABILITY_TAU, kmin=KMIN, kmax=KMAX
        )
        selected_features[m] = pool

        rows, stats = [], per_model_stats[m]
        keys = set(stats.keys()) | set(freq_map.keys()) | set(mean_scores.keys())
        for f in keys:
            s = stats.get(f, {"u":[], "dR2":[], "dMAE":[], "dACC":[], "comb":[]})
            rows.append({
                "Feature": f,
                "Type": "categorical" if f in cat_set else "numeric",
                "Univariate_Score": float(np.mean(s["u"]))   if s["u"]   else 0.0,
                "Perm_DeltaR2":     float(np.mean(s["dR2"])) if s["dR2"] else 0.0,
                "Perm_DeltaMAE":    float(np.mean(s["dMAE"])) if s["dMAE"] else 0.0,
                "Perm_DeltaAcc":    float(np.mean(s["dACC"])) if s["dACC"] else 0.0,
                "Combined_Score":   float(np.mean(s["comb"])) if s["comb"] else 0.0,
                "Stability_Freq":   freq_map.get(f, 0.0),
                "Selected":         f in pool
            })
        df_rep = pd.DataFrame(rows).sort_values(by=["Selected","Combined_Score"], ascending=[False, False]).reset_index(drop=True)
        df_rep["Rank"] = np.arange(1, len(df_rep)+1)
        selection_reports[m] = df_rep

    oof_preds   = pd.DataFrame(0.0, index=np.arange(len(X_df)), columns=model_names)
    base_results = {m: [] for m in model_names}

    for fold, (tr, va) in enumerate(kf.split(X_df, y), 1):
        Xtr_df = X_df.iloc[tr].reset_index(drop=True)
        Xva_df = X_df.iloc[va].reset_index(drop=True)
        ytr = y.iloc[tr].reset_index(drop=True)
        yva = y.iloc[va].reset_index(drop=True)

        pre, transform_fn, num_cols, cat_cols, orig_to_idx = build_fold_preprocessor(Xtr_df)
        Xtr_t_all = transform_fn(Xtr_df)
        Xva_t_all = transform_fn(Xva_df)

        for m in model_names:
            model = get_base_models_from_params(env_params)[m]
            feat_set = selected_features[m] if SELECTION_MODE != 'none' else list(X_df.columns)
            col_idxs = sorted({i for f in feat_set for i in orig_to_idx.get(f, [])})
            Xtr_t = Xtr_t_all if len(col_idxs)==0 else Xtr_t_all[:, col_idxs]
            Xva_t = Xva_t_all if len(col_idxs)==0 else Xva_t_all[:, col_idxs]

            if m == "xgboost":
                model.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)], verbose=False)
            elif m == "catboost":
                model.fit(Xtr_t, ytr, eval_set=(Xva_t, yva), verbose=False)
            elif m == "lightgbm":
                model.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)])
            else:
                model.fit(Xtr_t, ytr)

            ytr_pred = model.predict(Xtr_t)
            yva_pred = model.predict(Xva_t)

            oof_preds.loc[va, m] = yva_pred
            base_results[m].append({
                "train":        evaluate_predictions(ytr, ytr_pred),
                "valid":        evaluate_predictions(yva, yva_pred),
                "num_features": len(feat_set)
            })

    base_results_avg = {}
    for m, folds in base_results.items():
        keys = list(folds[0]["valid"].keys())
        avg_valid = {k: float(np.mean([f["valid"][k] for f in folds])) for k in keys}
        avg_train_acc = float(np.mean([f["train"]["Accuracy"] for f in folds]))
        avg_valid["Train_Accuracy"]  = avg_train_acc
        avg_valid["Avg_Num_Features"] = float(np.mean([f["num_features"] for f in folds]))
        base_results_avg[m] = avg_valid

    return oof_preds, base_results_avg, selection_reports, selected_features
