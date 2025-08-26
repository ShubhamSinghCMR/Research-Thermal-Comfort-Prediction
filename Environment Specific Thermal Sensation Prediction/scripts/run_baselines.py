
import os
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import KFold

# sklearn regressors
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# boosting libs
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

from features.feature_engineering import get_all_sheet_names, load_environment_sheet
from utils.config import (
    get_environment_params, 
    USE_STACKING, CALIBRATION_ON, 
    FOLDS, TOP_M_PER_FOLD, STABILITY_TAU, PERM_REPEATS, KMIN, KMAX,
    SELECTION_MODE, SELECTION_SCOPE, BASELINE_MODELS
)
from utils.preprocess import build_fold_preprocessor
from utils.feature_ranking import univariate_scores, permutation_rank
from utils.selection import stability_selection
from utils.metrics import evaluate_predictions
from utils.calibration import compute_bias_and_qhat, apply_calibration

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

def ensure_output():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def make_model(name, params):
    if name == "svr_rbf":
        return SVR(**params)
    elif name == "random_forest":
        return RandomForestRegressor(**params)
    elif name == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    elif name == "decision_tree":
        return DecisionTreeRegressor(**params)
    elif name == "linear_regression":
        return LinearRegression(**params)
    elif name == "ridge":
        return Ridge(**params)
    elif name == "lasso":
        return Lasso(**params)
    elif name == "knn_distance":
        # enforce distance weighting unless user overrides
        params = dict(params)
        params.setdefault("weights", "distance")
        return KNeighborsRegressor(**params)
    elif name == "mlp":
        return MLPRegressor(**params)
    elif name == "xgboost_single":
        return XGBRegressor(**params)
    elif name == "lightgbm_single":
        return lgb.LGBMRegressor(**params)
    elif name == "catboost_single":
        return CatBoostRegressor(**params)
    elif name == "adaboost":
        return AdaBoostRegressor(**params)
    else:
        raise ValueError(f"Unknown baseline model: {name}")

def combine_ranks(u_rank, p_rank, mode):
    if mode == "none":
        return None
    if mode == "univariate_only":
        return u_rank
    if mode == "permutation_only":
        return p_rank
    # hybrid: 0.4 * univariate + 0.6 * permutation (to match config doc)
    keys = set(u_rank.keys()) | set(p_rank.keys())
    return {k: 0.4*float(u_rank.get(k, 0.0)) + 0.6*float(p_rank.get(k, 0.0)) for k in keys}


def run_single_model(model_key, env_name, df, env_params):
    target = "Given Final TSV"
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in sheet '{env_name}'")
    Xdf = df.drop(columns=[target])
    y = df[target]

    # KFold CV
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=env_params.get(model_key, {}).get("random_state", 42))
    oof = np.zeros(len(Xdf))
    fold_metrics = []
    picks_per_fold = []
    scores_per_fold = []
    selected_features_final = None

    # Record predictions per-row to save later
    preds_df = df.copy()
    preds_df["TSV_Predicted_raw"] = np.nan
    preds_df["TSV_Predicted"] = np.nan
    preds_df["TSV_Lower"] = np.nan
    preds_df["TSV_Upper"] = np.nan

    for fold, (tr, va) in enumerate(kf.split(Xdf), 1):
        Xtr_df, Xva_df = Xdf.iloc[tr].copy(), Xdf.iloc[va].copy()
        ytr, yva = y.iloc[tr].copy(), y.iloc[va].copy()

        pre, transform_df, num_cols, cat_cols, orig_to_idx = build_fold_preprocessor(Xtr_df)

        # Univariate ranking on TRAIN only
        u_scores, u_rank = univariate_scores(Xtr_df, ytr, num_cols, cat_cols)

        # Candidate features per fold (top M by chosen ranking)
        combined = None
        dR2_rank = {}
        feat_set = list(Xtr_df.columns)
        if SELECTION_MODE != "none":
            # temporary fit with ALL features to compute permutation ranks
            Xtr_t = transform_df(Xtr_df); Xva_t = transform_df(Xva_df)
            mdl_tmp = make_model(model_key, env_params[model_key])
            # Some libs need eval_set signature variants
            try:
                mdl_tmp.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)], verbose=False)
            except Exception:
                try:
                    mdl_tmp.fit(Xtr_t, ytr, eval_set=(Xva_t, yva), verbose=False)
                except Exception:
                    mdl_tmp.fit(Xtr_t, ytr)

            def predict_fn(Xmat, mdl=mdl_tmp):
                return mdl.predict(Xmat)

            _, _, _, dR2_rank = permutation_rank(predict_fn, Xva_t, yva, orig_to_idx, repeats=PERM_REPEATS)
            combined = combine_ranks(u_rank, dR2_rank, SELECTION_MODE)

            # Top-M per fold
            if combined:
                ordered = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
                feat_set = [k for k,_ in ordered[:TOP_M_PER_FOLD]]

        picks_per_fold.append(list(feat_set))
        if combined is None:
            scores_per_fold.append({f: 1.0 for f in feat_set})
        else:
            scores_per_fold.append({f: float(combined.get(f, 0.0)) for f in feat_set})

    # Stability selection across folds (global pool)
    pool, freq_map, mean_scores = stability_selection(picks_per_fold, scores_per_fold, 
                                                      tau=STABILITY_TAU, kmin=KMIN, kmax=KMAX)
    selected_features_final = pool if pool else list(Xdf.columns)

    # Train-final CV with the selected features
    oof = np.zeros(len(Xdf))
    fold_metrics = []
    kf2 = KFold(n_splits=FOLDS, shuffle=True, random_state=env_params.get(model_key, {}).get("random_state", 42))

    for fold, (tr, va) in enumerate(kf2.split(Xdf), 1):
        Xtr_df, Xva_df = Xdf.iloc[tr][selected_features_final].copy(), Xdf.iloc[va][selected_features_final].copy()
        ytr, yva = y.iloc[tr].copy(), y.iloc[va].copy()

        pre, transform_df, num_cols, cat_cols, orig_to_idx = build_fold_preprocessor(Xtr_df)
        Xtr_t = transform_df(Xtr_df); Xva_t = transform_df(Xva_df)

        mdl = make_model(model_key, env_params[model_key])
        # Fit with eval_set where supported
        try:
            mdl.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)], verbose=False)
        except Exception:
            try:
                mdl.fit(Xtr_t, ytr, eval_set=(Xva_t, yva), verbose=False)
            except Exception:
                mdl.fit(Xtr_t, ytr)

        yp = mdl.predict(Xva_t)
        oof[va] = yp
        fold_metrics.append(evaluate_predictions(yva, yp))

        # store raw fold preds to preds_df
        preds_df.loc[preds_df.index[va], "TSV_Predicted_raw"] = yp

    # Calibration on OOF
    metrics = {}
    if CALIBRATION_ON:
        bias, qhat = compute_bias_and_qhat(y, oof, alpha=0.10)
        y_cal = np.array([apply_calibration(p, bias, (-3, 3)) for p in oof])
        metrics = evaluate_predictions(y, y_cal)
        preds_df["TSV_Predicted"] = y_cal
        preds_df["TSV_Lower"] = np.clip(oof - qhat + bias, -3, 3)
        preds_df["TSV_Upper"] = np.clip(oof + qhat + bias, -3, 3)
    else:
        metrics = evaluate_predictions(y, oof)
        preds_df["TSV_Predicted"] = oof

    # Average fold metrics for Train_Accuracy & num_features reporting (approx)
    avg_train_acc = float(np.mean([m["Accuracy"] for m in fold_metrics]))
    avg_num_features = float(len(selected_features_final))

    return {
        "oof": oof,
        "metrics": {**metrics, "Train_Accuracy": avg_train_acc, "Avg_Num_Features": avg_num_features},
        "selected_features": selected_features_final,
        "preds_df": preds_df,
    }

def run_all_baselines():
    ensure_output()
    sheets = get_all_sheet_names("dataset/input_dataset.xlsx")
    comparison_rows = []

    for sheet in sheets:
        df = load_environment_sheet(sheet, "dataset/input_dataset.xlsx")
        env_params = get_environment_params(sheet)

        # Per-environment summary
        env_rows = []

        for key in BASELINE_MODELS:
            res = run_single_model(key, sheet, df, env_params)

            # Save artifacts
            env_tag = sheet.replace(" ", "_")
            tag = f"{env_tag}_{key}"

            # Selected features
            sel_path = os.path.join(OUTPUT_DIR, f"{tag}_selected_features.csv")
            pd.DataFrame({"feature": res["selected_features"]}).to_csv(sel_path, index=False)

            # Predictions
            pred_path = os.path.join(OUTPUT_DIR, f"{tag}_predictions.csv")
            res["preds_df"].to_csv(pred_path, index=False)

            # Metrics
            met_path = os.path.join(OUTPUT_DIR, f"{tag}_metrics.csv")
            pd.DataFrame([res["metrics"]]).to_csv(met_path, index=False)

            # Update per-env and global comparison rows
            env_rows.append({"Model": key, **res["metrics"]})
            comparison_rows.append({"Environment": sheet, "Model": key, **res["metrics"]})

        # Save per-environment baselines summary
        env_sum_path = os.path.join(OUTPUT_DIR, f"{sheet.replace(' ','_')}_baselines_summary.csv")
        pd.DataFrame(env_rows).to_csv(env_sum_path, index=False)

    # Global baselines comparison
    base_comp_path = os.path.join(OUTPUT_DIR, "baselines_comparison.csv")
    pd.DataFrame(comparison_rows).to_csv(base_comp_path, index=False)

    # Combine with stacked-ensemble results if present
    all_rows = []
    # Add ensemble rows from existing pipeline
    for sheet in sheets:
        met = os.path.join(OUTPUT_DIR, f"{sheet.replace(' ','_')}_metrics.csv")
        if os.path.exists(met):
            m = pd.read_csv(met).iloc[0].to_dict()
            all_rows.append({"Environment": sheet, "Model": "stacked_ensemble", **m})
    # Add baselines
    all_rows.extend(comparison_rows)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(os.path.join(OUTPUT_DIR, "comparison_all_models.csv"), index=False)
        print("\\nSaved comparison_all_models.csv")

def main():
    run_all_baselines()

if __name__ == "__main__":
    main()
