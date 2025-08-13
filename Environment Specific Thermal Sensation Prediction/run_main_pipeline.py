import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import dump

from features.feature_engineering import get_all_sheet_names, load_environment_sheet
from utils.config import get_environment_params, USE_STACKING, CALIBRATION_ON
from utils.metrics import evaluate_predictions
from utils.calibration import compute_bias_and_qhat, apply_calibration
from models.base_models import train_base_models
from models.meta_model import train_meta_model_kfold

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_Xy(df, sheet):
    base_numeric = ['RATemp', 'MRT', 'Top', 'Air Velo', 'RH']
    base_cats = ['Season', 'Clothing', 'Activity']
    feats = [c for c in (base_numeric if sheet == "Classroom" else base_numeric + base_cats) if c in df.columns]
    target = "Given Final TSV"
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in sheet '{sheet}'")
    X = df[feats].copy()
    y = pd.to_numeric(df[target], errors='coerce')
    mask = ~y.isna()
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

def main():
    ensure_output()
    sheets = get_all_sheet_names('dataset/input_dataset.xlsx')
    summary_rows = []

    for sheet in sheets:
        print(f"\n=== Processing Environment: {sheet} ===")
        df = load_environment_sheet(sheet)
        X, y = extract_Xy(df, sheet)
        env_params = get_environment_params(sheet)

        oof_preds, base_results, selection_reports, selected_features = train_base_models(X, y, env_params)

        for m, rep in selection_reports.items():
            rep_path = os.path.join(OUTPUT_DIR, f"{sheet.replace(' ','_')}_{m}_selected_features.csv")
            rep.to_csv(rep_path, index=False)
            rep_sorted = rep.sort_values(by=['Selected','Combined_Score'], ascending=[False, False])
            top_used = rep_sorted.loc[rep_sorted['Selected']==True, 'Feature'].tolist()
            print(f"[Saved] {m} selected features → {rep_path}")
            print(f"Top used for {m}: {top_used}")

        y_series = pd.Series(y).reset_index(drop=True)

        if USE_STACKING:
            meta = train_meta_model_kfold(oof_preds, y_series, env_params)
            oof_meta = meta['oof_predictions']

            fi_path = os.path.join(OUTPUT_DIR, f"{sheet.replace(' ','_')}_meta_feature_importance.csv")
            meta['feature_importance'].to_csv(fi_path, index=False)
            lgb_params = env_params['lightgbm'].copy()
            if 'objective' not in lgb_params:
                lgb_params['objective'] = 'huber'
            meta_full = lgb.LGBMRegressor(**lgb_params)
            meta_full.fit(oof_preds, y_series)
            model_path = os.path.join(OUTPUT_DIR, f"{sheet.replace(' ','_')}_meta_model.joblib")
            dump(meta_full, model_path)
            y_pred = oof_meta
        else:
            best_m, best_r2 = None, -1e9
            for m in oof_preds.columns:
                r2 = evaluate_predictions(y_series, oof_preds[m])['R2']
                if r2 > best_r2:
                    best_r2, best_m = r2, m
            print(f"[No Stacking] Best single base model: {best_m} (R2={best_r2:.3f})")
            y_pred = oof_preds[best_m].values

        metrics = evaluate_predictions(y_series, y_pred)
        print("--- Final Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.2%}" if k == 'Accuracy' else f"{k}: {v:.4f}")

        if CALIBRATION_ON:
            bias, qhat = compute_bias_and_qhat(y_series, y_pred, alpha=0.10)
            y_cal = np.array([apply_calibration(p, bias, (-3, 3)) for p in y_pred])
            lower, upper = y_cal - qhat, y_cal + qhat
        else:
            y_cal, lower, upper = y_pred, np.full_like(y_pred, np.nan), np.full_like(y_pred, np.nan)

        pred_df = X.copy()
        pred_df['Given Final TSV']  = y_series
        pred_df['TSV_Predicted_raw'] = y_pred
        pred_df['TSV_Predicted']     = y_cal
        pred_df['TSV_Lower'] = lower
        pred_df['TSV_Upper'] = upper
        pred_path = os.path.join(OUTPUT_DIR, f"{sheet.replace(' ','_')}_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"[Saved] Predictions CSV → {pred_path}")

        base_rows = []
        for m, br in base_results.items():
            row = {'Model': m}; row.update(br); base_rows.append(row)
        base_summary = pd.DataFrame(base_rows)
        base_sum_path = os.path.join(OUTPUT_DIR, f"{sheet.replace(' ','_')}_base_summary.csv")
        base_summary.to_csv(base_sum_path, index=False)

        meta_path = os.path.join(OUTPUT_DIR, f"{sheet.replace(' ','_')}_metrics.csv")
        pd.DataFrame([metrics]).to_csv(meta_path, index=False)

        summary_rows.append({'Environment': sheet, **metrics})

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(OUTPUT_DIR, "predicted_tsv_results.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n=== Summary saved → {summary_path} ===")
        print(summary_df)

if __name__ == "__main__":
    main()
