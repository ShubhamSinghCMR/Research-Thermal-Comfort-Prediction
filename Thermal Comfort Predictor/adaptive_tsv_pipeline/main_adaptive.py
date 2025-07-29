"""
Final Main Script for Thermal Comfort Prediction System.
Enhancements:
- Saves predictions alongside Given Final TSV and all features (original unscaled values)
- Generates statistical summaries for each environment
- Creates visualizations (Feature Importance, Error Analysis, Correlation Heatmaps)
- Logs accuracy gap & overfitting warnings for model diagnosis
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other imports

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import seaborn as sns

from adaptive_tsv_pipeline.features.feature_engineering import load_and_preprocess_data, get_all_sheet_names
from adaptive_tsv_pipeline.utils.config import get_environment_params
from adaptive_tsv_pipeline.models.base_models import train_base_models
from adaptive_tsv_pipeline.models.meta_model import train_meta_model_kfold
from adaptive_tsv_pipeline.utils.metrics import evaluate_predictions

# Get the output directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

# ==================== VISUALIZATION FUNCTIONS ==================== #

def plot_feature_importance(importance_df, env_name):
    """Plot meta-model feature importance"""
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Meta-Model Feature Importance - {env_name}")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{env_name.replace(' ','_')}_meta_feature_importance.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[📊 Saved] Feature Importance Plot → {output_path}")


def plot_actual_vs_predicted(y_true, y_pred, env_name):
    """Scatter plot Actual vs Predicted TSV"""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Given Final TSV")
    plt.ylabel("TSV Predicted")
    plt.title(f"Actual vs Predicted TSV - {env_name}")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{env_name.replace(' ','_')}_actual_vs_predicted.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[📊 Saved] Actual vs Predicted Plot → {output_path}")


def plot_error_analysis(y_true, y_pred, env_name):
    """Error distribution plot with mean & median error lines"""
    errors = y_true - y_pred
    mean_error = errors.mean()
    median_error = np.median(errors)

    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error={mean_error:.2f}')
    plt.axvline(median_error, color='green', linestyle='dashed', linewidth=2, label=f'Median Error={median_error:.2f}')
    plt.xlabel("Error (Given Final TSV - TSV Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution - {env_name}")
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{env_name.replace(' ','_')}_error_analysis.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[📊 Saved] Error Analysis Plot → {output_path}")


def plot_correlation_heatmap(X, y, env_name):
    """Correlation heatmap for input features and TSV"""
    # Select only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    df_corr = X[numeric_cols].copy()
    df_corr["Given Final TSV"] = y.values
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Correlation Heatmap - {env_name}")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{env_name.replace(' ','_')}_correlation_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[📊 Saved] Correlation Heatmap → {output_path}")


def save_statistical_summary(X, y, env_name):
    """Save descriptive statistics summary to CSV"""
    summary_df = X.copy()
    summary_df["Given Final TSV"] = y.values
    stats = summary_df.describe().T
    stats["missing_values"] = summary_df.isna().sum()
    output_path = os.path.join(OUTPUT_DIR, f"{env_name.replace(' ','_')}_statistics_summary.csv")
    stats.to_csv(output_path)
    print(f"[📂 Saved] Statistical Summary CSV → {output_path}")


# ==================== MAIN PROCESS FUNCTIONS ==================== #

def print_status(message, status=""):
    """Colored status logging"""
    if status == "started":
        print(f"\n[🔄 Started] {message}")
    elif status == "completed":
        print(f"[✅ Completed] {message}")
    else:
        print(f"[ℹ️] {message}")


def run_environment(sheet_name):
    """
    Run training and evaluation for a single environment.
    """
    print_status(f"\n=== Processing Environment: {sheet_name} ===")

    # Load and preprocess
    X_scaled, y, X_original = load_and_preprocess_data(sheet_name)

    # 🔹 CHECK for missing required columns (Door, Window, Fan)
    required_columns = {"Door", "Window", "Fan"}
    if sheet_name == "Class room" or not required_columns.issubset(X_original.columns):
        print(f"[⚠️ Skipped] {sheet_name}: No Adaptive TSV is predicted due to missing data.")
        return None
    
    # Print input features
    print("\n--- Input Features ---")
    for idx, feature in enumerate(X_original.columns, 1):
        print(f"{idx}. {feature}")
    print("Target: Given Final TSV")
    print("-" * 20)
    
    print_status(f"Data Loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

    # Save statistics & correlation plots (use original)
    save_statistical_summary(X_original, y, sheet_name)
    plot_correlation_heatmap(X_original, y, sheet_name)

    # Get environment-specific parameters
    env_params = get_environment_params(sheet_name)

    # Train base models (OOF predictions)
    print_status("Training Base Models (K-Fold)", "started")
    oof_preds, base_results = train_base_models(X_scaled, y)
    print_status("Base Models Training Completed", "completed")

    # Accuracy gap warnings for base models
    print("\n--- Base Models Accuracy & Overfitting Check ---")
    for model_name, metrics in base_results.items():
        acc = metrics["Accuracy"]
        r2 = metrics["R2"]

        if acc < 0.4:
            print(f"[⚠️ Warning] {model_name} Accuracy is low ({acc:.2%}) → Potential underfitting")
        if r2 < 0.2:
            print(f"[⚠️ Warning] {model_name} R² is low ({r2:.3f}) → Predictions may be unstable")

        # Overfitting detection
        if "Train_Accuracy" in metrics:
            train_acc = metrics["Train_Accuracy"]
            acc_gap = train_acc - acc
            if acc_gap > 0.1:
                print(f"[⚠️ Warning] {model_name} Train-Valid Accuracy gap {acc_gap:.2%} → Possible overfitting")

    # Train meta-model (LightGBM)
    print_status("Training Meta-Model (LightGBM with K-Fold)", "started")
    meta_results = train_meta_model_kfold(oof_preds, y, env_params)
    print_status("Meta-Model Training Completed", "completed")

    # Evaluate meta-model using OOF predictions
    oof_meta_preds = meta_results["oof_predictions"]
    final_metrics = evaluate_predictions(y, oof_meta_preds)

    print("\n--- Final Meta-Model (LightGBM) Metrics ---")
    for k, v in final_metrics.items():
        if k == "Accuracy":
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.4f}")

    # Accuracy gap warnings for meta-model
    if final_metrics["Accuracy"] < 0.4:
        print(f"[⚠️ Warning] Meta-Model Accuracy is low ({final_metrics['Accuracy']:.2%}) → Potential underfitting")
    if final_metrics["R2"] < 0.3:
        print(f"[⚠️ Warning] Meta-Model R² is low ({final_metrics['R2']:.3f}) → Predictions may not generalize well")

    # Save predictions per environment
    predictions_df = X_original.copy()
    predictions_df["Given Final TSV"] = y.values
    predictions_df["TSV_Predicted"] = oof_meta_preds
    output_file = os.path.join(OUTPUT_DIR, f"{sheet_name.replace(' ', '_')}_predictions.csv")
    try:
        predictions_df.to_csv(output_file, index=False)
        print(f"[📂 Saved] Predictions for {sheet_name} → {output_file}")
    except PermissionError:
        print(f"[⚠️ Error] Could not save predictions to {output_file} - File is open in another program. Please close it and try again.")
        return None
    except Exception as e:
        print(f"[⚠️ Error] Failed to save predictions: {str(e)}")
        return None

    # Visualizations
    plot_feature_importance(meta_results["feature_importance"], sheet_name)
    plot_actual_vs_predicted(y, oof_meta_preds, sheet_name)
    plot_error_analysis(y, oof_meta_preds, sheet_name)

    return {
        "base_results": base_results,
        "meta_results": meta_results,
        "final_metrics": final_metrics,
    }


def main():
    """Main function"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sheet_names = get_all_sheet_names()
    all_results = {}
    failed_environments = []

    for sheet_name in sheet_names:
        results = run_environment(sheet_name)
        if results:  # Only add to all_results if run_environment returned a valid result
            all_results[sheet_name] = results
        else:
            failed_environments.append(sheet_name)

    if not all_results:
        print("\n[❌ Error] No environments were processed successfully.")
        if failed_environments:
            print("Failed environments:")
            for env in failed_environments:
                print(f"- {env}")
        return

    # Summary CSV
    summary_data = []
    for env_name, res in all_results.items():
        metrics = res["final_metrics"]
        summary_data.append({
            "Environment": env_name,
            **metrics
        })

    summary_df = pd.DataFrame(summary_data)
    try:
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "adaptive_tsv_results.csv"), index=False)
        print("\n=== Adaptive TSV Results Saved to output/adaptive_tsv_results.csv ===")
        print(summary_df)
    except PermissionError:
        print("\n[⚠️ Error] Could not save final results - File is open in another program.")
        print("Results summary:")
        print(summary_df)
    except Exception as e:
        print(f"\n[⚠️ Error] Failed to save final results: {str(e)}")
        print("Results summary:")
        print(summary_df)

    if failed_environments:
        print("\n[⚠️ Warning] Some environments failed to process:")
        for env in failed_environments:
            print(f"- {env}")


if __name__ == "__main__":
    main()
