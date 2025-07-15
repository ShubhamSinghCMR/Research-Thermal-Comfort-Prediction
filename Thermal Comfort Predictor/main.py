import pandas as pd
import os
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

from features.feature_engineering import engineer_features
from models.base_models import train_base_models
from models.meta_model import prepare_meta_features, train_meta_model
from models.rule_correction import apply_rule_correction_batch
from utils.metrics import regression_metrics, classification_metrics, tsv_uncertainty_coverage
from utils.config import MODEL_DIR, SEED, TSV_COMFORT_RANGE, TRAIN_SIZE_PERCENT, TEST_SIZE_PERCENT
from utils.thermal_comfort import estimate_temperature_from_features, comfort_range_to_temperature
from utils.visualizations import generate_all_visualizations

def main():
    logging.info('Pipeline started.')

    # Load and prepare data
    logging.info('Loading and preparing data...')
    data_path = os.path.join("dataset", "input_dataset.csv")
    df_raw = pd.read_csv(data_path)
    df = engineer_features(df_raw.copy())
    df = df.dropna(subset=['TSV'])
    logging.info('Data loaded and features engineered.')

    # Log the train/test split configuration
    logging.info(f'Using train/test split: {TRAIN_SIZE_PERCENT}%/{TEST_SIZE_PERCENT}%')

    # Prepare features and targets
    feature_cols = df.columns.difference(['TSV'])
    X = df[feature_cols]
    X = X.select_dtypes(include=[np.number])  # Keep only numeric columns for modeling
    X = X.fillna(0)  # Fill any remaining NaN values with 0
    y_tsv = df['TSV']
    
    # Estimate actual temperatures from TSV for training (this will be our new target)
    logging.info('Estimating temperatures from TSV for training...')
    estimated_temps = estimate_temperature_from_features(df, y_tsv.values)
    y_temp = pd.Series(estimated_temps, index=y_tsv.index)
    
    logging.info('Features and targets prepared.')

    # Train base models and meta-learners
    logging.info('Training base models...')
    base_preds, X_test_base, y_tsv_true, y_temp_true = train_base_models(X, y_tsv, y_temp)

    logging.info('Preparing meta features...')
    meta_X = prepare_meta_features(base_preds, X.loc[X_test_base.index])
    logging.info('Meta features prepared.')

    logging.info('Training meta model...')
    meta_outputs = train_meta_model(meta_X, y_tsv_true, y_temp_true)

    # Combine predictions and apply corrections
    meta_df = meta_X.copy()
    logging.info('Combining predictions and applying rule corrections...')
    test_meta_df = pd.DataFrame({
        'TSV_meta': np.array(meta_outputs["TSV_meta"]),
        'Temp_meta': np.array(meta_outputs["Temp_meta"]),
        'y_tsv_true': np.array(meta_outputs["y_tsv_true"]),
        'y_temp_true': np.array(meta_outputs["y_temp_true"])
    })
    test_meta_df['tsv_qrf_lower'] = base_preds["TSV"]["qrf_lower"]
    test_meta_df['tsv_qrf_upper'] = base_preds["TSV"]["qrf_upper"]
    meta_df.update(test_meta_df)
    test_meta_df = apply_rule_correction_batch(test_meta_df, tsv_col='TSV_meta')
    meta_df.update(test_meta_df[['TSV_final']])
    logging.info('Rule corrections applied.')

    # Convert TSV predictions to temperature estimates
    logging.info('Converting TSV predictions to temperature estimates...')
    test_features = X.loc[X_test_base.index].reset_index(drop=True)
    # No need to align with meta_outputs['test_indices'] since we use all base test data
    temp_estimates = estimate_temperature_from_features(test_features, test_meta_df['TSV_final'].values)
    test_meta_df['Temp_estimated'] = temp_estimates
    
    # Evaluate performance (only on rows with meta predictions)
    logging.info('Evaluating model performance...')
    eval_df = test_meta_df
    
    # Add human-readable output columns
    def tsv_to_comfort_level(tsv_value):
        """Convert TSV value to human-readable comfort level"""
        if tsv_value <= -2.5:
            return "Very Cold"
        elif tsv_value <= -1.5:
            return "Cold"
        elif tsv_value <= -0.5:
            return "Cool"
        elif tsv_value <= 0.5:
            return "Comfortable"
        elif tsv_value <= 1.5:
            return "Warm"
        elif tsv_value <= 2.5:
            return "Hot"
        else:
            return "Very Hot"
    
    eval_df['OutputTSV'] = eval_df['TSV_final'].apply(tsv_to_comfort_level)
    eval_df['OutputTemp_Celsius'] = eval_df['Temp_estimated'].round(1)  # Temperature in °C
    
    # Add comfort range information
    eval_df['ComfortRange'] = eval_df['OutputTSV'].apply(lambda x: comfort_range_to_temperature(x))
    eval_df['TypicalTemp'] = eval_df['ComfortRange'].apply(lambda x: x['typical'])
    eval_df['TempRange'] = eval_df['ComfortRange'].apply(lambda x: f"{x['min']}-{x['max']}°C")
    
    eval_df_clean = eval_df.dropna()
    if len(eval_df_clean) == 0:
        logging.error('No valid predictions after removing NaN values. Check data alignment.')
        return
    logging.info(f"Evaluating on {len(eval_df_clean)} samples (dropped {len(eval_df) - len(eval_df_clean)} samples with NaN values)")
    tsv_scores = regression_metrics(eval_df_clean['y_tsv_true'], eval_df_clean['TSV_final'])
    class_scores = classification_metrics(eval_df_clean['y_tsv_true'], eval_df_clean['TSV_final'])
    temp_scores = regression_metrics(eval_df_clean['y_temp_true'], eval_df_clean['Temp_estimated'])
    coverage = tsv_uncertainty_coverage(eval_df_clean['y_tsv_true'], eval_df_clean['tsv_qrf_lower'], eval_df_clean['tsv_qrf_upper'])
    logging.info('Evaluation completed.')

    # Generate visualizations
    logging.info('Generating performance visualizations...')
    graph_files = generate_all_visualizations(eval_df_clean, "output/")
    logging.info('Visualizations completed.')

    # Print results
    logging.info(f"TSV Regression: {tsv_scores}")
    logging.info(f"Comfort Classification: {class_scores}")
    logging.info(f"Temperature Regression: {temp_scores}")
    logging.info(f"TSV Quantile Coverage: {coverage:.2%}")

    # Save results
    logging.info('Saving results...')
    os.makedirs("output", exist_ok=True)
    
    with open("output/metrics_report.txt", "w") as f:
        f.write("TSV Regression Metrics:\n")
        for k, v in tsv_scores.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nComfort Classification Metrics:\n")
        for k, v in class_scores.items():
            if k == "Confusion Matrix":
                f.write("Confusion Matrix:\n")
                f.write(str(v) + "\n")
            else:
                f.write(f"{k}: {v:.4f}\n")
        f.write("\nTemperature Regression Metrics (Degrees Celsius):\n")
        for k, v in temp_scores.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"\nTSV Quantile Coverage: {coverage:.2%}\n")
        
        # Add temperature statistics
        f.write(f"\nTemperature Statistics:\n")
        f.write(f"Average Predicted Temperature: {eval_df_clean['Temp_estimated'].mean():.1f}°C\n")
        f.write(f"Temperature Range: {eval_df_clean['Temp_estimated'].min():.1f}°C to {eval_df_clean['Temp_estimated'].max():.1f}°C\n")
        f.write(f"Temperature Standard Deviation: {eval_df_clean['Temp_estimated'].std():.1f}°C\n")
    
    logging.info('Results saved to output/.')
    logging.info('Pipeline completed.')

    # --- NEW: Output full input data with predictions for test set rows ---
    df_full = df_raw.copy()
    # Prepare empty columns
    df_full['Predicted TSV Value'] = ''
    df_full['Predicted comfort'] = ''
    df_full['Predicted Temperature (degrees Celsius)'] = ''
    # Map predictions to the correct rows
    # X_test_base.index gives the indices in df of the base model test set
    # Now we use ALL base test indices since meta model uses all of them
    base_test_indices = X_test_base.index
    # Fill in predictions for these rows
    df_full.loc[base_test_indices, 'Predicted TSV Value'] = test_meta_df['TSV_final'].values
    df_full.loc[base_test_indices, 'Predicted comfort'] = test_meta_df['OutputTSV'].values
    df_full.loc[base_test_indices, 'Predicted Temperature (degrees Celsius)'] = test_meta_df['Temp_estimated'].round(1).values
    # Save
    df_full.to_csv('output/complete_dataset_with_predictions.csv', index=False)
    logging.info('Complete dataset with predictions saved to output/complete_dataset_with_predictions.csv')

    # --- NEW: Create results.csv with only predicted rows and specific columns ---
    # Get only rows that have predictions (non-empty Predicted TSV Value)
    results_df = df_full[df_full['Predicted TSV Value'] != ''].copy()
    
    # Select only the specified columns
    selected_columns = [
        'Sr .No.', 'Surveyr', 'Building Name', 'Location', 'Condition Type', 
        'Occupant Name', 'Occupant Designation', 'Predicted TSV Value', 
        'Predicted comfort', 'Predicted Temperature (degrees Celsius)'
    ]
    
    # Filter to only include columns that exist in the dataset
    existing_columns = [col for col in selected_columns if col in results_df.columns]
    results_df = results_df[existing_columns]
    
    # Save the results file
    results_df.to_csv('output/results.csv', index=False)
    logging.info(f'Results file saved to output/results.csv with {len(results_df)} predicted rows')

if __name__ == "__main__":
    main()
