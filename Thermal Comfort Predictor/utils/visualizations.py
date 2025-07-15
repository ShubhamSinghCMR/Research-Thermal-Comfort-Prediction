import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def create_performance_visualizations(eval_df_clean, output_dir="output/"):
    """
    Creates comprehensive visualizations for model performance analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting parameters
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. TSV Prediction vs True Values
    create_prediction_scatter_plot(eval_df_clean, output_dir)
    
    # 2. Temperature Prediction vs True Values  
    create_temperature_scatter_plot(eval_df_clean, output_dir)
    
    # 3. Residual Analysis
    create_residual_plots(eval_df_clean, output_dir)
    
    # 4. Confusion Matrix for Comfort Classification
    create_confusion_matrix_plot(eval_df_clean, output_dir)
    
    # 5. Distribution Plots
    create_distribution_plots(eval_df_clean, output_dir)
    
    # 6. Uncertainty Analysis
    create_uncertainty_plots(eval_df_clean, output_dir)
    
    # 7. Comfort Level Analysis
    create_comfort_analysis_plots(eval_df_clean, output_dir)
    
    # 8. Model Performance Summary
    create_performance_summary(eval_df_clean, output_dir)
    
    print(f"All visualization graphs saved to {output_dir}")

def create_prediction_scatter_plot(df, output_dir):
    """TSV Predictions vs True Values with perfect prediction line"""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(df['y_tsv_true'], df['TSV_final'], alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(df['y_tsv_true'].min(), df['TSV_final'].min())
    max_val = max(df['y_tsv_true'].max(), df['TSV_final'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(df['y_tsv_true'], df['TSV_final'])
    
    plt.xlabel('True TSV Values', fontsize=12)
    plt.ylabel('Predicted TSV Values', fontsize=12)
    plt.title(f'TSV Predictions vs True Values\nR² = {r2:.3f}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotation
    plt.text(0.05, 0.95, f'Number of samples: {len(df)}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tsv_predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_temperature_scatter_plot(df, output_dir):
    """Temperature Predictions vs True Values"""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(df['y_temp_true'], df['Temp_estimated'], alpha=0.6, s=50, color='orange')
    
    # Perfect prediction line
    min_val = min(df['y_temp_true'].min(), df['Temp_estimated'].min())
    max_val = max(df['y_temp_true'].max(), df['Temp_estimated'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(df['y_temp_true'], df['Temp_estimated'])
    
    plt.xlabel('True Temperature (°C)', fontsize=12)
    plt.ylabel('Predicted Temperature (°C)', fontsize=12)
    plt.title(f'Temperature Predictions vs True Values\nR² = {r2:.3f}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotation
    plt.text(0.05, 0.95, f'Number of samples: {len(df)}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temperature_predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_residual_plots(df, output_dir):
    """Residual analysis plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # TSV Residuals vs Predicted
    tsv_residuals = df['y_tsv_true'] - df['TSV_final']
    ax1.scatter(df['TSV_final'], tsv_residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted TSV')
    ax1.set_ylabel('Residuals (True - Predicted)')
    ax1.set_title('TSV Residuals vs Predicted Values')
    ax1.grid(True, alpha=0.3)
    
    # Temperature Residuals vs Predicted
    temp_residuals = df['y_temp_true'] - df['Temp_estimated']
    ax2.scatter(df['Temp_estimated'], temp_residuals, alpha=0.6, color='orange')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Temperature (°C)')
    ax2.set_ylabel('Residuals (True - Predicted)')
    ax2.set_title('Temperature Residuals vs Predicted Values')
    ax2.grid(True, alpha=0.3)
    
    # TSV Residuals Distribution
    ax3.hist(tsv_residuals, bins=20, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('TSV Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of TSV Residuals')
    ax3.axvline(x=0, color='r', linestyle='--')
    
    # Temperature Residuals Distribution
    ax4.hist(temp_residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Temperature Residuals (°C)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Temperature Residuals')
    ax4.axvline(x=0, color='r', linestyle='--')
    
    plt.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_plot(df, output_dir):
    """Confusion matrix for comfort classification"""
    # Convert TSV to comfort classes (binary: comfortable vs uncomfortable)
    y_true_comfort = (df['y_tsv_true'].between(-1, 1)).astype(int)
    y_pred_comfort = (df['TSV_final'].between(-1, 1)).astype(int)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_comfort, y_pred_comfort)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Uncomfortable', 'Comfortable'],
                yticklabels=['Uncomfortable', 'Comfortable'])
    plt.title('Comfort Classification Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', transform=plt.gca().transAxes, 
             ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_plots(df, output_dir):
    """Distribution comparison plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # TSV Distributions
    ax1.hist(df['y_tsv_true'], bins=20, alpha=0.7, label='True TSV', color='blue')
    ax1.hist(df['TSV_final'], bins=20, alpha=0.7, label='Predicted TSV', color='red')
    ax1.set_xlabel('TSV Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('TSV Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Temperature Distributions
    ax2.hist(df['y_temp_true'], bins=20, alpha=0.7, label='True Temperature', color='blue')
    ax2.hist(df['Temp_estimated'], bins=20, alpha=0.7, label='Predicted Temperature', color='orange')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Temperature Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Box plots for TSV
    box_data_tsv = [df['y_tsv_true'], df['TSV_final']]
    ax3.boxplot(box_data_tsv, labels=['True TSV', 'Predicted TSV'])
    ax3.set_ylabel('TSV Value')
    ax3.set_title('TSV Box Plot Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Box plots for Temperature
    box_data_temp = [df['y_temp_true'], df['Temp_estimated']]
    ax4.boxplot(box_data_temp, labels=['True Temperature', 'Predicted Temperature'])
    ax4.set_ylabel('Temperature (°C)')
    ax4.set_title('Temperature Box Plot Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_uncertainty_plots(df, output_dir):
    """Uncertainty quantification plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prediction intervals plot
    indices = np.arange(len(df))
    ax1.fill_between(indices, df['tsv_qrf_lower'], df['tsv_qrf_upper'], 
                     alpha=0.3, label='90% Prediction Interval')
    ax1.scatter(indices, df['y_tsv_true'], alpha=0.7, s=20, color='red', label='True TSV')
    ax1.scatter(indices, df['TSV_final'], alpha=0.7, s=20, color='blue', label='Predicted TSV')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('TSV Value')
    ax1.set_title('Prediction Intervals vs True/Predicted Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Coverage analysis
    in_interval = ((df['y_tsv_true'] >= df['tsv_qrf_lower']) & 
                   (df['y_tsv_true'] <= df['tsv_qrf_upper']))
    coverage = in_interval.mean()
    
    ax2.bar(['In Interval', 'Out of Interval'], 
            [coverage * 100, (1 - coverage) * 100],
            color=['green', 'red'], alpha=0.7)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title(f'Prediction Interval Coverage\n{coverage:.1%} of predictions within interval')
    ax2.grid(True, alpha=0.3)
    
    # Add coverage text
    ax2.text(0.5, 0.9, f'Target: 90%\nActual: {coverage:.1%}', 
             transform=ax2.transAxes, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comfort_analysis_plots(df, output_dir):
    """Comfort level analysis"""
    # Map TSV to comfort levels
    def tsv_to_comfort_level(tsv):
        if tsv <= -2.5: return "Very Cold"
        elif tsv <= -1.5: return "Cold"
        elif tsv <= -0.5: return "Cool"
        elif tsv <= 0.5: return "Comfortable"
        elif tsv <= 1.5: return "Warm"
        elif tsv <= 2.5: return "Hot"
        else: return "Very Hot"
    
    df_comfort = df.copy()
    df_comfort['True_Comfort'] = df_comfort['y_tsv_true'].apply(tsv_to_comfort_level)
    df_comfort['Pred_Comfort'] = df_comfort['TSV_final'].apply(tsv_to_comfort_level)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get all unique comfort levels and create aligned counts
    all_comfort_levels = sorted(set(df_comfort['True_Comfort'].unique()) | set(df_comfort['Pred_Comfort'].unique()))
    
    # Create aligned counts for true and predicted
    comfort_counts_true = df_comfort['True_Comfort'].value_counts().reindex(all_comfort_levels, fill_value=0)
    comfort_counts_pred = df_comfort['Pred_Comfort'].value_counts().reindex(all_comfort_levels, fill_value=0)
    
    x_pos = np.arange(len(all_comfort_levels))
    width = 0.35
    
    ax1.bar(x_pos - width/2, comfort_counts_true.values, width, 
            label='True', alpha=0.7, color='blue')
    ax1.bar(x_pos + width/2, comfort_counts_pred.values, width,
            label='Predicted', alpha=0.7, color='red')
    
    ax1.set_xlabel('Comfort Level')
    ax1.set_ylabel('Count')
    ax1.set_title('Comfort Level Distribution Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(all_comfort_levels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Temperature vs Comfort Level
    comfort_order = ["Very Cold", "Cold", "Cool", "Comfortable", "Warm", "Hot", "Very Hot"]
    temp_by_comfort = []
    labels = []
    
    for comfort in comfort_order:
        comfort_data = df_comfort[df_comfort['True_Comfort'] == comfort]['Temp_estimated']
        if len(comfort_data) > 0:
            temp_by_comfort.append(comfort_data)
            labels.append(comfort)
    
    if temp_by_comfort:
        ax2.boxplot(temp_by_comfort, labels=labels)
        ax2.set_xlabel('Comfort Level')
        ax2.set_ylabel('Predicted Temperature (°C)')
        ax2.set_title('Temperature Distribution by Comfort Level')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comfort_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_summary(df, output_dir):
    """Performance summary dashboard"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # TSV Metrics
    tsv_mae = mean_absolute_error(df['y_tsv_true'], df['TSV_final'])
    tsv_rmse = np.sqrt(mean_squared_error(df['y_tsv_true'], df['TSV_final']))
    tsv_r2 = r2_score(df['y_tsv_true'], df['TSV_final'])
    
    # Temperature Metrics
    temp_mae = mean_absolute_error(df['y_temp_true'], df['Temp_estimated'])
    temp_rmse = np.sqrt(mean_squared_error(df['y_temp_true'], df['Temp_estimated']))
    temp_r2 = r2_score(df['y_temp_true'], df['Temp_estimated'])
    
    # Comfort Classification Metrics
    y_true_comfort = (df['y_tsv_true'].between(-1, 1)).astype(int)
    y_pred_comfort = (df['TSV_final'].between(-1, 1)).astype(int)
    comfort_accuracy = (y_true_comfort == y_pred_comfort).mean()
    
    # Coverage
    in_interval = ((df['y_tsv_true'] >= df['tsv_qrf_lower']) & 
                   (df['y_tsv_true'] <= df['tsv_qrf_upper']))
    coverage = in_interval.mean()
    
    # Create metric cards
    metrics_data = [
        ('TSV R²', f'{tsv_r2:.3f}', 'Higher is better'),
        ('TSV MAE', f'{tsv_mae:.3f}', 'Lower is better'),
        ('TSV RMSE', f'{tsv_rmse:.3f}', 'Lower is better'),
        ('Temp R²', f'{temp_r2:.3f}', 'Higher is better'),
        ('Temp MAE', f'{temp_mae:.3f}°C', 'Lower is better'),
        ('Temp RMSE', f'{temp_rmse:.3f}°C', 'Lower is better'),
        ('Comfort Accuracy', f'{comfort_accuracy:.3f}', 'Higher is better'),
        ('Coverage', f'{coverage:.1%}', 'Target: 90%')
    ]
    
    colors = ['lightblue', 'lightcoral', 'lightcoral', 'lightgreen', 
              'lightyellow', 'lightyellow', 'lightpink', 'lightgray']
    
    for i, ((metric, value, note), color) in enumerate(zip(metrics_data, colors)):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        
        ax.text(0.5, 0.7, metric, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.4, value, ha='center', va='center', fontsize=20, fontweight='bold')
        ax.text(0.5, 0.1, note, ha='center', va='center', fontsize=8, style='italic')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, 
                                  facecolor=color, alpha=0.3, linewidth=2, edgecolor='black'))
    
    # Add title
    fig.suptitle('Model Performance Summary Dashboard', fontsize=20, fontweight='bold', y=0.95)
    
    # Add sample info
    fig.text(0.5, 0.02, f'Based on {len(df)} test samples', ha='center', fontsize=12)
    
    plt.savefig(f'{output_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_visualizations(eval_df_clean, output_dir="output/"):
    """Main function to generate all visualizations"""
    print("🎨 Generating performance visualizations...")
    create_performance_visualizations(eval_df_clean, output_dir)
    print("✅ All visualizations completed!")
    
    return [
        f"{output_dir}/tsv_predictions_scatter.png",
        f"{output_dir}/temperature_predictions_scatter.png", 
        f"{output_dir}/residual_analysis.png",
        f"{output_dir}/confusion_matrix.png",
        f"{output_dir}/distribution_analysis.png",
        f"{output_dir}/uncertainty_analysis.png",
        f"{output_dir}/comfort_analysis.png",
        f"{output_dir}/performance_summary.png"
    ] 