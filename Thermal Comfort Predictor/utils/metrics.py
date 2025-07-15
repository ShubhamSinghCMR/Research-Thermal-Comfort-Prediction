from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "Explained Variance": explained_variance_score(y_true, y_pred)
    }

def error_bucket_stats(y_true, y_pred):
    abs_errors = np.abs(y_true - y_pred)
    return {
        "Within ±0.5": np.mean(abs_errors <= 0.5),
        "Within ±1.0": np.mean(abs_errors <= 1.0),
        "Within ±1.5": np.mean(abs_errors <= 1.5),
        "Max Error": np.max(abs_errors)
    }

def comfort_class(tsv_value):
    """Converts TSV to binary class: 1 = Comfortable, 0 = Uncomfortable"""
    return 1 if -1 <= tsv_value <= 1 else 0

def classification_metrics(y_true_tsv, y_pred_tsv):
    y_true_class = np.array([comfort_class(x) for x in y_true_tsv])
    y_pred_class = np.array([comfort_class(x) for x in y_pred_tsv])

    return {
        "Accuracy": accuracy_score(y_true_class, y_pred_class),
        "Precision": precision_score(y_true_class, y_pred_class),
        "Recall": recall_score(y_true_class, y_pred_class),
        "F1 Score": f1_score(y_true_class, y_pred_class),
        "Confusion Matrix": confusion_matrix(y_true_class, y_pred_class)
    }

def comfort_class_distribution(y_tsv):
    binary = np.array([comfort_class(x) for x in y_tsv])
    unique, counts = np.unique(binary, return_counts=True)
    return dict(zip(["Uncomfortable", "Comfortable"], counts))

def tsv_uncertainty_coverage(y_true, lower_bound, upper_bound):
    within_bounds = np.logical_and(y_true >= lower_bound, y_true <= upper_bound)
    return np.mean(within_bounds)

def tsv_interval_width(lower, upper):
    return np.mean(np.abs(upper - lower))

def plot_tsv_vs_true(y_true, y_pred, title="Predicted vs True TSV"):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([-3, 3], [-3, 3], 'r--')
    plt.xlabel("True TSV")
    plt.ylabel("Predicted TSV")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true_tsv, y_pred_tsv, labels=["Uncomfortable", "Comfortable"]):
    y_true_class = np.array([comfort_class(x) for x in y_true_tsv])
    y_pred_class = np.array([comfort_class(x) for x in y_pred_tsv])
    cm = confusion_matrix(y_true_class, y_pred_class)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def plot_residual_histogram(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, bins=20)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residual Error Distribution")
    plt.xlabel("Residuals (True - Predicted)")
    plt.tight_layout()
    plt.show()

def residual_bias_plot(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residual Bias Plot")
    plt.xlabel("Predicted TSV")
    plt.ylabel("Residual (True - Predicted)")
    plt.tight_layout()
    plt.show()
