# selective_features_pipeline/utils/metrics.py

from sklearn.metrics import mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Returns a dictionary of regression metrics: R² and RMSE.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    return {
        "r2_score": r2,
        "rmse": rmse
    }
