
# Ensures LightGBM gets numpy arrays (fit/predict) to avoid feature-name mismatch warnings,
# and applies targeted warning filters when SHOW_WARNINGS=False.

import warnings
try:
    from sklearn.exceptions import ConvergenceWarning
except Exception:
    ConvergenceWarning = Warning

try:
    from utils.config import SHOW_WARNINGS
except Exception:
    SHOW_WARNINGS = False

if not SHOW_WARNINGS:
    try:
        warnings.filterwarnings(
            "ignore",
            message=r".*X does not have valid feature names, but LGBMRegressor was fitted with feature names.*",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            message=r".*Training interrupted by user.*",
            category=UserWarning,
        )
    except Exception:
        pass

# LightGBM IO patch
try:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    _orig_fit = lgb.LGBMRegressor.fit
    _orig_predict = lgb.LGBMRegressor.predict

    def _to_array(X):
        return X.values if isinstance(X, pd.DataFrame) else X

    def _fit_np(self, X, y=None, *args, **kwargs):
        return _orig_fit(self, _to_array(X), y, *args, **kwargs)

    def _predict_np(self, X, *args, **kwargs):
        return _orig_predict(self, _to_array(X), *args, **kwargs)

    lgb.LGBMRegressor.fit = _fit_np
    lgb.LGBMRegressor.predict = _predict_np
except Exception:
    pass
