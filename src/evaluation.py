"""Reusable evaluation helpers, mirrored from the analysis notebook.

Kept separate from the notebook so the scoring logic can be reused
(e.g. in a future script or API) without re-running the full analysis.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.abs(y_pred)))


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    return {
        "Model": name,
        "Train R2": r2_score(y_train, train_preds),
        "Test R2": r2_score(y_test, test_preds),
        "Train MAE": mean_absolute_error(y_train, train_preds),
        "Test MAE": mean_absolute_error(y_test, test_preds),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, train_preds)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
        "Train RMSLE": rmsle(y_train, train_preds),
        "Test RMSLE": rmsle(y_test, test_preds),
    }
