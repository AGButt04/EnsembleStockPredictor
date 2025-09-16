import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def simple_ensemble(predictions_list):
    """Create ensemble by averaging predictions"""
    return np.mean(predictions_list, axis=0)


def weighted_ensemble(predictions_list, weights):
    """Create weighted ensemble of predictions"""
    return np.average(predictions_list, axis=0, weights=weights)


def evaluate_ensemble(y_true, ensemble_pred):
    """Evaluate ensemble performance"""
    mse = mean_squared_error(y_true, ensemble_pred)
    r2 = r2_score(y_true, ensemble_pred)
    rmse = np.sqrt(mse)

    return {
        'mse': mse,
        'r2': r2,
        'rmse': rmse
    }