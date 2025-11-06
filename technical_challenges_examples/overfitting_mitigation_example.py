"""
Temporal Overfitting Mitigation Examples

This file demonstrates strategies used to mitigate temporal overfitting
in the GC-Ridge pipeline.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler


def ridge_regularization(features_train, y_train):
    """
    Example: Ridge regularization to prevent overfitting.
    
    This demonstrates how L2 regularization penalizes large coefficients,
    preventing the model from overfitting to specific temporal patterns.
    
    Args:
        features_train: Training features (n_train, n_features)
        y_train: Training targets (n_train,)
    
    Returns:
        Trained Ridge model with regularization
    """
    # Ridge regression with L2 regularization
    # Alpha=0.05 penalizes large coefficients
    model = Ridge(alpha=0.05)
    model.fit(features_train, y_train)
    
    return model


def feature_selection_complexity_reduction(features_train, y_train):
    """
    Example: Feature selection to reduce model complexity.
    
    This demonstrates how selecting top K features prevents overfitting
    by reducing model complexity.
    
    Args:
        features_train: Training features (n_train, n_features)
        y_train: Training targets (n_train,)
    
    Returns:
        Selected features and selector
    """
    # Select top K features to prevent overfitting
    k = min(30, features_train.shape[1])
    selector = SelectKBest(f_regression, k=k)
    selector.fit(features_train, y_train)
    
    selected_indices = selector.get_support(indices=True)
    features_selected = features_train[:, selected_indices]
    
    return features_selected, selector


def minimum_sample_requirement(y_train):
    """
    Example: Minimum sample requirement to prevent overfitting.
    
    This demonstrates how requiring a minimum number of training samples
    ensures statistical validity and prevents overfitting on sparse targets.
    
    Args:
        y_train: Training targets (n_samples,)
    
    Returns:
        Whether target has sufficient samples
    """
    # Filter out NaN samples
    valid = ~np.isnan(y_train)
    n_valid = valid.sum()
    
    # Require minimum 50 valid samples
    if n_valid < 50:
        return False  # Insufficient data - skip target
    
    return True  # Sufficient data - proceed with training


def temporal_validation_window(y, val_start=1829, val_end=1950):
    """
    Example: Fixed temporal validation window for consistent evaluation.
    
    This demonstrates how using a fixed validation window allows
    consistent evaluation and detection of temporal overfitting.
    
    Args:
        y: All target data (n_samples, n_targets)
        val_start: Validation start date
        val_end: Validation end date
    
    Returns:
        Validation subset
    """
    # Fixed validation window
    y_val = y[val_start:val_end]
    
    # This fixed window allows:
    # 1. Consistent evaluation across experiments
    # 2. Detection of temporal overfitting
    # 3. Comparison with other models
    
    return y_val


def destroyer_neutralization_as_regularization(y_true, y_pred, destroyers):
    """
    Example: Destroyer neutralization as a form of regularization.
    
    This demonstrates how replacing unstable predictions with stable baselines
    acts as regularization, reducing temporal variance.
    
    Args:
        y_true: Ground truth (n_days, n_targets)
        y_pred: Predictions (n_days, n_targets)
        destroyers: List of destroyer target indices
    
    Returns:
        Neutralized predictions (more stable)
    """
    from scipy.stats import rankdata
    
    y_pred_new = y_pred.copy()
    n_days, n_tgt = y_pred.shape
    
    for j in destroyers:
        # Replace predictions with prior-day ranks (stable baseline)
        baseline = np.zeros((n_days,), dtype=float)
        series = y_true[:, j]
        
        for t in range(n_days):
            # Use ground truth if available, else use predictions
            ref = y_true[t, :] if not np.isnan(series[t]) else y_pred[t, :]
            ranks = rankdata(ref, method='average') / n_tgt
            
            # Set baseline to prior-day rank
            if t == 0:
                baseline[t] = ranks[j]
            else:
                baseline[t] = prev_rank
            
            prev_rank = ranks[j]
        
        # Replace unstable predictions with stable baseline
        y_pred_new[:, j] = baseline
    
    return y_pred_new


if __name__ == '__main__':
    print("Temporal Overfitting Mitigation Examples")
    print("=" * 50)
    print("These examples demonstrate strategies to mitigate overfitting.")
    print("See TECHNICAL_CHALLENGES.md for detailed explanations.")

