"""
NaN Handling Examples

This file demonstrates the NaN handling strategies used in the GC-Ridge pipeline.
These examples show how NaN values are handled at different stages of the pipeline.
"""

import numpy as np
from scipy.stats import spearmanr, rankdata


def graph_convolution_nan_handling(X, adjs):
    """
    Example: NaN handling in graph convolution layer.
    
    This demonstrates how NaN values are handled before graph convolution
    to prevent NaN propagation through the network.
    
    Args:
        X: Input features (n_samples, n_nodes, n_features) - may contain NaN
        adjs: Adjacency matrices for graph convolution
    
    Returns:
        Convolved features (n_samples, n_nodes, hidden_dim) - no NaN
    """
    n, nodes, feats = X.shape
    
    # Convert NaN to zero before convolution
    # This prevents NaN propagation while preserving structure
    Xc = np.nan_to_num(X, nan=0.0)
    
    # Graph convolution proceeds with NaN-free features
    out = np.zeros((n, nodes, 64))  # hidden_dim=64
    
    # ... graph convolution logic ...
    
    return out


def target_level_nan_filtering(y_train, features_train):
    """
    Example: Filtering NaN targets during training.
    
    This demonstrates how we filter out samples with NaN targets
    before training to ensure model quality.
    
    Args:
        y_train: Target values (n_samples, n_targets) - may contain NaN
        features_train: Features (n_samples, n_features)
    
    Returns:
        Filtered data with only valid samples
    """
    # Example for a single target
    target_idx = 0
    yt = y_train[:, target_idx]
    
    # Create mask for valid (non-NaN) samples
    valid = ~np.isnan(yt)
    
    # Require minimum number of valid samples
    if valid.sum() < 50:
        return None, None  # Skip target if insufficient data
    
    # Filter to valid samples only
    yv = yt[valid]
    Fv = features_train[valid]
    
    return Fv, yv


def nan_aware_correlation(y_true, y_pred):
    """
    Example: Computing correlation while handling NaN values.
    
    This demonstrates how correlations are computed only on valid pairs,
    preventing NaN contamination in evaluation metrics.
    
    Args:
        y_true: Ground truth (n_samples, n_targets) - may contain NaN
        y_pred: Predictions (n_samples, n_targets) - may contain NaN
    
    Returns:
        Correlation coefficient (ignoring NaN pairs)
    """
    # Example for a single target
    target_idx = 0
    a = y_true[:, target_idx]
    b = y_pred[:, target_idx]
    
    # Create mask for valid (non-NaN) pairs
    mask = ~(np.isnan(a) | np.isnan(b))
    
    # Require minimum number of valid pairs
    if np.sum(mask) < 20:
        return 0.0  # Insufficient data
    
    # Compute correlation only on valid pairs
    c, _ = spearmanr(a[mask], b[mask])
    
    # Handle NaN correlation result
    return 0.0 if np.isnan(c) else float(c)


def prediction_initialization_with_nan(n_samples, n_targets):
    """
    Example: Initializing predictions with NaN to preserve missing data structure.
    
    This demonstrates how predictions are initialized with NaN and only
    filled for targets with trained models.
    
    Args:
        n_samples: Number of samples
        n_targets: Number of targets
    
    Returns:
        Predictions array initialized with NaN
    """
    # Initialize with NaN (preserves missing data structure)
    y_pred = np.full((n_samples, n_targets), np.nan)
    
    # Only fill predictions for targets with trained models
    # Example: fill prediction for target 0
    target_idx = 0
    if target_idx in trained_models:  # Assuming trained_models dict exists
        y_pred[:, target_idx] = trained_models[target_idx].predict(features)
    
    return y_pred


if __name__ == '__main__':
    print("NaN Handling Examples")
    print("=" * 50)
    print("These examples demonstrate NaN handling strategies used in the pipeline.")
    print("See TECHNICAL_CHALLENGES.md for detailed explanations.")

