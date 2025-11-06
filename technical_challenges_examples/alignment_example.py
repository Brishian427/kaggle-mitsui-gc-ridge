"""
Data Alignment Examples

This file demonstrates how data alignment is ensured throughout the pipeline.
These examples show explicit index management and shape verification.
"""

import numpy as np
import pandas as pd


def explicit_temporal_indexing(X, y, masks):
    """
    Example: Explicit temporal indexing for alignment.
    
    This demonstrates how explicit array slicing based on date indices
    ensures proper alignment between features, targets, and masks.
    
    Args:
        X: Features (n_samples, n_nodes, n_features)
        y: Targets (n_samples, n_targets)
        masks: Trading masks (n_samples, n_nodes)
    
    Returns:
        Aligned training and validation sets
    """
    # Explicit temporal split using date indices
    train_end = 1828
    val_start, val_end = 1829, 1950
    
    # Explicit array slicing (preserves order and alignment)
    X_train = X[:train_end]  # Days 0 to 1828
    X_val = X[val_start:val_end]  # Days 1829 to 1950
    y_train = y[:train_end]
    y_val = y[val_start:val_end]
    masks_train = masks[:train_end]
    masks_val = masks[val_start:val_end]
    
    # Verify alignment
    assert X_train.shape[0] == y_train.shape[0] == masks_train.shape[0], \
        "Training data misaligned!"
    assert X_val.shape[0] == y_val.shape[0] == masks_val.shape[0], \
        "Validation data misaligned!"
    
    return X_train, X_val, y_train, y_val, masks_train, masks_val


def prediction_shape_verification(y_val, y_pred_val):
    """
    Example: Shape verification for predictions.
    
    This demonstrates how shape assertions catch alignment errors early.
    
    Args:
        y_val: Validation ground truth (n_val, n_targets)
        y_pred_val: Validation predictions (n_val, n_targets)
    
    Returns:
        Verified predictions
    """
    # Verify shapes match exactly
    assert y_pred_val.shape == y_val.shape, \
        f"Shape mismatch: predictions {y_pred_val.shape} vs ground truth {y_val.shape}"
    
    # Verify number of samples matches
    assert y_pred_val.shape[0] == y_val.shape[0], \
        f"Sample count mismatch: {y_pred_val.shape[0]} vs {y_val.shape[0]}"
    
    # Verify number of targets matches
    assert y_pred_val.shape[1] == y_val.shape[1], \
        f"Target count mismatch: {y_pred_val.shape[1]} vs {y_val.shape[1]}"
    
    return y_pred_val


def dataframe_alignment_for_evaluation(y_true, y_pred, target_names):
    """
    Example: DataFrame alignment for competition evaluation.
    
    This demonstrates how DataFrames are created with explicit column names
    and row IDs to ensure proper alignment during evaluation.
    
    Args:
        y_true: Ground truth (n_samples, n_targets)
        y_pred: Predictions (n_samples, n_targets)
        target_names: List of target column names
    
    Returns:
        Aligned DataFrames for evaluation
    """
    n_samples, n_targets = y_true.shape
    
    # Create DataFrames with explicit column names
    solution_df = pd.DataFrame(y_true, columns=target_names)
    submission_df = pd.DataFrame(y_pred, columns=target_names)
    
    # Add row IDs for competition format
    solution_df['row_id'] = range(n_samples)
    submission_df['row_id'] = range(n_samples)
    
    # Verify alignment
    assert solution_df.shape == submission_df.shape, "Shape mismatch!"
    assert all(solution_df.columns == submission_df.columns), "Column mismatch!"
    
    # Verify row IDs match
    assert all(solution_df['row_id'] == submission_df['row_id']), "Row ID mismatch!"
    
    return solution_df, submission_df


def target_mapping_consistency(mapping, X_train, X_val):
    """
    Example: Consistent target mapping for train and validation.
    
    This demonstrates how the same target-to-node mapping is used
    for both training and validation to ensure consistency.
    
    Args:
        mapping: Target to node mapping dictionary
        X_train: Training features (n_train, n_nodes, n_features)
        X_val: Validation features (n_val, n_nodes, n_features)
    
    Returns:
        Extracted features for train and validation (consistent mapping)
    """
    target_idx = 0
    m = mapping[target_idx]
    
    # Same mapping used for both training and validation
    if m['type'] == 'single':
        # Single asset target
        node_idx = m['node_idx']
        F_train = X_train[:, node_idx, :]  # Training features
        F_val = X_val[:, node_idx, :]      # Validation features (same mapping)
    else:
        # Spread target
        n1, n2 = m['n1'], m['n2']
        F_train = X_train[:, n1, :] - X_train[:, n2, :]  # Training
        F_val = X_val[:, n1, :] - X_val[:, n2, :]         # Validation (same mapping)
    
    # Verify feature shapes are consistent
    assert F_train.shape[1] == F_val.shape[1], "Feature dimension mismatch!"
    
    return F_train, F_val


if __name__ == '__main__':
    print("Data Alignment Examples")
    print("=" * 50)
    print("These examples demonstrate how data alignment is ensured.")
    print("See TECHNICAL_CHALLENGES.md for detailed explanations.")

