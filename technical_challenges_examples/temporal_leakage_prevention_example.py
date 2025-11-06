"""
Temporal Leakage Prevention Examples

This file demonstrates how temporal data leakage is prevented in the GC-Ridge pipeline.
These examples show the strict temporal separation enforced at multiple levels.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge


def temporal_split_before_processing(X, y, masks):
    """
    Example: Temporal split before any processing.
    
    This demonstrates the critical principle: split data temporally
    BEFORE any feature processing to prevent leakage.
    
    Args:
        X: Features (n_samples, n_nodes, n_features)
        y: Targets (n_samples, n_targets)
        masks: Trading masks (n_samples, n_nodes)
    
    Returns:
        Training and validation sets (temporally separated)
    """
    # Temporal split (same as production)
    train_end = 1828
    val_start, val_end = 1829, 1950
    
    # Split BEFORE any feature processing
    # This ensures no future information can leak into training
    X_train = X[:train_end]
    X_val = X[val_start:val_end]
    y_train = y[:train_end]
    y_val = y[val_start:val_end]
    masks_train = masks[:train_end]
    masks_val = masks[val_start:val_end]
    
    return X_train, X_val, y_train, y_val, masks_train, masks_val


def fit_transform_separation(features_train, y_train, features_val):
    """
    Example: Fit-transform separation to prevent leakage.
    
    This demonstrates how transformers are fitted on training data only
    and then applied to validation data.
    
    Args:
        features_train: Training features (n_train, n_features)
        y_train: Training targets (n_train,)
        features_val: Validation features (n_val, n_features)
    
    Returns:
        Transformed validation features (no leakage)
    """
    # Step 1: Feature selection (fitted on training only)
    k = min(30, features_train.shape[1])
    selector = SelectKBest(f_regression, k=k)
    selector.fit(features_train, y_train)  # Training data only
    selected_indices = selector.get_support(indices=True)
    
    # Step 2: Scaling (fitted on training only)
    scaler = StandardScaler()
    scaler.fit(features_train[:, selected_indices])  # Training data only
    
    # Step 3: Transform validation data (using pre-fitted transformers)
    features_val_selected = features_val[:, selected_indices]
    features_val_scaled = scaler.transform(features_val_selected)  # No leakage
    
    return features_val_scaled, selector, scaler


def model_training_without_leakage(features_train, y_train, features_val):
    """
    Example: Model training without temporal leakage.
    
    This demonstrates how models are trained exclusively on training data
    and then used to predict on validation data.
    
    Args:
        features_train: Training features (n_train, n_features)
        y_train: Training targets (n_train,)
        features_val: Validation features (n_val, n_features)
    
    Returns:
        Validation predictions (no leakage)
    """
    # Fit transformers on training data only
    selector = SelectKBest(f_regression, k=30)
    selector.fit(features_train, y_train)
    
    scaler = StandardScaler()
    scaler.fit(features_train[:, selector.get_support(indices=True)])
    
    # Train model on training data only
    model = Ridge(alpha=0.05)
    features_train_transformed = scaler.transform(
        features_train[:, selector.get_support(indices=True)]
    )
    model.fit(features_train_transformed, y_train)  # Training data only
    
    # Predict on validation (using pre-fitted transformers)
    features_val_transformed = scaler.transform(
        features_val[:, selector.get_support(indices=True)]
    )
    y_pred_val = model.predict(features_val_transformed)  # No leakage
    
    return y_pred_val


def graph_convolution_separation(X_train, X_val, adjs):
    """
    Example: Graph convolution applied separately to prevent leakage.
    
    This demonstrates how graph convolution is applied independently
    to training and validation sets.
    
    Args:
        X_train: Training features (n_train, n_nodes, n_features)
        X_val: Validation features (n_val, n_nodes, n_features)
        adjs: Adjacency matrices (no temporal dependency)
    
    Returns:
        Convolved features for train and validation (separate)
    """
    # Graph convolution applied separately
    # Convolution weights are learned from training data only
    gc = GraphConv(hidden_dim=64, random_state=42)
    
    Xc_train = gc.conv(X_train, adjs)  # Training convolution
    Xc_val = gc.conv(X_val, adjs)      # Validation convolution (separate)
    
    return Xc_train, Xc_val


if __name__ == '__main__':
    print("Temporal Leakage Prevention Examples")
    print("=" * 50)
    print("These examples demonstrate how temporal leakage is prevented.")
    print("See TECHNICAL_CHALLENGES.md for detailed explanations.")


# Placeholder for GraphConv class (for example purposes)
class GraphConv:
    def __init__(self, hidden_dim=64, random_state=42):
        self.hidden_dim = hidden_dim
        np.random.seed(random_state)
    
    def conv(self, X, adjs):
        """Simplified graph convolution for example."""
        n, nodes, feats = X.shape
        Xc = np.nan_to_num(X, nan=0.0)
        out = np.zeros((n, nodes, self.hidden_dim))
        # ... graph convolution logic ...
        return out

