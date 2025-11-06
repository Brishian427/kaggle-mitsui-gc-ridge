#!/usr/bin/env python3
"""
Graph Convolutional Ridge Training Pipeline

This script trains the GC-Ridge model on training data and generates predictions
on validation data. It implements the complete training pipeline that achieved
0.42+ Sharpe ratio after destroyer neutralization.

Outputs:
  - output/val_y_true.npy: Validation ground truth (n_days, 424)
  - output/val_y_pred.npy: Raw validation predictions (n_days, 424)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Add src directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)


class GraphConv:
    """Graph Convolutional Layer for market structure modeling."""
    
    def __init__(self, hidden_dim=64, random_state=42):
        self.hidden_dim = hidden_dim
        self.proj = {}
        np.random.seed(random_state)

    def conv(self, X, adjs):
        """Apply graph convolution with 6 edge types."""
        n, nodes, feats = X.shape
        Xc = np.nan_to_num(X, nan=0.0)
        out = np.zeros((n, nodes, self.hidden_dim))
        
        for et in range(6):
            if et not in self.proj:
                limit = np.sqrt(6.0 / (feats + self.hidden_dim // 6))
                self.proj[et] = np.random.uniform(-limit, limit, (feats, self.hidden_dim // 6))
            
            adj = adjs[et]
            P = self.proj[et]
            
            for i in range(nodes):
                nbrs = np.where(adj[i] > 0)[0]
                s, e = et * (self.hidden_dim // 6), (et + 1) * (self.hidden_dim // 6)
                
                if len(nbrs) > 0:
                    w = adj[i, nbrs]
                    wf = np.average(Xc[:, nbrs, :], weights=w, axis=1)
                    out[:, i, s:e] = wf @ P
                else:
                    out[:, i, s:e] = Xc[:, i, :] @ P
        
        return out


def load_adjacency_matrices(npz_path, remove_type=1):
    """
    Load and prepare adjacency matrices for graph convolution.
    
    Args:
        npz_path: Path to adjacency matrix .npz file
        remove_type: Edge type to remove (1 = permanently removed for performance)
    
    Returns:
        List of 6 adjacency matrices (147x147 each)
    """
    adj_data = np.load(npz_path)
    A = csr_matrix((adj_data['data'], adj_data['indices'], adj_data['indptr']), 
                   shape=adj_data['shape']).toarray()
    
    # Pad to 147x147 (146 nodes + 1 phantom node)
    P = np.zeros((147, 147))
    P[:146, :146] = A
    P[146, 146] = 1.0  # Self-loop for phantom node
    
    # Create 6 edge type matrices based on correlation thresholds
    mats = []
    for i in range(6):
        thr = 0.1 + 0.1 * i  # Thresholds: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
        M = (P > thr).astype(float)
        
        # Permanently remove edge type 1 (improves performance)
        if i == remove_type:
            M = np.zeros_like(M)
        
        mats.append(M)
    
    return mats


def simple_aggregation(Xc, i1, i2):
    """Simple aggregation for spread targets: difference of two nodes."""
    return Xc[:, i1, :] - Xc[:, i2, :]


def load_data(data_dir):
    """Load all required data files.
    
    Args:
        data_dir: Directory containing data files.
        
    Returns:
        Dictionary containing loaded data and file paths.
        
    Raises:
        FileNotFoundError: If any required data file is missing.
    """
    print("Loading data...")
    
    # Load training labels
    labels_path = os.path.join(data_dir, 'train_labels.csv')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Training labels not found: {labels_path}")
    train_labels = pd.read_csv(labels_path)
    print(f"  Training labels: {train_labels.shape}")
    
    # Load tensor features
    feats_path = os.path.join(data_dir, 'tensor_features_cleaned_1961.pkl')
    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Tensor features not found: {feats_path}")
    with open(feats_path, 'rb') as f:
        cleaned = pickle.load(f)
    X = cleaned['tensor_data']
    masks = cleaned.get('trading_masks', np.ones((1961, 147)))
    print(f"  Tensor features: {X.shape}")
    print(f"  Trading masks: {masks.shape}")
    
    # Load adjacency matrices
    adjs_path = os.path.join(data_dir, 'adjacency_matrices', 'topology_matrix_146.npz')
    if not os.path.exists(adjs_path):
        raise FileNotFoundError(f"Adjacency matrices not found: {adjs_path}")
    print(f"  Adjacency matrices: {adjs_path}")
    
    # Load node mapping
    node_map_path = os.path.join(data_dir, 'corrected_node_close_mappings.json')
    if not os.path.exists(node_map_path):
        raise FileNotFoundError(f"Node mapping not found: {node_map_path}")
    print(f"  Node mapping: {node_map_path}")
    
    # Load target pairs
    target_pairs_path = os.path.join(data_dir, 'target_pairs.csv')
    if not os.path.exists(target_pairs_path):
        raise FileNotFoundError(f"Target pairs not found: {target_pairs_path}")
    print(f"  Target pairs: {target_pairs_path}")
    
    return {
        'train_labels': train_labels,
        'X': X,
        'masks': masks,
        'adjs_path': adjs_path,
        'node_map_path': node_map_path,
        'target_pairs_path': target_pairs_path
    }


def create_target_mapping(node_map_path, target_pairs_path):
    """Create mapping from targets to graph nodes.
    
    Args:
        node_map_path: Path to node mapping JSON file.
        target_pairs_path: Path to target pairs CSV file.
        
    Returns:
        Dictionary mapping target indices to node indices and types.
    """
    print("Creating target mapping...")
    
    with open(node_map_path, 'r') as f:
        node_map = json.load(f)
    
    # Build asset to node mapping
    asset2node = {}
    for ni, (k, arr) in enumerate(node_map.items()):
        for a in arr:
            asset2node[a] = ni
    
    # Load target pairs
    pairs = pd.read_csv(target_pairs_path)
    
    # Create target mapping
    mapping = {}
    for idx, row in pairs.iterrows():
        tname = row['target']
        tidx = int(tname.split('_')[1])
        pair = row['pair']
        
        if '-' not in pair:
            # Single asset target
            if pair in asset2node:
                mapping[tidx] = {'type': 'single', 'node_idx': asset2node[pair]}
        else:
            # Spread target
            a1, a2 = pair.split(' - ')
            if a1 in asset2node and a2 in asset2node:
                mapping[tidx] = {'type': 'spread', 'n1': asset2node[a1], 'n2': asset2node[a2]}
    
    print(f"  Mapped {len(mapping)} targets")
    return mapping


def train_and_predict(X_train, X_val, y_train, y_val, masks_train, masks_val, 
                     adjs_path, mapping, output_dir):
    """Train GC-Ridge models and generate predictions.
    
    Args:
        X_train: Training tensor features (n_train, 147, 11).
        X_val: Validation tensor features (n_val, 147, 11).
        y_train: Training target values (n_train, 424).
        y_val: Validation target values (n_val, 424).
        masks_train: Training trading masks (n_train, 147).
        masks_val: Validation trading masks (n_val, 147).
        adjs_path: Path to adjacency matrices file.
        mapping: Target to node mapping dictionary.
        output_dir: Directory to save output files.
        
    Returns:
        Validation predictions array (n_val, 424).
    """
    print("\nTraining GC-Ridge models...")
    
    # Load adjacency matrices
    adjs = load_adjacency_matrices(adjs_path, remove_type=1)
    print(f"  Loaded {len(adjs)} adjacency matrices (edge type 1 removed)")
    
    # Graph convolution
    gc = GraphConv(hidden_dim=64, random_state=42)
    Xc_train = gc.conv(X_train, adjs) * masks_train.reshape(-1, 147, 1)
    Xc_val = gc.conv(X_val, adjs) * masks_val.reshape(-1, 147, 1)
    print(f"  Graph convolution: {Xc_train.shape} -> {Xc_val.shape}")
    
    # Train per-target Ridge models with feature selection
    models = {}
    scalers = {}
    selectors = {}
    trained_count = 0
    
    for t, m in mapping.items():
        yt = y_train[:, t]
        valid = ~np.isnan(yt)
        
        if valid.sum() < 50:  # Skip targets with too few samples
            continue
        
        # Extract features based on target type
        if m['type'] == 'single':
            F = Xc_train[:, m['node_idx'], :]
        else:
            F = simple_aggregation(Xc_train, m['n1'], m['n2'])
        
        Fv = F[valid]
        yv = yt[valid]
        
        # Feature selection (top K features)
        k = min(30, Fv.shape[1])
        sel = SelectKBest(f_regression, k=k).fit(Fv, yv)
        idxs = sel.get_support(indices=True)
        selectors[t] = idxs
        
        # Standardize features
        Sc = StandardScaler().fit(Fv[:, idxs])
        scalers[t] = Sc
        
        # Train Ridge model
        model = Ridge(alpha=0.05)
        model.fit(Sc.transform(Fv[:, idxs]), yv)
        models[t] = model
        trained_count += 1
        
        if trained_count % 50 == 0:
            print(f"  Progress: {trained_count} targets trained...")
    
    print(f"  Trained {trained_count} target models")
    
    # Generate predictions on validation set
    print("\nGenerating validation predictions...")
    n_val = Xc_val.shape[0]
    y_pred_val = np.full((n_val, y_val.shape[1]), np.nan)
    
    for t, m in mapping.items():
        if t not in models:
            continue
        
        # Extract features for validation
        if m['type'] == 'single':
            F = Xc_val[:, m['node_idx'], :]
        else:
            F = simple_aggregation(Xc_val, m['n1'], m['n2'])
        
        # Apply feature selection and scaling
        idxs = selectors[t]
        Sc = scalers[t]
        y_pred_val[:, t] = models[t].predict(Sc.transform(F[:, idxs]))
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'val_y_true.npy'), y_val)
    np.save(os.path.join(output_dir, 'val_y_pred.npy'), y_pred_val)
    
    print(f"\nSaved outputs to {output_dir}:")
    print(f"  - val_y_true.npy: {y_val.shape}")
    print(f"  - val_y_pred.npy: {y_pred_val.shape}")
    
    return y_pred_val


def main():
    """Main training pipeline."""
    print("="*70)
    print("GC-Ridge Training Pipeline")
    print("="*70)
    
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')
    OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
    
    # Load data
    data = load_data(DATA_DIR)
    
    # Prepare data
    train_labels = data['train_labels']
    X = data['X']
    masks = data['masks']
    y = train_labels.iloc[:, 1:].values  # Skip date_id column
    
    # Temporal split (same as production)
    train_end = 1828
    val_start, val_end = 1829, 1950
    
    X_train = X[:train_end]
    X_val = X[val_start:val_end]
    y_train = y[:train_end]
    y_val = y[val_start:val_end]
    masks_train = masks[:train_end]
    masks_val = masks[val_start:val_end]
    
    print(f"\nData Split:")
    print(f"  Training: {X_train.shape[0]} days (0-{train_end})")
    print(f"  Validation: {X_val.shape[0]} days ({val_start}-{val_end})")
    
    # Create target mapping
    mapping = create_target_mapping(data['node_map_path'], data['target_pairs_path'])
    
    # Train and predict
    y_pred_val = train_and_predict(
        X_train, X_val, y_train, y_val,
        masks_train, masks_val,
        data['adjs_path'], mapping, OUTPUT_DIR
    )
    
    print("\n" + "="*70)
    print("Training pipeline completed successfully!")
    print("="*70)
    print(f"\nNext step: Run neutralize_destroyers.py to apply destroyer neutralization")
    print(f"  python src/neutralize_destroyers.py")
    print("="*70)


if __name__ == '__main__':
    main()

