# Data Preparation Guide

This guide explains how to prepare the required data files for the GC-Ridge training pipeline.

## Required Files

The following files must be placed in the `data/` directory:

### 1. `train_labels.csv`

Training labels with target values for each date.

**Format**:
- First column: `date_id` (integer, 0-1960)
- Remaining columns: `target_0`, `target_1`, ..., `target_423` (424 target columns)
- Values: Float or NaN

**Example**:
```csv
date_id,target_0,target_1,...,target_423
0,0.001234,-0.000567,...,0.002345
1,0.000890,0.001234,...,-0.000123
...
```

### 2. `target_pairs.csv`

Target pair definitions mapping each target to its underlying assets.

**Format**:
- Columns: `target`, `pair`, `lag`
- `target`: Target name (e.g., "target_0")
- `pair`: Asset pair (e.g., "LME_Aluminum" or "LME_Aluminum - LME_Copper")
- `lag`: Lag group (1, 2, 3, or 4)

**Example**:
```csv
target,pair,lag
target_0,LME_Aluminum,1
target_1,LME_Copper - LME_Aluminum,2
...
```

### 3. `tensor_features_cleaned_1961.pkl`

Preprocessed tensor features for graph convolution.

**Format**:
- Python pickle file
- Dictionary with keys:
  - `tensor_data`: numpy array (1961, 147, 11) - Node features for each day
  - `trading_masks`: numpy array (1961, 147) - Trading masks (optional)

**How to create**:
```python
import pickle
import numpy as np

# Your feature extraction code here
tensor_data = ...  # (1961, 147, 11)
trading_masks = ...  # (1961, 147)

data = {
    'tensor_data': tensor_data,
    'trading_masks': trading_masks
}

with open('tensor_features_cleaned_1961.pkl', 'wb') as f:
    pickle.dump(data, f)
```

### 4. `corrected_node_close_mappings.json`

Mapping from assets to graph nodes.

**Format**:
- JSON file
- Dictionary mapping node indices to lists of asset names

**Example**:
```json
{
  "0": ["LME_Aluminum", "LME_Aluminum_Spot"],
  "1": ["LME_Copper", "LME_Copper_Spot"],
  ...
}
```

### 5. `adjacency_matrices/topology_matrix_146.npz`

Adjacency matrices for graph convolution.

**Format**:
- NumPy compressed format (.npz)
- Contains sparse matrix data: `data`, `indices`, `indptr`, `shape`

**How to create**:
```python
import numpy as np
from scipy.sparse import csr_matrix

# Your adjacency matrix construction code here
adj_matrix = ...  # (146, 146) sparse matrix

np.savez_compressed('topology_matrix_146.npz',
                    data=adj_matrix.data,
                    indices=adj_matrix.indices,
                    indptr=adj_matrix.indptr,
                    shape=adj_matrix.shape)
```

## Data Preparation Steps

### Option 1: Verify Data Files

Run the verification script to check if all required data files are present:

```bash
python setup_data.py
```

This script will:
1. Check if all required files exist in the `data/` directory
2. Report file sizes and locations
3. List any missing files that need to be provided

### Option 2: Provide Data Files

If you have access to the original Mitsui competition data, place the following files in the `data/` directory:

1. `train_labels.csv` - Training labels from competition data
2. `target_pairs.csv` - Target pair definitions from competition data
3. `tensor_features_cleaned_1961.pkl` - Preprocessed tensor features
4. `corrected_node_close_mappings.json` - Node mapping file
5. `adjacency_matrices/topology_matrix_146.npz` - Adjacency matrices

Note: These files are typically large and may not be included in the repository. You will need to obtain them separately or generate them from raw data.

### Option 3: Prepare from Scratch

If you need to prepare data from scratch:

1. **Extract features**: Create tensor features from raw price data
2. **Build graph**: Construct adjacency matrices from asset correlations
3. **Map targets**: Create target-to-node mapping from target pairs
4. **Prepare labels**: Format training labels with date_id and target columns

## Directory Structure

After preparation, your `data/` directory should look like:

```
data/
├── train_labels.csv
├── target_pairs.csv
├── tensor_features_cleaned_1961.pkl
├── corrected_node_close_mappings.json
└── adjacency_matrices/
    └── topology_matrix_146.npz
```

## Verification

To verify your data files are correct:

```python
import os
import pickle
import numpy as np
import pandas as pd
import json

data_dir = 'data'

# Check train_labels.csv
labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
print(f"Labels shape: {labels.shape}")
print(f"Date range: {labels['date_id'].min()} - {labels['date_id'].max()}")

# Check target_pairs.csv
pairs = pd.read_csv(os.path.join(data_dir, 'target_pairs.csv'))
print(f"Target pairs: {len(pairs)}")

# Check tensor features
with open(os.path.join(data_dir, 'tensor_features_cleaned_1961.pkl'), 'rb') as f:
    features = pickle.load(f)
print(f"Tensor data shape: {features['tensor_data'].shape}")

# Check node mapping
with open(os.path.join(data_dir, 'corrected_node_close_mappings.json'), 'r') as f:
    node_map = json.load(f)
print(f"Node mapping: {len(node_map)} nodes")

# Check adjacency matrices
adj_data = np.load(os.path.join(data_dir, 'adjacency_matrices', 'topology_matrix_146.npz'))
print(f"Adjacency matrix shape: {adj_data['shape']}")
```

## Important Notes

1. **File names must match exactly** (case-sensitive)
2. **Data dimensions must match**:
   - Tensor features: (1961, 147, 11)
   - Training labels: (1961, 425) - 1 date_id + 424 targets
   - Adjacency matrix: (146, 146) - will be padded to (147, 147)
3. **Date range**: Training uses dates 0-1828, validation uses 1829-1950
4. **Target count**: Must have exactly 424 targets

## Troubleshooting

### Missing Files

If files are missing:
- Check file paths are correct
- Verify file names match exactly
- Ensure files are in the `data/` directory

### Dimension Mismatches

If you see dimension errors:
- Check tensor features shape: (1961, 147, 11)
- Check training labels shape: (1961, 425)
- Verify target count: 424 targets

### Format Errors

If you see format errors:
- Verify CSV files have correct headers
- Check JSON file is valid
- Ensure pickle file is not corrupted

---

For detailed instructions on preparing these files, see the sections above. All data files should be placed in the `data/` directory before running the pipeline.

