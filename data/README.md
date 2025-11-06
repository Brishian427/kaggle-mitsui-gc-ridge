# Data Directory

This directory contains the required data files for the GC-Ridge training pipeline.

## Required Files

The following files must be present in this directory:

- `train_labels.csv` - Training labels with target values (1961 rows, 425 columns)
- `target_pairs.csv` - Target pair definitions mapping targets to assets (424 rows)
- `tensor_features_cleaned_1961.pkl` - Preprocessed tensor features (1961, 147, 11)
- `corrected_node_close_mappings.json` - Node mapping from assets to graph nodes
- `adjacency_matrices/topology_matrix_146.npz` - Adjacency matrices for graph convolution

## File Sizes

These files are typically large (several MB to GB) and may not be included in the repository. You will need to:

1. Obtain them from the original competition data
2. Generate them from raw data using the preprocessing pipeline
3. Request them separately if not available

## Verification

Run the verification script to check if all files are present:

```bash
python setup_data.py
```

## Notes

- The data files are required for training and validation
- File names must match exactly (case-sensitive)
- See `DATA_PREPARATION.md` in the root directory of this project for detailed instructions

