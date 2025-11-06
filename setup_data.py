#!/usr/bin/env python3
"""
Data Setup Verification Script

This script verifies that all required data files are present in the data/
directory. It does not copy files from external locations, as this folder
is designed to be self-contained.

Usage:
    python setup_data.py
"""

import os
import sys

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
ADJ_DIR = os.path.join(DATA_DIR, 'adjacency_matrices')

# Required files and their expected locations
REQUIRED_FILES = {
    'train_labels.csv': os.path.join(DATA_DIR, 'train_labels.csv'),
    'target_pairs.csv': os.path.join(DATA_DIR, 'target_pairs.csv'),
    'tensor_features_cleaned_1961.pkl': os.path.join(DATA_DIR, 'tensor_features_cleaned_1961.pkl'),
    'corrected_node_close_mappings.json': os.path.join(DATA_DIR, 'corrected_node_close_mappings.json'),
    'topology_matrix_146.npz': os.path.join(ADJ_DIR, 'topology_matrix_146.npz'),
}

def verify_data():
    """Verify that all required data files are present.
    
    Returns:
        True if all files are present, False otherwise.
    """
    print("="*70)
    print("Verifying data directory")
    print("="*70)
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ADJ_DIR, exist_ok=True)
    
    found_files = []
    missing_files = []
    
    # Check each required file
    for filename, filepath in REQUIRED_FILES.items():
        if os.path.exists(filepath):
            found_files.append(filename)
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
            print(f"  [OK] {filename}")
            print(f"    Location: {filepath}")
            print(f"    Size: {file_size:.2f} MB")
        else:
            missing_files.append(filename)
            print(f"  [MISSING] {filename}")
            print(f"    Expected: {filepath}")
    
    print("\n" + "="*70)
    print("Verification Summary")
    print("="*70)
    print(f"  Found: {len(found_files)} files")
    print(f"  Missing: {len(missing_files)} files")
    
    if found_files:
        print(f"\n  Found files:")
        for filename in found_files:
            print(f"    - {filename}")
    
    if missing_files:
        print(f"\n  Missing files:")
        for filename in missing_files:
            print(f"    - {filename}")
        print(f"\n  Please provide these files in the data/ directory.")
        print(f"  See DATA_PREPARATION.md for instructions on preparing these files.")
    
    print("="*70)
    
    return len(missing_files) == 0

if __name__ == '__main__':
    success = verify_data()
    if success:
        print("\nAll required data files are present!")
        print("You can now run the training pipeline with: python run_pipeline.py")
    else:
        print("\nSome required data files are missing.")
        print("Please provide the missing files before running the training pipeline.")
        sys.exit(1)
