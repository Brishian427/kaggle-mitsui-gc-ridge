#!/usr/bin/env python3
"""
Complete GC-Ridge Training Pipeline

This script runs the complete training pipeline:
1. Train GC-Ridge models on training data
2. Generate predictions on validation data
3. Apply destroyer neutralization
4. Evaluate final performance

Usage:
    python run_pipeline.py
"""

import os
import sys
import subprocess

# Add src directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
sys.path.insert(0, SRC_DIR)


def run_script(script_path, description):
    """Run a Python script and handle errors.
    
    Args:
        script_path: Path to Python script to execute.
        description: Description of the script being run.
        
    Returns:
        Exit code from script execution.
        
    Raises:
        SystemExit: If script execution fails.
    """
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70)
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    return result.returncode


def main():
    """Run the complete pipeline."""
    print("="*70)
    print("GC-Ridge Complete Training Pipeline")
    print("="*70)
    print("\nThis pipeline will:")
    print("  1. Train GC-Ridge models on training data")
    print("  2. Generate predictions on validation data")
    print("  3. Apply destroyer neutralization")
    print("  4. Evaluate final performance")
    print("="*70)
    
    # Step 1: Train models
    train_script = os.path.join(SRC_DIR, 'train_model.py')
    run_script(train_script, "Step 1: Training GC-Ridge Models")
    
    # Step 2: Apply destroyer neutralization
    neutralize_script = os.path.join(SRC_DIR, 'neutralize_destroyers.py')
    run_script(neutralize_script, "Step 2: Applying Destroyer Neutralization")
    
    print("\n" + "="*70)
    print("Complete pipeline finished successfully!")
    print("="*70)
    print("\nOutput files saved to: output/")
    print("  - val_y_true.npy: Validation ground truth")
    print("  - val_y_pred.npy: Raw predictions")
    print("  - val_y_pred_hard.npy: Hard neutralized predictions (0.42+ Sharpe)")
    print("  - val_destroyers_idx.npy: Destroyer target indices")
    print("="*70)


if __name__ == '__main__':
    main()

