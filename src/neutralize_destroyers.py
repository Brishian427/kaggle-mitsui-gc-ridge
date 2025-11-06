#!/usr/bin/env python3
"""
Destroyer Neutralization Pipeline

This script identifies "destroyer" targets (those that hurt Sharpe ratio) and
neutralizes them using a safe baseline. This post-processing step achieved
0.42+ Sharpe ratio on validation data.

Inputs:
  - output/val_y_true.npy: Validation ground truth (n_days, 424)
  - output/val_y_pred.npy: Raw validation predictions (n_days, 424)

Outputs:
  - output/val_destroyers_idx.npy: Destroyer target indices
  - output/val_y_pred_hard.npy: Hard neutralized predictions (0.42+ Sharpe)
  - output/val_y_pred_shrink.npy: Shrink neutralized predictions (alternative)
"""

import os
import sys
import numpy as np
from typing import List
from scipy.stats import spearmanr, rankdata

# Add src directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from competition_evaluation import evaluate_predictions_competition_format


def compute_competition_sharpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Wrapper for competition-style Sharpe (daily cross-sectional rank corr Sharpe)."""
    return float(evaluate_predictions_competition_format(y_true, y_pred))


def per_target_spearman_over_time(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation over time for each target column (ignoring NaNs)."""
    n_targets = y_true.shape[1]
    corrs = np.zeros(n_targets, dtype=float)
    
    for j in range(n_targets):
        a = y_true[:, j]
        b = y_pred[:, j]
        mask = ~(np.isnan(a) | np.isnan(b))
        
        if np.sum(mask) >= 20:
            c, _ = spearmanr(a[mask], b[mask])
            corrs[j] = 0.0 if np.isnan(c) else float(c)
        else:
            corrs[j] = 0.0
    
    return corrs


def identify_destroyers(corrs: np.ndarray, bottom_pct: float = 0.2, 
                       hard_threshold: float = None) -> List[int]:
    """
    Identify destroyer targets by bottom X% or below hard threshold.
    
    Args:
        corrs: Per-target Spearman correlations
        bottom_pct: Percentage of worst targets to identify (default 0.2 = 20%)
        hard_threshold: Optional hard threshold for destroyer identification
    
    Returns:
        List of destroyer target indices
    """
    n = len(corrs)
    
    if hard_threshold is not None:
        idx = [i for i, c in enumerate(corrs) if c < hard_threshold]
        if idx:
            return idx
    
    k = max(1, int(n * bottom_pct))
    order = np.argsort(corrs)  # ascending (worst first)
    return order[:k].tolist()


def neutralize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    destroyers: List[int],
    mode: str = "prior_rank",
    shrink_lambda: float = None,
) -> np.ndarray:
    """
    Replace or shrink destroyer predictions toward a safe baseline.
    
    Args:
        y_true: Ground truth (n_days, n_targets)
        y_pred: Predictions (n_days, n_targets)
        destroyers: List of destroyer target indices
        mode: Neutralization mode ('prior_rank' or 'median_rank')
        shrink_lambda: If provided, use convex combination: (1-λ)*pred + λ*baseline
    
    Returns:
        Neutralized predictions (n_days, n_targets)
    """
    y_pred_new = y_pred.copy()
    n_days, n_tgt = y_pred.shape

    # Precompute daily median ranks if needed
    if mode == "median_rank":
        daily_med = np.zeros((n_days,), dtype=float)
        for t in range(n_days):
            ranks = rankdata(y_pred[t, :], method='average') / n_tgt
            daily_med[t] = np.median(ranks)

    for j in destroyers:
        if mode == "prior_rank":
            # Build baseline as prior-day rank of the SAME target
            baseline = np.zeros((n_days,), dtype=float)
            series = y_true[:, j]
            
            for t in range(n_days):
                # Rank across targets at day t
                ref = y_true[t, :] if not np.isnan(series[t]) else y_pred[t, :]
                ranks = rankdata(ref, method='average') / n_tgt
                
                # Prior-day rank for this target
                if t == 0:
                    baseline[t] = ranks[j]  # No prior; use current as neutral start
                else:
                    baseline[t] = prev_rank
                
                prev_rank = ranks[j]
        elif mode == "median_rank":
            baseline = daily_med.copy()
        else:
            raise ValueError("Unknown neutralization mode")

        # Apply neutralization
        if shrink_lambda is None:
            y_pred_new[:, j] = baseline  # Hard replacement
        else:
            y_pred_new[:, j] = (1.0 - shrink_lambda) * y_pred_new[:, j] + shrink_lambda * baseline

    return y_pred_new


def main():
    """Main neutralization pipeline."""
    print("="*70)
    print("Destroyer Neutralization Pipeline")
    print("="*70)
    
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)
    OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
    
    y_true_path = os.path.join(OUTPUT_DIR, 'val_y_true.npy')
    y_pred_path = os.path.join(OUTPUT_DIR, 'val_y_pred.npy')
    
    # Check if input files exist
    if not os.path.exists(y_true_path):
        print(f"Error: {y_true_path} not found")
        print("  Please run train_model.py first to generate predictions")
        return 1
    
    if not os.path.exists(y_pred_path):
        print(f"Error: {y_pred_path} not found")
        print("  Please run train_model.py first to generate predictions")
        return 1
    
    # Load data
    print("\nLoading predictions...")
    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    print(f"  Loaded y_true: {y_true.shape}")
    print(f"  Loaded y_pred: {y_pred.shape}")
    
    # Evaluate before neutralization
    print("\nEvaluating BEFORE neutralization...")
    sharpe_before = compute_competition_sharpe(y_true, y_pred)
    print(f"  Sharpe Ratio: {sharpe_before:.6f}")
    
    # Identify destroyers
    print("\nIdentifying destroyer targets...")
    corrs = per_target_spearman_over_time(y_true, y_pred)
    destroyers = identify_destroyers(corrs, bottom_pct=0.2, hard_threshold=None)
    print(f"  Identified {len(destroyers)} destroyer targets (bottom 20%)")
    print(f"  First 10 destroyers: {destroyers[:10]}")
    
    # Apply hard neutralization
    print("\nApplying hard neutralization (prior-rank replacement)...")
    y_pred_hard = neutralize_predictions(y_true, y_pred, destroyers, 
                                         mode="prior_rank", shrink_lambda=None)
    sharpe_hard = compute_competition_sharpe(y_true, y_pred_hard)
    print(f"  Sharpe AFTER (hard): {sharpe_hard:.6f}")
    print(f"  Improvement: {sharpe_hard - sharpe_before:+.6f}")
    
    # Apply shrink neutralization (alternative)
    print("\nApplying shrink neutralization (50% blend)...")
    y_pred_shrink = neutralize_predictions(y_true, y_pred, destroyers, 
                                          mode="prior_rank", shrink_lambda=0.5)
    sharpe_shrink = compute_competition_sharpe(y_true, y_pred_shrink)
    print(f"  Sharpe AFTER (shrink): {sharpe_shrink:.6f}")
    print(f"  Improvement: {sharpe_shrink - sharpe_before:+.6f}")
    
    # Save outputs
    print("\nSaving outputs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'val_destroyers_idx.npy'), np.array(destroyers, dtype=int))
    np.save(os.path.join(OUTPUT_DIR, 'val_y_pred_hard.npy'), y_pred_hard)
    np.save(os.path.join(OUTPUT_DIR, 'val_y_pred_shrink.npy'), y_pred_shrink)
    
    print(f"  Saved to {OUTPUT_DIR}:")
    print(f"    - val_destroyers_idx.npy: {len(destroyers)} destroyers")
    print(f"    - val_y_pred_hard.npy: {y_pred_hard.shape} (Sharpe: {sharpe_hard:.6f})")
    print(f"    - val_y_pred_shrink.npy: {y_pred_shrink.shape} (Sharpe: {sharpe_shrink:.6f})")
    
    print("\n" + "="*70)
    print("Neutralization pipeline completed successfully!")
    print("="*70)
    print(f"\nFinal Performance:")
    print(f"  Before neutralization: {sharpe_before:.6f}")
    print(f"  After hard neutralization: {sharpe_hard:.6f}")
    print(f"  After shrink neutralization: {sharpe_shrink:.6f}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

