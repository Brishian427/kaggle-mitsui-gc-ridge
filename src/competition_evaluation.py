"""
Competition Evaluation Framework

This module implements the exact evaluation method used by the Mitsui Competition.
It matches the rank_correlation_sharpe_ratio function from the competition API.
"""

import numpy as np
import pandas as pd
from typing import Optional, List

# Competition constants
SOLUTION_NULL_FILLER = -999999
NUM_TARGET_COLUMNS = 424


def rank_correlation_sharpe_ratio(merged_df: pd.DataFrame) -> float:
    """
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).
    
    This is the EXACT function from the competition evaluation framework.
    """
    prediction_cols = [col for col in merged_df.columns if col.startswith('prediction_')]
    target_cols = [col for col in merged_df.columns if col.startswith('target_')]

    def _compute_rank_correlation(row):
        """Compute rank correlation for a single row (timestep)."""
        non_null_targets = [col for col in target_cols if not pd.isnull(row[col])]
        matching_predictions = [col for col in prediction_cols 
                               if col.replace('prediction_', 'target_') in non_null_targets]
        
        if not non_null_targets:
            raise ValueError('No non-null target values found')
        
        if row[non_null_targets].std(ddof=0) == 0 or row[matching_predictions].std(ddof=0) == 0:
            raise ZeroDivisionError('Denominator is zero, unable to compute rank correlation.')
        
        return np.corrcoef(
            row[matching_predictions].rank(method='average'), 
            row[non_null_targets].rank(method='average')
        )[0, 1]

    # Compute daily rank correlations
    daily_rank_corrs = merged_df.apply(_compute_rank_correlation, axis=1)
    
    # Calculate Sharpe ratio
    std_dev = daily_rank_corrs.std(ddof=0)
    if std_dev == 0:
        raise ZeroDivisionError('Denominator is zero, unable to compute Sharpe ratio.')
    
    sharpe_ratio = daily_rank_corrs.mean() / std_dev
    return float(sharpe_ratio)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).
    
    This is the EXACT function from the competition evaluation framework.
    """
    solution_copy = solution.copy()
    submission_copy = submission.copy()
    
    # Remove row ID column
    if row_id_column_name in solution_copy.columns:
        del solution_copy[row_id_column_name]
    if row_id_column_name in submission_copy.columns:
        del submission_copy[row_id_column_name]
    
    # Ensure columns match
    assert all(solution_copy.columns == submission_copy.columns), \
        f"Columns don't match: {solution_copy.columns} vs {submission_copy.columns}"

    # Rename submission columns to prediction format
    submission_copy = submission_copy.rename(columns={
        col: col.replace('target_', 'prediction_') for col in submission_copy.columns
    })

    # Handle null fillers in solution
    solution_copy = solution_copy.replace(SOLUTION_NULL_FILLER, None)
    
    # Merge and evaluate
    merged_df = pd.concat([solution_copy, submission_copy], axis='columns')
    return rank_correlation_sharpe_ratio(merged_df)


def evaluate_predictions_competition_format(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_names: Optional[List[str]] = None
) -> float:
    """
    Evaluate predictions using the exact competition format.
    
    Args:
        y_true: True target values (n_samples, n_targets)
        y_pred: Predicted target values (n_samples, n_targets)
        target_names: Optional list of target names
    
    Returns:
        Competition-style Sharpe ratio
    """
    n_samples, n_targets = y_true.shape
    
    # Create target column names
    if target_names is None:
        target_names = [f'target_{i}' for i in range(n_targets)]
    
    # Create DataFrames in competition format
    solution_df = pd.DataFrame(y_true, columns=target_names)
    submission_df = pd.DataFrame(y_pred, columns=target_names)
    
    # Add dummy row ID column (competition format requirement)
    solution_df['row_id'] = range(n_samples)
    submission_df['row_id'] = range(n_samples)
    
    # Use competition evaluation
    return score(solution_df, submission_df, 'row_id')

