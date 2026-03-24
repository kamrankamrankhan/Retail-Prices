"""
Statistical analysis utilities.
"""
import numpy as np

def calculate_percentile(values, percentile):
    """Calculate percentile of values."""
    return np.percentile(values, percentile)

def calculate_iqr(values):
    """Calculate interquartile range."""
    return np.percentile(values, 75) - np.percentile(values, 25)

def calculate_zscore(value, mean, std):
    """Calculate z-score."""
    if std == 0:
        return 0
    return (value - mean) / std
