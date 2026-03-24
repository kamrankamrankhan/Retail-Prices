"""
Data normalization utilities.
"""
import numpy as np

def min_max_normalize(values):
    """Min-Max normalization."""
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def zscore_normalize(values):
    """Z-score normalization."""
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return [0] * len(values)
    return [(v - mean) / std for v in values]
