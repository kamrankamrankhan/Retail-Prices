"""
Mathematical utility functions.
"""
import numpy as np

def calculate_mean(values):
    """Calculate mean of values."""
    return np.mean(values) if values else 0

def calculate_std(values):
    """Calculate standard deviation."""
    return np.std(values) if values else 0

def calculate_median(values):
    """Calculate median of values."""
    return np.median(values) if values else 0
