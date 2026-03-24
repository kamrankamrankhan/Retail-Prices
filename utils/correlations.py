"""
Correlation analysis utilities.
"""
import pandas as pd
import numpy as np

def pearson_correlation(x, y):
    """Calculate Pearson correlation."""
    return np.corrcoef(x, y)[0, 1]

def correlation_matrix(df):
    """Calculate correlation matrix."""
    return df.corr()
