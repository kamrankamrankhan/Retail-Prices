"""
Missing value imputation utilities.
"""
import numpy as np

def mean_impute(values):
    """Impute missing values with mean."""
    mean_val = np.nanmean(values)
    return [v if not np.isnan(v) else mean_val for v in values]

def median_impute(values):
    """Impute missing values with median."""
    median_val = np.nanmedian(values)
    return [v if not np.isnan(v) else median_val for v in values]
