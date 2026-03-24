"""
Outlier detection utilities.
"""
import numpy as np

def detect_outliers_iqr(values, threshold=1.5):
    """Detect outliers using IQR method."""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    return [i for i, v in enumerate(values) if v < lower or v > upper]

def detect_outliers_zscore(values, threshold=3):
    """Detect outliers using Z-score method."""
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return []
    return [i for i, v in enumerate(values) if abs((v - mean) / std) > threshold]
