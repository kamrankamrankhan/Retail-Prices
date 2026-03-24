"""
Seasonality analysis utilities.
"""
import numpy as np

def detect_seasonality(values, period=12):
    """Detect seasonal pattern."""
    n = len(values)
    if n < period * 2:
        return None
    seasonal = []
    for i in range(period):
        indices = range(i, n, period)
        seasonal.append(np.mean([values[j] for j in indices]))
    return seasonal

def remove_seasonality(values, period=12):
    """Remove seasonal component."""
    seasonal = detect_seasonality(values, period)
    if seasonal is None:
        return values
    return [values[i] - seasonal[i % period] for i in range(len(values))]
