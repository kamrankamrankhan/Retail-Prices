"""
Simple forecasting utilities.
"""
import numpy as np

def naive_forecast(values, periods=1):
    """Naive forecast using last value."""
    return [values[-1]] * periods

def average_forecast(values, periods=1):
    """Average forecast using mean of values."""
    avg = np.mean(values)
    return [avg] * periods

def drift_forecast(values, periods=1):
    """Drift forecast using linear trend."""
    n = len(values)
    slope = (values[-1] - values[0]) / (n - 1)
    return [values[-1] + slope * (i + 1) for i in range(periods)]
