"""
Trend analysis utilities.
"""
import numpy as np

def calculate_trend(values):
    """Calculate linear trend."""
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    return slope, intercept

def detrend(values):
    """Remove linear trend from data."""
    slope, intercept = calculate_trend(values)
    x = np.arange(len(values))
    trend = slope * x + intercept
    return [v - t for v, t in zip(values, trend)]
