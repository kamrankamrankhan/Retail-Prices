"""
Time series decomposition utilities.
"""
import numpy as np

def decompose(values, period=12):
    """Decompose time series into trend, seasonal, residual."""
    n = len(values)
    
    # Trend
    trend = np.convolve(values, np.ones(period)/period, mode='same')
    
    # Seasonal
    detrended = [v - t for v, t in zip(values, trend)]
    seasonal = []
    for i in range(period):
        indices = range(i, n, period)
        seasonal.append(np.mean([detrended[j] for j in indices if j < len(detrended)]))
    seasonal = [seasonal[i % period] for i in range(n)]
    
    # Residual
    residual = [v - t - s for v, t, s in zip(values, trend, seasonal)]
    
    return trend, seasonal, residual
