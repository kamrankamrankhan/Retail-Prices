"""
Data smoothing utilities.
"""

def moving_average(values, window=3):
    """Calculate moving average."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result

def exponential_smoothing(values, alpha=0.3):
    """Calculate exponential smoothing."""
    result = [values[0]]
    for i in range(1, len(values)):
        result.append(alpha * values[i] + (1 - alpha) * result[-1])
    return result
