"""
Distance calculation utilities.
"""
import numpy as np

def euclidean_distance(a, b):
    """Calculate Euclidean distance."""
    return np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def manhattan_distance(a, b):
    """Calculate Manhattan distance."""
    return sum(abs(x - y) for x, y in zip(a, b))
