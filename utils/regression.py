"""
Regression analysis utilities.
"""
import numpy as np

def linear_regression(x, y):
    """Simple linear regression."""
    coefficients = np.polyfit(x, y, 1)
    return coefficients[0], coefficients[1]  # slope, intercept

def predict(slope, intercept, x):
    """Predict using linear model."""
    return slope * x + intercept
