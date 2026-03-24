"""
Custom exceptions for the application.
"""

class PriceOptimizationError(Exception):
    """Base exception for price optimization."""
    pass

class DataValidationError(Exception):
    """Exception for data validation errors."""
    pass

class ModelNotTrainedError(Exception):
    """Exception when model is used before training."""
    pass
