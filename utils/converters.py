"""
Data conversion utilities.
"""

def to_float(value, default=0.0):
    """Convert value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def to_int(value, default=0):
    """Convert value to int safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
