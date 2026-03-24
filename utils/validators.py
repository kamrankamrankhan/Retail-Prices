"""
Data validation utilities.
"""

def validate_price(price):
    """Validate that price is positive."""
    if price is None or price < 0:
        raise ValueError("Price must be positive")
    return True

def validate_quantity(qty):
    """Validate quantity is non-negative integer."""
    if qty is None or qty < 0:
        raise ValueError("Quantity must be non-negative")
    return True
