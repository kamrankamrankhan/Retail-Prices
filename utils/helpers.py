"""
Helper functions for retail price calculations.
"""

def calculate_margin(price, cost):
    """Calculate profit margin."""
    if cost == 0:
        return 0
    return (price - cost) / price * 100


def calculate_markup(price, cost):
    """Calculate price markup."""
    if cost == 0:
        return 0
    return (price - cost) / cost * 100
