"""
Data formatting utilities.
"""

def format_currency(value, symbol="$"):
    """Format value as currency."""
    return f"{symbol}{value:,.2f}"

def format_percentage(value):
    """Format value as percentage."""
    return f"{value:.1f}%"

def format_number(value):
    """Format number with commas."""
    return f"{value:,}"
