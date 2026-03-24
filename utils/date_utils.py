"""
Date handling utilities.
"""
from datetime import datetime, timedelta

def get_current_date():
    """Get current date."""
    return datetime.now().strftime('%Y-%m-%d')

def get_date_range(days):
    """Get date range for specified days."""
    end = datetime.now()
    start = end - timedelta(days=days)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
