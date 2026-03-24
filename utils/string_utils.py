"""
String manipulation utilities.
"""

def clean_string(s):
    """Clean and normalize string."""
    if s is None:
        return ""
    return str(s).strip().lower()

def truncate_string(s, max_length=100):
    """Truncate string to max length."""
    if len(s) <= max_length:
        return s
    return s[:max_length] + "..."
