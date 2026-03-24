"""
File handling utilities.
"""
import os

def ensure_directory(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
    return path

def get_file_extension(filename):
    """Get file extension."""
    return os.path.splitext(filename)[1]
