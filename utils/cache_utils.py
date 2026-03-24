"""
Caching utilities.
"""
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_computation(key):
    """Example of cached computation."""
    return key * 2

def clear_cache():
    """Clear the cache."""
    cached_computation.cache_clear()
