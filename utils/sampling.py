"""
Data sampling utilities.
"""
import random

def random_sample(data, n):
    """Random sample from data."""
    return random.sample(data, min(n, len(data)))

def stratified_sample(df, column, n_per_group):
    """Stratified sampling."""
    return df.groupby(column).apply(lambda x: x.sample(min(n_per_group, len(x))))
