"""
Data filtering utilities.
"""

def filter_by_range(df, column, min_val, max_val):
    """Filter dataframe by value range."""
    return df[(df[column] >= min_val) & (df[column] <= max_val)]

def filter_by_values(df, column, values):
    """Filter dataframe by list of values."""
    return df[df[column].isin(values)]
