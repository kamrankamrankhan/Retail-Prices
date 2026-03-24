"""
Data sorting utilities.
"""

def sort_by_column(df, column, ascending=True):
    """Sort dataframe by column."""
    return df.sort_values(column, ascending=ascending)

def sort_by_multiple(df, columns, ascending=None):
    """Sort dataframe by multiple columns."""
    if ascending is None:
        ascending = [True] * len(columns)
    return df.sort_values(columns, ascending=ascending)
