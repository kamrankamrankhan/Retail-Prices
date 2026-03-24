"""
Data grouping utilities.
"""

def group_and_count(df, column):
    """Group by column and count."""
    return df.groupby(column).size()

def group_and_sum(df, group_col, sum_col):
    """Group by column and sum another column."""
    return df.groupby(group_col)[sum_col].sum()
