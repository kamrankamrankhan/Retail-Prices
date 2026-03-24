"""
Data aggregation utilities.
"""
import pandas as pd

def aggregate_by_column(df, column, agg_func='sum'):
    """Aggregate dataframe by column."""
    return df.groupby(column).agg(agg_func)

def pivot_data(df, index, columns, values):
    """Create pivot table."""
    return df.pivot_table(index=index, columns=columns, values=values)
