"""
Encoding utilities for categorical data.
"""

def label_encode(categories):
    """Label encode categories."""
    unique = list(set(categories))
    mapping = {cat: i for i, cat in enumerate(unique)}
    return [mapping[cat] for cat in categories], mapping

def one_hot_encode(categories):
    """One-hot encode categories."""
    unique = list(set(categories))
    result = []
    for cat in categories:
        row = [1 if cat == u else 0 for u in unique]
        result.append(row)
    return result, unique
