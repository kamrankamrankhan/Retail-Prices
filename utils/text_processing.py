"""
Text processing utilities.
"""

def tokenize(text):
    """Simple tokenizer."""
    return text.lower().split()

def remove_punctuation(text):
    """Remove punctuation from text."""
    import string
    return text.translate(str.maketrans('', '', string.punctuation))

def count_words(text):
    """Count words in text."""
    return len(text.split())
