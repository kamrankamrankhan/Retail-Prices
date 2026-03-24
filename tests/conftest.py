"""
Pytest configuration and fixtures for Retail Price Optimization tests.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_data():
    """Create sample retail data for testing."""
    np.random.seed(42)
    n = 100

    data = pd.DataFrame({
        'product_id': [f'p{i}' for i in range(n)],
        'product_category_name': np.random.choice(['cat1', 'cat2', 'cat3'], n),
        'month_year': pd.date_range('2023-01-01', periods=n, freq='D').strftime('%d-%m-%Y'),
        'qty': np.random.randint(1, 50, n),
        'total_price': np.random.uniform(50, 500, n),
        'freight_price': np.random.uniform(5, 30, n),
        'unit_price': np.random.uniform(10, 100, n),
        'product_score': np.random.uniform(3, 5, n),
        'customers': np.random.randint(1, 20, n),
        'comp_1': np.random.uniform(8, 95, n),
        'comp_2': np.random.uniform(8, 95, n),
        'comp_3': np.random.uniform(8, 95, n),
    })

    return data


@pytest.fixture
def trained_model(sample_data):
    """Create and train a model for testing."""
    from models.price_predictor import PricePredictor

    sample_data['comp_price_diff'] = sample_data['unit_price'] - sample_data['comp_1']

    predictor = PricePredictor(model_type='random_forest')
    predictor.fit(sample_data)

    return predictor


@pytest.fixture
def price_optimizer(sample_data):
    """Create a price optimizer for testing."""
    from utils.price_optimizer import PriceOptimizer

    return PriceOptimizer(sample_data)


@pytest.fixture
def preprocessor(sample_data):
    """Create a data preprocessor for testing."""
    from utils.data_preprocessing import DataPreprocessor

    return DataPreprocessor(sample_data)


@pytest.fixture
def analytics_engine(sample_data):
    """Create an analytics engine for testing."""
    from utils.analytics import AnalyticsEngine

    return AnalyticsEngine(sample_data)
