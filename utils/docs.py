"""
Documentation Module.

Provides comprehensive documentation generation for the Retail Price Optimization Dashboard.
"""

import os
from typing import Dict, List


def get_module_docs():
    """Get documentation for all modules."""
    return {
        'models': 'Machine Learning models for price prediction',
        'utils': 'Utility functions for data processing and analysis',
        'api': 'REST API endpoints for external integrations',
        'pages': 'Streamlit dashboard pages'
    }


def generate_api_docs():
    """Generate API documentation."""
    return """
    # API Documentation

    ## Endpoints

    - GET /health - Health check
    - POST /predict - Price prediction
    - POST /optimize - Price optimization
    - GET /products - List products
    - GET /categories - List categories
    """
