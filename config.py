"""
Configuration settings for the Retail Price Optimization Dashboard.
"""

import os
from typing import Dict, List

class Config:
    """Configuration class for application settings."""
    
    # Application settings
    APP_TITLE = "Retail Price Analytics Dashboard"
    APP_ICON = "📊"
    PAGE_LAYOUT = "wide"
    SIDEBAR_STATE = "expanded"
    
    # Data settings
    DATA_FILE = "retail_price.csv"
    CACHE_TTL = 3600  # Cache time-to-live in seconds
    
    # Required columns for data validation
    REQUIRED_COLUMNS = [
        'product_id', 'product_category_name', 'month_year', 
        'qty', 'total_price', 'unit_price', 'product_score'
    ]
    
    # Numeric columns for validation
    NUMERIC_COLUMNS = ['qty', 'total_price', 'unit_price', 'product_score']
    
    # Feature columns for machine learning
    FEATURE_COLUMNS = ['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']
    
    # Model settings
    MODEL_TYPES = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
    DEFAULT_TEST_SIZE = 20
    DEFAULT_RANDOM_STATE = 42
    
    # Visualization settings
    CHART_TYPES = [
        'Histogram', 'Box Plot', 'Scatter Plot', 'Bar Chart', 
        'Correlation Heatmap', 'Bar Chart - Price Difference',
        'Time Series Analysis', 'Price Distribution by Category'
    ]
    
    # Color themes
    COLOR_THEMES = {
        'Blue': '#667eea',
        'Purple': '#764ba2', 
        'Pink': '#f093fb',
        'Green': '#4CAF50'
    }
    
    # Data validation thresholds
    MAX_PRICE_THRESHOLD = 10000
    MAX_QUANTITY_THRESHOLD = 1000
    MIN_QUALITY_SCORE = 70
    
    # Performance settings
    MAX_ROWS_DISPLAY = 1000
    CHART_HEIGHT = 500
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_env_config(cls) -> Dict[str, str]:
        """Get configuration from environment variables."""
        return {
            'STREAMLIT_SERVER_PORT': os.getenv('STREAMLIT_SERVER_PORT', '8501'),
            'STREAMLIT_SERVER_ADDRESS': os.getenv('STREAMLIT_SERVER_ADDRESS', 'localhost'),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', cls.LOG_LEVEL)
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        try:
            # Check if data file exists
            if not os.path.exists(cls.DATA_FILE):
                return False
            
            # Validate numeric thresholds
            if cls.MAX_PRICE_THRESHOLD <= 0 or cls.MAX_QUANTITY_THRESHOLD <= 0:
                return False
            
            # Validate quality score threshold
            if not (0 <= cls.MIN_QUALITY_SCORE <= 100):
                return False
            
            return True
        except Exception:
            return False