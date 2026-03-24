"""
Configuration settings for the Retail Price Optimization Dashboard.

This module contains all configuration parameters including:
- Application settings
- Data settings
- Model parameters
- Visualization options
- API configurations
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """Application configuration settings."""
    title: str = "Retail Price Analytics Dashboard"
    icon: str = "📊"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    version: str = "2.0.0"
    author: str = "Retail Analytics Team"


@dataclass
class DataConfig:
    """Data configuration settings."""
    data_file: str = "retail_price.csv"
    cache_ttl: int = 3600
    max_rows_display: int = 1000
    required_columns: List[str] = field(default_factory=lambda: [
        'product_id', 'product_category_name', 'month_year',
        'qty', 'total_price', 'unit_price', 'product_score'
    ])
    numeric_columns: List[str] = field(default_factory=lambda: [
        'qty', 'total_price', 'unit_price', 'product_score'
    ])
    feature_columns: List[str] = field(default_factory=lambda: [
        'qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff'
    ])


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    model_types: List[str] = field(default_factory=lambda: [
        'Decision Tree', 'Random Forest', 'Gradient Boosting',
        'Linear Regression', 'Ridge', 'Lasso'
    ])
    default_model: str = "Random Forest"
    default_test_size: float = 0.2
    default_random_state: int = 42
    cross_validation_folds: int = 5


@dataclass
class VisualizationConfig:
    """Visualization configuration settings."""
    chart_types: List[str] = field(default_factory=lambda: [
        'Histogram', 'Box Plot', 'Scatter Plot', 'Bar Chart',
        'Correlation Heatmap', 'Bar Chart - Price Difference',
        'Time Series Analysis', 'Price Distribution by Category'
    ])
    color_themes: Dict[str, str] = field(default_factory=lambda: {
        'Blue': '#667eea',
        'Purple': '#764ba2',
        'Pink': '#f093fb',
        'Green': '#4CAF50'
    })
    chart_height: int = 500
    default_theme: str = 'plotly_white'


@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_path: str = "retail_data.db"
    backup_dir: str = "backups"
    enable_logging: bool = True


class Config:
    """
    Main configuration class combining all settings.
    """

    # Application settings
    APP = AppConfig()

    # Data settings
    DATA = DataConfig()

    # Model settings
    MODEL = ModelConfig()

    # Visualization settings
    VIZ = VisualizationConfig()

    # API settings
    API = APIConfig()

    # Database settings
    DB = DatabaseConfig()

    # Validation thresholds
    MAX_PRICE_THRESHOLD: float = 10000.0
    MAX_QUANTITY_THRESHOLD: int = 1000
    MIN_QUALITY_SCORE: float = 70.0

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DIR: str = "logs"

    @classmethod
    def get_env_config(cls) -> Dict[str, str]:
        """Get configuration from environment variables."""
        return {
            'STREAMLIT_SERVER_PORT': os.getenv('STREAMLIT_SERVER_PORT', '8501'),
            'STREAMLIT_SERVER_ADDRESS': os.getenv('STREAMLIT_SERVER_ADDRESS', 'localhost'),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', cls.LOG_LEVEL),
            'API_PORT': os.getenv('API_PORT', str(cls.API.port)),
            'DB_PATH': os.getenv('DB_PATH', cls.DB.db_path)
        }

    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        try:
            # Check if data file exists
            if not os.path.exists(cls.DATA.data_file):
                print(f"Warning: Data file '{cls.DATA.data_file}' not found")
                return False

            # Validate numeric thresholds
            if cls.MAX_PRICE_THRESHOLD <= 0:
                raise ValueError("MAX_PRICE_THRESHOLD must be positive")

            if cls.MAX_QUANTITY_THRESHOLD <= 0:
                raise ValueError("MAX_QUANTITY_THRESHOLD must be positive")

            # Validate quality score threshold
            if not (0 <= cls.MIN_QUALITY_SCORE <= 100):
                raise ValueError("MIN_QUALITY_SCORE must be between 0 and 100")

            return True

        except Exception as e:
            print(f"Configuration validation error: {str(e)}")
            return False

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'app': {
                'title': cls.APP.title,
                'version': cls.APP.version,
                'layout': cls.APP.layout
            },
            'data': {
                'file': cls.DATA.data_file,
                'cache_ttl': cls.DATA.cache_ttl
            },
            'model': {
                'default_model': cls.MODEL.default_model,
                'test_size': cls.MODEL.default_test_size
            },
            'api': {
                'host': cls.API.host,
                'port': cls.API.port
            }
        }

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 50)
        print(f"  {cls.APP.title} v{cls.APP.version}")
        print("=" * 50)
        print(f"  Data file: {cls.DATA.data_file}")
        print(f"  Default model: {cls.MODEL.default_model}")
        print(f"  API port: {cls.API.port}")
        print("=" * 50)


# Create a global config instance
config = Config()
