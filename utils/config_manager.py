"""
Configuration Management Module.

Provides centralized configuration management with environment
variable support and validation.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager for application settings.

    Supports:
    - Environment variables
    - JSON configuration files
    - Default values
    - Configuration validation
    """

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager."""
        if not self._config:
            self._load_defaults()
            self._load_from_env()
            self._load_from_file()

    def _load_defaults(self):
        """Load default configuration values."""
        self._config = {
            # Application
            'app_name': 'Retail Price Optimization Dashboard',
            'app_version': '2.0.0',
            'debug_mode': False,

            # Server
            'streamlit_port': 8501,
            'api_port': 8000,
            'api_host': '0.0.0.0',

            # Data
            'data_file': 'retail_price.csv',
            'cache_ttl': 3600,
            'max_rows': 10000,

            # Model
            'default_model': 'random_forest',
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,

            # Logging
            'log_level': 'INFO',
            'log_dir': 'logs',
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',

            # Database
            'db_path': 'retail_data.db',
            'db_backup_dir': 'backups',

            # Visualization
            'chart_height': 500,
            'chart_theme': 'plotly_white',
            'color_palette': ['#667eea', '#764ba2', '#f093fb', '#f5576c']
        }

    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'STREAMLIT_SERVER_PORT': 'streamlit_port',
            'API_PORT': 'api_port',
            'API_HOST': 'api_host',
            'LOG_LEVEL': 'log_level',
            'DEBUG': 'debug_mode',
            'DATA_FILE': 'data_file',
            'DB_PATH': 'db_path'
        }

        for env_key, config_key in env_mappings.items():
            value = os.getenv(env_key)
            if value is not None:
                # Type conversion
                if config_key in ['streamlit_port', 'api_port']:
                    value = int(value)
                elif config_key == 'debug_mode':
                    value = value.lower() in ('true', '1', 'yes')

                self._config[config_key] = value
                logger.debug(f"Loaded {config_key} from environment: {value}")

    def _load_from_file(self):
        """Load configuration from JSON file if exists."""
        config_file = Path('config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    self._config.update(file_config)
                    logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Could not load config file: {str(e)}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()

    def validate(self) -> bool:
        """Validate configuration."""
        required_keys = ['data_file', 'streamlit_port', 'api_port']

        for key in required_keys:
            if key not in self._config or self._config[key] is None:
                logger.error(f"Missing required configuration: {key}")
                return False

        # Validate data file exists
        if not Path(self._config['data_file']).exists():
            logger.warning(f"Data file not found: {self._config['data_file']}")

        return True

    def save_to_file(self, filepath: str = 'config.json'):
        """Save current configuration to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Could not save configuration: {str(e)}")
            return False

    def reload(self):
        """Reload configuration from all sources."""
        self._config = {}
        self._load_defaults()
        self._load_from_env()
        self._load_from_file()
        logger.info("Configuration reloaded")


# Global config instance
config = ConfigManager()


def get_config(key: str, default: Any = None) -> Any:
    """Convenience function to get config value."""
    return config.get(key, default)
