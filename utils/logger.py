"""
Logging utilities for the Retail Price Optimization Dashboard.

This module provides a centralized logging configuration for consistent
log formatting and handling across all components of the application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import contextmanager
import os


def setup_logger(
    name: str = 'retail_optimizer',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name: Logger name for identification
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        format_string: Custom format string for log messages
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger('my_module', logging.DEBUG)
        >>> logger.info("Application started")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Default format
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggerContext:
    """
    Context manager for temporary logging configuration.
    
    Allows temporary modification of logging levels within a context.
    
    Example:
        >>> with LoggerContext('my_module', logging.DEBUG):
        ...     # Debug logging enabled in this block
        ...     pass
    """
    
    def __init__(self, logger_name: str, temp_level: int):
        """
        Initialize context manager.
        
        Args:
            logger_name: Name of the logger to modify
            temp_level: Temporary logging level
        """
        self.logger = logging.getLogger(logger_name)
        self.temp_level = temp_level
        self.original_level = self.logger.level
    
    def __enter__(self):
        """Enter context with temporary level."""
        self.logger.setLevel(self.temp_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original level."""
        self.logger.setLevel(self.original_level)
        return False


class PerformanceLogger:
    """
    Logger for tracking performance metrics and timing.
    
    Useful for benchmarking operations and tracking execution times.
    
    Example:
        >>> perf = PerformanceLogger('benchmark')
        >>> with perf.measure('data_processing'):
        ...     # Process data
        ...     pass
    """
    
    def __init__(self, name: str = 'performance'):
        """
        Initialize performance logger.
        
        Args:
            name: Logger name for performance tracking
        """
        self.logger = setup_logger(name, logging.INFO)
        self.metrics = {}
    
    @contextmanager
    def measure(self, operation_name: str):
        """
        Context manager to measure operation duration.
        
        Args:
            operation_name: Name of the operation to measure
        
        Yields:
            None
        """
        import time
        start_time = time.time()
        yield
        duration = time.time() - start_time
        self.metrics[operation_name] = duration
        self.logger.info(f"{operation_name} completed in {duration:.4f}s")
    
    def get_metrics(self) -> dict:
        """
        Get all recorded performance metrics.
        
        Returns:
            Dictionary of operation names and their durations
        """
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all recorded metrics."""
        self.metrics.clear()


def get_logger(module_name: str) -> logging.Logger:
    """
    Get or create a logger for a module.
    
    Args:
        module_name: Name of the module requesting the logger
    
    Returns:
        Logger instance for the module
    """
    return setup_logger(f'retail_optimizer.{module_name}')


def configure_root_logger(
    level: int = logging.INFO,
    log_dir: Optional[str] = None
) -> None:
    """
    Configure the root logger for the entire application.
    
    Args:
        level: Logging level for root logger
        log_dir: Directory for log files (optional)
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)
    
    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_file = log_path / f"retail_optimizer_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# Default logger instance
default_logger = setup_logger()


if __name__ == '__main__':
    # Demo usage
    logger = setup_logger('demo', logging.DEBUG)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Performance logging demo
    perf = PerformanceLogger('demo_perf')
    import time
    with perf.measure('test_operation'):
        time.sleep(0.1)
    
    print(f"Metrics: {perf.get_metrics()}")
