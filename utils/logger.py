"""
Logging Utilities Module.

This module provides logging configuration and utilities for
the Retail Price Optimization Dashboard.
"""

import logging
import os
from datetime import datetime
from typing import Optional
import sys


def setup_logger(name: str = 'retail_price',
                 level: int = logging.INFO,
                 log_to_file: bool = True,
                 log_dir: str = 'logs') -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        level: Logging level
        log_to_file: Whether to log to file
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class LoggerAdapter:
    """Adapter for adding context to log messages."""

    def __init__(self, logger: logging.Logger, context: dict):
        self.logger = logger
        self.context = context

    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def _log(self, level: int, msg: str, *args, **kwargs):
        context_str = ' | '.join(f'{k}={v}' for k, v in self.context.items())
        full_msg = f"[{context_str}] {msg}"
        self.logger.log(level, full_msg, *args, **kwargs)
