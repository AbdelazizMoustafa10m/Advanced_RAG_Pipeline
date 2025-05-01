# --- utils/logging.py ---

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from core.config import LoggingConfig


def setup_logging(config: Optional[LoggingConfig] = None):
    """Set up logging configuration.
    
    Args:
        config: Optional logging configuration
    """
    if config is None:
        config = LoggingConfig()
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if enabled
    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(config.level)
        console_formatter = logging.Formatter(config.format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if file path is provided
    if config.file_path:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(config.file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Set up rotating file handler
        file_handler = RotatingFileHandler(
            filename=config.file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(config.level)
        file_formatter = logging.Formatter(config.format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Log setup completion
    logging.info(f"Logging initialized with level {config.level}")