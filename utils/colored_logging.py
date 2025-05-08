"""
Colored logging utility for the Advanced RAG Pipeline.
Provides colorized output for different log levels:
- INFO: Green
- WARNING: Yellow
- ERROR: Red
- CRITICAL: Bold Red
- DEBUG: Blue
"""

import logging
import sys

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
BRIGHT_RED = "\033[91m"

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels
    """
    COLORS = {
        'DEBUG': BLUE,
        'INFO': GREEN,
        'WARNING': YELLOW,
        'ERROR': RED,
        'CRITICAL': BOLD + BRIGHT_RED
    }

    def format(self, record):
        # Save the original levelname
        levelname = record.levelname
        # Add color to the levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{RESET}"
        # Format the message using the parent formatter
        result = super().format(record)
        # Restore the original levelname
        record.levelname = levelname
        return result

def setup_colored_logging(level=logging.INFO, format_str=None):
    """
    Set up colored logging with the specified level and format.
    
    Args:
        level: The logging level (default: logging.INFO)
        format_str: The log format string (default: '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create a colored formatter
    colored_formatter = ColoredFormatter(format_str)
    
    # Create a handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(colored_formatter)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our colored handler
    root_logger.addHandler(console_handler)
    
    return root_logger
