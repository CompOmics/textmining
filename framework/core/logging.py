"""
Centralized logging configuration for the extraction framework.

This module provides a consistent logging setup across all modules with:
- Configurable log level (via LOG_LEVEL env var or explicitly)
- Structured format with timestamps and module names
- Optional file output
- Colored console output

Usage:
    from core.logging import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing file: %s", filename)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional


# ============================================================================
# Configuration
# ============================================================================

# Default log level (can be overridden by LOG_LEVEL env var)
DEFAULT_LOG_LEVEL = "INFO"

# Log format with timestamp, level, module, and message
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
LOG_FORMAT_SIMPLE = "[%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File logging
LOG_FILE_NAME = "extraction_framework.log"


# ============================================================================
# Color Support
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, fmt: str = None, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the record with optional colors."""
        message = super().format(record)
        
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            # Color just the level name
            message = message.replace(
                f"[{record.levelname}]",
                f"{color}[{record.levelname}]{self.RESET}"
            )
        
        return message


# ============================================================================
# Logger Setup
# ============================================================================

def get_log_level() -> int:
    """Get log level from environment or default."""
    level_name = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.INFO)


def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    use_colors: bool = True,
    simple_format: bool = False,
) -> None:
    """
    Configure logging for the entire framework.
    
    Args:
        level: Log level (default: from LOG_LEVEL env var or INFO)
        log_file: Optional file to write logs to
        use_colors: Use colored output in console
        simple_format: Use simpler format without timestamps
    """
    if level is None:
        level = get_log_level()
    
    fmt = LOG_FORMAT_SIMPLE if simple_format else LOG_FORMAT
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(fmt, DATE_FORMAT, use_colors))
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, DATE_FORMAT))
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Ensure basic setup if not already configured
    if not logging.root.handlers:
        setup_logging()
    
    return logger


# ============================================================================
# Convenience Functions
# ============================================================================

def set_level(level: str) -> None:
    """
    Set the log level for all loggers.
    
    Args:
        level: Level name ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.root.setLevel(numeric_level)
    for handler in logging.root.handlers:
        handler.setLevel(numeric_level)


def enable_debug() -> None:
    """Enable debug logging."""
    set_level('DEBUG')


def enable_quiet() -> None:
    """Enable quiet mode (warnings and errors only)."""
    set_level('WARNING')


# ============================================================================
# Module-level setup
# ============================================================================

# Auto-setup with defaults when module is imported
# This ensures consistent logging across all modules
if not logging.root.handlers:
    setup_logging()
