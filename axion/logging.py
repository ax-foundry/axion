"""
Public API for logging functionality.

This module provides zero-config logging with automatic configuration.
Just use get_logger() and it auto-configures from environment variables.

Quick Start:
    >>> from axion.logging import get_logger
    >>> logger = get_logger(__name__)  # Auto-configures from env vars
    >>> logger.info("Hello world")

Environment Variables:
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
    - LOG_USE_RICH: Enable rich formatting (true/false); unset = only on a TTY
    - LOG_JSON: One JSON object per record (true/false); unset = when rich is also unset and stderr is not a TTY
    - LOG_FILE: Optional log file path

Public API:
    Core Functions:
        - configure_logging: Configure logging (optional, auto-configures on first use)
        - get_logger: Get a logger instance
        - is_logging_configured: Check if logging has been configured
        - clear_logging_config: Clear configuration (useful for testing)
        - log_summary: Log a summary of all logging activity during the session

    Classes:
        - RichLogger: Custom logger class with enhanced logging methods
        - LoguruHandler: Handler that forwards records to loguru logger

    Convenience:
        - logger: Pre-configured logger instance for 'axion'
"""

from axion._core.logging import (
    JsonFormatter,
    LoguruHandler,
    RichLogger,
    clear_logging_config,
    configure_logging,
    get_logger,
    is_logging_configured,
    log_summary,
    logger,
)

__all__ = [
    'clear_logging_config',
    'JsonFormatter',
    'configure_logging',
    'get_logger',
    'is_logging_configured',
    'log_summary',
    'logger',
    'LoguruHandler',
    'RichLogger',
]
