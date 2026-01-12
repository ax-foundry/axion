"""
Public API for logging functionality.

This module re-exports all public APIs from axion._core.logging to provide
a cleaner import path: `from axion.logging import configure_logging, get_logger`

Public API:
    Core Functions:
        - configure_logging: Configure the application's logging system
        - get_logger: Get a logger instance
        - log_summary: Log a summary of all logging activity during the session

    Classes:
        - RichLogger: Custom logger class with enhanced logging methods
        - LoguruHandler: Handler that forwards records to loguru logger

    Convenience:
        - logger: Pre-configured logger instance for 'axion'
"""

from axion._core.logging import (
    configure_logging,
    get_logger,
    log_summary,
    RichLogger,
    LoguruHandler,
    logger,
)

__all__ = [
    'configure_logging',
    'get_logger',
    'log_summary',
    'RichLogger',
    'LoguruHandler',
    'logger',
]
