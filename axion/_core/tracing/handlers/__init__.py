"""Backward compatibility layer for handlers module.

The handler classes have been removed. Statistics and logging logic
now lives directly in the tracer implementations and metadata classes.

This module re-exports BaseTracer as BaseTraceHandler for backward
compatibility with existing code.
"""

from axion._core.tracing.registry import BaseTracer
from axion._core.tracing.statistics import (
    MAX_ARG_LENGTH,
    MAX_ERROR_LENGTH,
    MAX_RESULT_LENGTH,
)

# Backward compatibility: BaseTraceHandler was used as a type hint for tracers
# Now use BaseTracer directly, but we keep this alias for backward compatibility
BaseTraceHandler = BaseTracer

# TracerProtocol is no longer needed - use BaseTracer instead
TracerProtocol = BaseTracer

__all__ = [
    'BaseTraceHandler',
    'TracerProtocol',
    'MAX_ERROR_LENGTH',
    'MAX_ARG_LENGTH',
    'MAX_RESULT_LENGTH',
]
