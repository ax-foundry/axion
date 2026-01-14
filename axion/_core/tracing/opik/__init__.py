"""
Opik (Comet) tracing provider for Axion.

This module provides Opik-based tracing for LLM observability.

Example:
    from axion._core.tracing import configure_tracing, Tracer

    # Configure via environment or directly
    configure_tracing(tracing_mode='opik')

    tracer = Tracer('llm')
    with tracer.span('my-operation'):
        # your code here
        pass

    tracer.flush()
"""

from axion._core.tracing.opik.span import OpikSpan
from axion._core.tracing.opik.tracer import OPIK_AVAILABLE, OpikTracer

__all__ = ['OpikTracer', 'OpikSpan', 'OPIK_AVAILABLE']
