"""
Langfuse tracing provider for Axion.

This module provides Langfuse-based tracing for LLM observability.

Example:
    from axion._core.tracing import configure_tracing, Tracer

    # Configure via environment or directly
    configure_tracing(tracing_mode='langfuse')

    tracer = Tracer('llm')
    with tracer.span('my-operation'):
        # your code here
        pass

    tracer.flush()
"""

from axion._core.tracing.langfuse.span import LangfuseSpan
from axion._core.tracing.langfuse.tracer import LangfuseTracer

__all__ = ['LangfuseTracer', 'LangfuseSpan']
