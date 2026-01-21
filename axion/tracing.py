"""
Public API for tracing functionality.

This module provides zero-config tracing with automatic provider detection.
Just use Tracer() and it auto-configures from environment variables.

Quick Start:
    >>> from axion.tracing import Tracer
    >>> tracer = Tracer('llm')  # Auto-configures from env vars
    >>> with tracer.span('my-operation'):
    ...     pass

Environment Variables:
    - TRACING_MODE: Provider selection (noop, logfire, otel, langfuse, opik)
    - Auto-detects from: LANGFUSE_SECRET_KEY, OPIK_API_KEY, LOGFIRE_TOKEN

Public API:
    Core Functions:
        - configure_tracing: Configure tracing (optional, auto-configures on first use)
        - list_providers: List available tracing providers
        - is_tracing_configured: Check if tracing has been configured
        - clear_tracing_config: Clear configuration (useful for testing)
        - get_tracer: Get the configured tracer class
        - init_tracer: Initialize a tracer instance
        - Tracer: Factory function for tracer instances

    Context Management:
        - get_current_tracer: Get active tracer from context
        - set_current_tracer: Set tracer in context
        - reset_tracer_context: Reset tracer context

    Decorators:
        - trace: Method decorator for automatic tracing

    Utilities:
        - infer_tool_metadata: Auto-generate metadata from call stack
        - set_default_global_tracer: Set global tracer (notebook compatibility)
        - get_default_global_tracer: Get global tracer (notebook compatibility)

    Types:
        - TracingMode: Enumeration of available tracer modes
        - TraceGranularity: Controls trace granularity during evaluation
        - TracerRegistry: Registry for tracer implementations
        - BaseTracer: Abstract base class for tracer implementations
"""

from axion._core.tracing import (
    BaseTracer,
    TraceGranularity,
    Tracer,
    # Registry and base classes
    TracerRegistry,
    TracingMode,
    # Core configuration
    clear_tracing_config,
    configure_tracing,
    # Context management
    get_current_tracer,
    get_default_global_tracer,
    get_tracer,
    get_tracer_mode_param,
    infer_tool_metadata,
    # Factory
    init_tracer,
    is_tracing_configured,
    list_providers,
    reset_tracer_context,
    set_current_tracer,
    # Utilities
    set_default_global_tracer,
    # Decorators
    trace,
    trace_function,
    trace_method,
)
from axion._core.tracing.loaders import (
    BaseTraceLoader,
    FetchedTraceData,
    LangfuseTraceLoader,
    LogfireTraceLoader,
    OpikTraceLoader,
)

__all__ = [
    # Core configuration
    'clear_tracing_config',
    'configure_tracing',
    'get_tracer',
    'get_tracer_mode_param',
    'is_tracing_configured',
    'list_providers',
    'TracingMode',
    'TraceGranularity',
    # Registry and base classes
    'TracerRegistry',
    'BaseTracer',
    # Context management
    'get_current_tracer',
    'set_current_tracer',
    'reset_tracer_context',
    # Factory
    'init_tracer',
    'Tracer',
    'infer_tool_metadata',
    # Decorators
    'trace',
    'trace_function',
    'trace_method',
    # Utilities
    'set_default_global_tracer',
    'get_default_global_tracer',
    # Trace loaders
    'FetchedTraceData',
    'BaseTraceLoader',
    'LangfuseTraceLoader',
    'OpikTraceLoader',
    'LogfireTraceLoader',
]
