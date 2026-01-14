"""
Public API:
    Core Functions:
        - configure_tracing: Configure the global tracing system
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
        - reset_tracing: Reset configuration (useful for testing)
        - set_default_global_tracer: Set global tracer (notebook compatibility)
        - get_default_global_tracer: Get global tracer (notebook compatibility)

    Types:
        - TracingMode: Enumeration of available tracer modes
        - TracerRegistry: Registry for tracer implementations
        - BaseTracer: Abstract base class for tracer implementations
"""

from axion._core.environment import TracingMode
from axion._core.tracing.config import (
    clear_tracing_config,
    configure_tracing,
    get_tracer,
    get_tracer_mode_param,
    is_tracing_configured,
    list_providers,
)
from axion._core.tracing.context import (
    get_current_tracer,
    reset_tracer_context,
    set_current_tracer,
)
from axion._core.tracing.decorators import trace, trace_function, trace_method
from axion._core.tracing.factory import Tracer, infer_tool_metadata, init_tracer
from axion._core.tracing.registry import BaseTracer, TracerRegistry
from axion._core.tracing.utils import (
    get_default_global_tracer,
    set_default_global_tracer,
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
]


__version__ = '1.0.0'
__author__ = 'AI Engineering'
