import threading
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from axion._core.tracing.registry import BaseTracer

# Global registry for notebook/cross-context compatibility
_global_tracer_registry: Dict[str, 'BaseTracer'] = {}
_registry_lock = threading.Lock()


def set_default_global_tracer(tracer: 'BaseTracer') -> None:
    """
    Set the default global tracer for notebook compatibility.

    This function registers a tracer globally so it can be accessed
    across different contexts, which is particularly useful in
    Jupyter notebooks where context variables may not propagate.

    Args:
        tracer: The tracer to set as default

    Example:
        ```
        tracer = init_tracer('llm')
        set_default_global_tracer(tracer)

        # Now all init_tracer() calls will use this tracer
        # even across notebook cells
        ```
    """
    with _registry_lock:
        _global_tracer_registry['default'] = tracer


def get_default_global_tracer() -> Optional['BaseTracer']:
    """
    Get the default global tracer.

    Returns:
        The default tracer, or None if not set

    Example:
        ```
        tracer = get_default_global_tracer()
        if tracer:
            # Use the tracer
            pass
        ```
    """
    with _registry_lock:
        return _global_tracer_registry.get('default')
