import sys
from importlib import import_module
from typing import Optional, Type, Union

from axion._core.environment import TracingMode, settings
from axion._core.logging import get_logger
from axion._core.tracing.handlers import BaseTraceHandler

logger = get_logger(__name__)

# Global configuration state
_tracer_instance: Optional['BaseTraceHandler'] = None
_tracing_configured: bool = False


_TRACER_REGISTRY = {
    TracingMode.NOOP: 'axion._core.tracing.noop.NoOpTracer',
    TracingMode.LOGFIRE_LOCAL: 'axion._core.tracing.logfire.LogfireTracer',
    TracingMode.LOGFIRE_HOSTED: 'axion._core.tracing.logfire.LogfireTracer',
    TracingMode.LOGFIRE_OTEL: 'axion._core.tracing.logfire.LogfireTracer',
}


def _load_tracer_class(path: str) -> Type[BaseTraceHandler]:
    """
    Dynamically loads a tracer class from a given module path.
    """
    try:
        module_path, class_name = path.rsplit('.', 1)
        if module_path not in sys.modules:
            module = import_module(module_path)
        else:
            module = sys.modules[module_path]
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load tracer class from '{path}': {e}")
        # Fallback to NoOpTracer class
        from axion._core.tracing.noop import NoOpTracer

        return NoOpTracer


def configure_tracing(
    tracing_mode: Optional[Union[TracingMode, str]] = None, force: bool = False
) -> None:
    """
    Configures the application's tracing system based on the unified tracing_mode.

    Prioritizes a direct function argument over the global settings object.

    Args:
        tracing_mode: Override the tracing mode to use.
        force: If True, will overwrite an existing configuration.
    """
    global _tracer_instance, _tracing_configured
    logger.debug('Attempting to configure tracing...')

    # Resolve the final tracing mode, prioritizing the direct argument.
    final_mode_arg = tracing_mode or settings.tracing_mode

    # Convert string argument to Enum member if necessary
    try:
        final_mode = TracingMode(str(final_mode_arg).lower())
    except ValueError:
        logger.warning(
            f"Unknown tracing_mode '{final_mode_arg}' provided. "
            f"Falling back to global setting: '{settings.tracing_mode.value}'"
        )
        final_mode = settings.tracing_mode

    # Check if reconfiguration is necessary
    if _tracing_configured and not force:
        current_mode = getattr(_tracer_instance, '_tracing_mode', None)
        if current_mode == final_mode:
            logger.debug(
                f'Tracing already configured with mode "{final_mode}". Skipping.'
            )
            return
        else:
            logger.debug(
                f"Reconfiguring tracer from '{current_mode}' to '{final_mode}'"
            )

    # Look up and load the tracer class from the registry
    tracer_path = _TRACER_REGISTRY.get(final_mode)
    if not tracer_path:
        logger.error(
            f"No tracer implementation registered for mode '{final_mode}'. Using NOOP."
        )
        TracerClass = _load_tracer_class(_TRACER_REGISTRY[TracingMode.NOOP])
        final_mode = TracingMode.NOOP
    else:
        TracerClass = _load_tracer_class(tracer_path)

    # Set the global tracer to the CLASS itself (not an instance)
    _tracer_instance = TracerClass
    setattr(_tracer_instance, '_tracing_mode', final_mode)

    _tracing_configured = True
    logger.debug(f"Tracing successfully configured to use mode: '{final_mode.value}'")


def get_tracer() -> 'BaseTraceHandler':
    """
    Gets the configured tracer class (not instance).

    This function returns the tracer class that has been configured via
    configure_tracing(). If tracing hasn't been configured yet, it will
    apply a default configuration first.

    Returns:
        The configured tracer class (LogfireTracer, NoOpTracer, etc.)

    Example:
        ```
        TracerClass = get_tracer()
        tracer_instance = TracerClass.create(metadata_type='llm')
        ```
    """
    global _tracer_instance, _tracing_configured
    if not _tracing_configured:
        logger.debug('Tracing not configured. Applying default configuration.')
        configure_tracing()

    return _tracer_instance  # type: ignore


def reset_tracing() -> None:
    """
    Resets the tracing configuration.

    This function clears the global tracer configuration, allowing for
    fresh configuration. This is particularly useful for testing scenarios
    or when you need to completely reconfigure tracing from scratch.

    Example:
        ```
        # Reset and reconfigure
        reset_tracing()
        configure_tracing(tracer_type='noop')
        ```
    """
    global _tracer_instance, _tracing_configured
    _tracer_instance = None
    _tracing_configured = False
    logger.debug('Tracing configuration reset.')
