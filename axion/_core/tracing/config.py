from typing import Optional, Type, Union

from axion._core.environment import TracingMode, settings
from axion._core.logging import get_logger
from axion._core.tracing.registry import BaseTracer, TracerRegistry

logger = get_logger(__name__)

__all__ = [
    'configure_tracing',
    'get_tracer',
    'get_tracer_mode_param',
    'reset_tracing',
]

# Global configuration state
_tracer_instance: Optional[Type[BaseTracer]] = None
_tracing_configured: bool = False
_tracer_mode_param: Optional[str] = None  # For logfire mode (local/hosted/otel)


def _ensure_providers_registered() -> None:
    """
    Import tracer modules to trigger their registration with TracerRegistry.

    This function is called lazily when configure_tracing() is first invoked,
    ensuring all built-in providers are registered before use.
    """
    # Import each provider module - the @TracerRegistry.register decorator
    # will automatically register them when the module is imported
    try:
        from axion._core.tracing import noop  # noqa: F401 - registers 'noop'
    except ImportError as e:
        logger.debug(f'Failed to import noop tracer: {e}')

    try:
        from axion._core.tracing import logfire  # noqa: F401 - registers 'logfire'
    except ImportError as e:
        logger.debug(f'Failed to import logfire tracer: {e}')

    try:
        from axion._core.tracing import langfuse  # noqa: F401 - registers 'langfuse'
    except ImportError as e:
        logger.debug(f'Failed to import langfuse tracer: {e}')

    try:
        from axion._core.tracing import opik  # noqa: F401 - registers 'opik'
    except ImportError as e:
        logger.debug(f'Failed to import opik tracer: {e}')


# Map TracingMode enum to (registry_key, mode_param) tuples
# registry_key: The name used to look up the tracer in TracerRegistry
# mode_param: Optional parameter passed to the tracer for sub-modes (e.g., logfire local/hosted/otel)
_MODE_TO_REGISTRY = {
    TracingMode.NOOP: ('noop', None),
    TracingMode.LOGFIRE_LOCAL: ('logfire', 'local'),
    TracingMode.LOGFIRE_HOSTED: ('logfire', 'hosted'),
    TracingMode.LOGFIRE_OTEL: ('logfire', 'otel'),
    TracingMode.LANGFUSE: ('langfuse', None),
    TracingMode.OPIK: ('opik', None),
}


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
    global _tracer_instance, _tracing_configured, _tracer_mode_param
    logger.debug('Attempting to configure tracing...')

    # Ensure all providers are registered
    _ensure_providers_registered()

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

    # Look up the registry key and mode parameter for this TracingMode
    registry_info = _MODE_TO_REGISTRY.get(final_mode)
    if not registry_info:
        logger.error(
            f"No tracer implementation registered for mode '{final_mode}'. Using NOOP."
        )
        registry_key, mode_param = 'noop', None
        final_mode = TracingMode.NOOP
    else:
        registry_key, mode_param = registry_info

    # Get the tracer class from the registry
    try:
        TracerClass = TracerRegistry.get(registry_key)
    except ValueError as e:
        logger.error(f'{e}. Falling back to NOOP.')
        TracerClass = TracerRegistry.get('noop')
        final_mode = TracingMode.NOOP
        mode_param = None

    # Set the global tracer to the CLASS itself (not an instance)
    _tracer_instance = TracerClass
    _tracer_mode_param = mode_param
    setattr(_tracer_instance, '_tracing_mode', final_mode)

    _tracing_configured = True
    logger.debug(f"Tracing successfully configured to use mode: '{final_mode.value}'")


def get_tracer() -> Type[BaseTracer]:
    """
    Gets the configured tracer class (not instance).

    This function returns the tracer class that has been configured via
    configure_tracing(). If tracing hasn't been configured yet, it will
    apply a default configuration first.

    Returns:
        The configured tracer class (LogfireTracer, NoOpTracer, LangfuseTracer, etc.)

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


def get_tracer_mode_param() -> Optional[str]:
    """
    Gets the mode parameter for the current tracer configuration.

    For Logfire tracers, this returns 'local', 'hosted', or 'otel'.
    For other tracers, this returns None.

    Returns:
        The mode parameter string, or None if not applicable.
    """
    return _tracer_mode_param


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
        configure_tracing(tracing_mode='noop')
        ```
    """
    global _tracer_instance, _tracing_configured, _tracer_mode_param
    _tracer_instance = None
    _tracing_configured = False
    _tracer_mode_param = None
    logger.debug('Tracing configuration reset.')
