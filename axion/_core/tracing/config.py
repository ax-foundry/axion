from typing import Optional, Type

from axion._core.environment import detect_tracing_provider, list_tracing_providers
from axion._core.logging import get_logger
from axion._core.tracing.registry import BaseTracer, TracerRegistry

logger = get_logger(__name__)

__all__ = [
    'clear_tracing_config',
    'configure_tracing',
    'get_tracer',
    'get_tracer_mode_param',
    'is_tracing_configured',
    'list_providers',
]

# Global configuration state
_tracer_instance: Optional[Type[BaseTracer]] = None
_tracing_configured: bool = False
_tracer_mode_param: Optional[str] = None


def _ensure_providers_registered() -> None:
    """
    Import tracer modules to trigger their registration with TracerRegistry.

    This function is called lazily when configure_tracing() is first invoked,
    ensuring all built-in providers are registered before use.
    """
    try:
        from axion._core.tracing import noop  # noqa: F401
    except ImportError as e:
        logger.debug(f'Failed to import noop tracer: {e}')

    try:
        from axion._core.tracing import logfire  # noqa: F401
    except ImportError as e:
        logger.debug(f'Failed to import logfire tracer: {e}')

    try:
        from axion._core.tracing import langfuse  # noqa: F401
    except ImportError as e:
        logger.debug(f'Failed to import langfuse tracer: {e}')

    try:
        from axion._core.tracing import opik  # noqa: F401
    except ImportError as e:
        logger.debug(f'Failed to import opik tracer: {e}')


# Map provider strings to (registry_key, mode_param) tuples
_PROVIDER_TO_REGISTRY = {
    'noop': ('noop', None),
    'logfire': ('logfire', None),
    'otel': ('logfire', 'otel'),
    'langfuse': ('langfuse', None),
    'opik': ('opik', None),
}


def configure_tracing(provider: Optional[str] = None) -> None:
    """
    Configure the tracing provider.

    Called automatically on first use of Tracer() or get_tracer().
    Call explicitly to switch providers at any time.

    Args:
        provider: Provider name. One of: 'noop', 'logfire', 'otel', 'langfuse', 'opik'.
            If None, auto-detects from environment variables.

    Auto-Detection Priority:
        1. TRACING_MODE env var
        2. LANGFUSE_SECRET_KEY → 'langfuse'
        3. OPIK_API_KEY → 'opik'
        4. LOGFIRE_TOKEN → 'logfire'
        5. OTEL_EXPORTER_OTLP_ENDPOINT → 'otel'
        6. Default: 'noop'

    Example:
        >>> configure_tracing(provider='langfuse')
        >>> # Or just use Tracer() directly - it auto-configures
        >>> tracer = Tracer('llm')
        >>> # Can switch providers at any time
        >>> configure_tracing(provider='opik')
    """
    global _tracer_instance, _tracing_configured, _tracer_mode_param
    logger.debug('Configuring tracing...')

    _ensure_providers_registered()

    # Resolve provider
    final_provider = provider.lower() if provider else detect_tracing_provider()

    # Validate
    valid_providers = list_tracing_providers()
    if final_provider not in valid_providers:
        logger.warning(
            f"Unknown provider '{final_provider}'. "
            f"Available: {valid_providers}. Using 'noop'."
        )
        final_provider = 'noop'

    # Skip if already configured with same provider
    if _tracing_configured:
        current = getattr(_tracer_instance, '_tracing_provider', None)
        if current == final_provider:
            logger.debug(f'Already configured with "{final_provider}". Skipping.')
            return

    # Look up registry
    registry_key, mode_param = _PROVIDER_TO_REGISTRY.get(final_provider, ('noop', None))

    # Get tracer class
    try:
        TracerClass = TracerRegistry.get(registry_key)
    except ValueError as e:
        logger.error(f'{e}. Falling back to noop.')
        TracerClass = TracerRegistry.get('noop')
        final_provider = 'noop'
        mode_param = None

    # Set global state
    _tracer_instance = TracerClass
    _tracer_mode_param = mode_param
    setattr(_tracer_instance, '_tracing_provider', final_provider)
    _tracing_configured = True

    logger.debug(f"Tracing configured: '{final_provider}'")


def get_tracer() -> Type[BaseTracer]:
    """
    Get the configured tracer class.

    Auto-configures if not already configured.

    Returns:
        The tracer class (not an instance).

    Example:
        >>> TracerClass = get_tracer()
        >>> tracer = TracerClass.create(metadata_type='llm')
    """
    if not _tracing_configured:
        configure_tracing()
    return _tracer_instance  # type: ignore


def get_tracer_mode_param() -> Optional[str]:
    """Get the mode parameter for logfire tracer ('otel' or None)."""
    return _tracer_mode_param


def is_tracing_configured() -> bool:
    """
    Check if tracing has been configured.

    Returns:
        True if configure_tracing() has been called (explicitly or automatically).

    Example:
        >>> is_tracing_configured()
        False
        >>> tracer = Tracer('llm')  # Auto-configures
        >>> is_tracing_configured()
        True
    """
    return _tracing_configured


def clear_tracing_config() -> None:
    """
    Clear the tracing configuration.

    Resets to unconfigured state. Next Tracer() call will re-configure.

    Example:
        >>> clear_tracing_config()
        >>> configure_tracing(provider='opik')
    """
    global _tracer_instance, _tracing_configured, _tracer_mode_param
    _tracer_instance = None
    _tracing_configured = False
    _tracer_mode_param = None
    logger.debug('Tracing configuration cleared.')


def list_providers() -> list[str]:
    """
    List available tracing providers.

    Returns:
        ['noop', 'logfire', 'otel', 'langfuse', 'opik']
    """
    return list_tracing_providers()
