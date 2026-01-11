from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from axion._core.tracing.registry import BaseTracer

# Unified context variable for any tracer type
# This is main context for context aware tracing
_tracer_context: ContextVar[Optional['BaseTracer']] = ContextVar(
    'tracer', default=None
)


def get_current_tracer() -> 'BaseTracer':
    """
    Retrieves the current tracer instance from the context.

    This function works with any tracer type (LogfireTracer, NoOpTracer, etc.)
    and is the primary way to access the active tracer from within decorated
    functions or context-managed code blocks.
    """
    tracer = _tracer_context.get()
    if tracer is None:
        raise LookupError('No tracer found in the current context.')
    return tracer


def set_current_tracer(tracer: 'BaseTracer') -> Any:
    """
    Sets the current tracer in the context.
    """
    return _tracer_context.set(tracer)


def reset_tracer_context(token: Any) -> None:
    """
    Resets the tracer context using the provided token.
    """
    _tracer_context.reset(token)


def has_current_tracer() -> bool:
    """
    Check if there is a tracer in the current context.
    """
    return _tracer_context.get() is not None


def get_current_tracer_safe() -> Optional['BaseTracer']:
    """
    Safely retrieve the current tracer without raising exceptions.
    """
    return _tracer_context.get()
