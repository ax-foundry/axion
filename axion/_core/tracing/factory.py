import inspect
from typing import TYPE_CHECKING, Optional

from axion._core.metadata.schema import ToolMetadata
from axion._core.tracing.config import get_tracer
from axion._core.tracing.context import get_current_tracer

if TYPE_CHECKING:
    from axion._core.tracing.registry import BaseTracer


def infer_tool_metadata(
    name_prefix: str = 'tracer_', stack_level: int = 2
) -> ToolMetadata:
    """
    Creates descriptive default ToolMetadata by inspecting the call stack.

    This helper function identifies the module that called it, making it easy
    to see where a component (like a tracer) was initialized without
    explicit metadata.

    Args:
        name_prefix: A prefix to use for the generated tool name
        stack_level: How many frames to go up the stack to find the
                    original caller. The default of 2 typically points
                    to the code that called the function containing this helper

    Returns:
        A ToolMetadata object with inferred details
    """
    try:
        # Go up the stack to find the original caller's frame
        frame = inspect.currentframe()
        # Move up the stack by the specified level to get to the caller
        caller_frame = frame
        for _ in range(stack_level):
            if caller_frame.f_back:
                caller_frame = caller_frame.f_back
            else:
                break

        module_name = caller_frame.f_globals.get('__name__', 'unknown_module')

        # Generate a descriptive name and description
        name = f'{name_prefix}_{module_name.split(".")[-1]}'
        description = f'Auto-generated metadata for module: {module_name}'

    except Exception:
        # Fallback if stack inspection fails for any reason
        name = f'{name_prefix}_uninspected'
        description = 'Auto-generated metadata (stack inspection failed)'
    finally:
        # Clean up frame references to prevent potential reference cycles
        if 'frame' in locals():
            del frame
        if 'caller_frame' in locals():
            del caller_frame

    return ToolMetadata(
        name=name,
        description=description,
        owner='system (auto-generated)',
        version='1.0.0',
    )


def init_tracer(
    metadata_type: str,
    tool_metadata: Optional[ToolMetadata] = None,
    tracer: Optional['BaseTracer'] = None,
) -> 'BaseTracer':
    """
    Initializes and returns a tracer instance, respecting global configuration.

    This helper function streamlines tracer initialization by following a
    standard pattern:
    1. Use an explicitly provided tracer if available
    2. Fall back to an active tracer from the current context
    3. Try the global registry (notebook-friendly fallback)
    4. As a last resort, create a new tracer using the globally configured
       tracer class from the get_tracer() factory

    Args:
        metadata_type: The type of metadata to use if a new tracer is created
                      (e.g., 'llm', 'base', 'database', 'knowledge')
        tool_metadata: Optional tool-specific metadata for a new tracer
        tracer: An optional, explicit tracer instance to use directly

    Returns:
        The resolved tracer instance (LogfireTracer, NoOpTracer, etc.)

    Example:
        ```python
        # Create a new tracer for LLM operations
        tracer = init_tracer('llm', tool_metadata=my_metadata)

        # Use explicit tracer
        tracer = init_tracer('llm', tracer=existing_tracer)

        # Within a context, gets the current tracer
        async with some_tracer.async_span("operation"):
            nested_tracer = init_tracer('llm')  # Returns some_tracer

        # Notebook-friendly usage
        from axion._core.tracing import set_default_global_tracer
        set_default_global_tracer(my_tracer)
        nested_tracer = init_tracer('llm')  # Returns my_tracer
        ```
    """
    # 1. Use the explicitly passed tracer if it exists
    if tracer:
        return tracer

    # 2. Try to get an active tracer from the context
    try:
        return get_current_tracer()
    except LookupError:
        # 3. Try global registry (notebook-friendly fallback)
        from axion._core.tracing.utils import get_default_global_tracer

        global_tracer = get_default_global_tracer()
        if global_tracer:
            return global_tracer

        # 4. If no context exists, use the global factory to get the
        #    correct tracer class and create an instance of it
        TracerClass = get_tracer()
        return TracerClass.create(
            metadata_type=metadata_type,
            tool_metadata=tool_metadata or infer_tool_metadata(),
        )


def Tracer(*args, **kwargs):
    """
    Factory function that always returns a new tracer instance using the
    currently configured tracer class.

    This ensures that when tracing is reconfigured, calls to Tracer()
    will use the new tracer type.

    Args:
        *args, **kwargs: Arguments passed to the tracer constructor

    Returns:
        BaseTraceHandler: A new tracer instance of the configured type
    """
    TracerClass = get_tracer()
    if args or kwargs:
        # Create new instance with provided arguments
        return TracerClass(*args, **kwargs)
    else:
        # Return the class itself (for backward compatibility)
        return TracerClass
