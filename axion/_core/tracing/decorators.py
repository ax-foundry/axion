import asyncio
import functools
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel

from axion._core.tracing.statistics import (
    MAX_ARG_LENGTH,
    MAX_ERROR_LENGTH,
    MAX_RESULT_LENGTH,
)


def trace(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    capture_args: bool = False,
    capture_result: bool = False,
    capture_response: bool = False,
    span_type: Optional[str] = None,
):
    """
    A unified decorator that instruments a method for tracing.

    This decorator can be used with or without arguments and automatically
    creates spans for traced methods. It requires that the decorated method's
    class has a 'tracer' attribute.

    Args:
        _func: The function being decorated (when used without parentheses)
        name: Custom name for the span (defaults to function name)
        capture_args: Whether to capture and log function arguments
        capture_result: Whether to capture and log function result
        capture_response: Alias for capture_result
        span_type: Explicit span type override (e.g., 'tool', 'evaluator', 'retriever').
            If not specified, type is auto-inferred from span name and attributes.

    Usage:
        ```python
        class MyService:
            def __init__(self):
                self.tracer = init_tracer('service')

            @trace
            async def simple_method(self):
                return "result"

            @trace("custom_span_name")
            async def named_method(self):
                return "result"

            @trace(name="process", capture_args=True, capture_result=True)
            async def detailed_method(self, data):
                return process(data)
        ```

    Note:
        The decorated method's class MUST have a 'tracer' attribute that
        provides span() or async_span() context managers.
    """
    # Allow `capture_response` as an alias for `capture_result`
    capture_result = capture_result or capture_response

    def decorator(func: Callable):
        # Static fallback name (used if self.name is not available)
        static_span_name = name or func.__name__

        def _get_span_name(instance) -> str:
            """Get span name, preferring instance.name if available."""
            if instance is not None and hasattr(instance, 'name'):
                try:
                    instance_name = instance.name
                    if instance_name:
                        return instance_name
                except Exception:
                    pass
            return static_span_name

        def _build_attributes(args, kwargs) -> Dict[str, Any]:
            """Build span attributes from function arguments."""
            if not capture_args:
                return {}

            # Exclude 'self' from the captured positional arguments
            args_to_capture = (
                args[1:] if args and hasattr(args[0], '__class__') else args
            )

            safe_args = [
                (
                    arg.model_dump_json()
                    if isinstance(arg, BaseModel)
                    else (
                        str(arg)
                        if isinstance(arg, (str, int, float, bool))
                        and len(str(arg)) < 2000
                        else f'<{type(arg).__name__}>'
                    )
                )
                for arg in args_to_capture
            ]
            safe_kwargs = {
                k: (
                    v
                    if isinstance(v, (str, int, float, bool))
                    and len(str(v)) < MAX_ARG_LENGTH
                    else f'<{type(v).__name__}>'
                )
                for k, v in kwargs.items()
            }
            return {'function.args': safe_args, 'function.kwargs': safe_kwargs}

        def _filter_non_null(data: Any) -> Any:
            """Recursively filter out null values from dicts/lists."""
            if isinstance(data, dict):
                return {
                    k: _filter_non_null(v) for k, v in data.items() if v is not None
                }
            elif isinstance(data, list):
                return [_filter_non_null(item) for item in data if item is not None]
            return data

        def _build_input_data(args, kwargs) -> Any:
            """Build input data for span.set_input().

            Only captures the first positional argument (typically data_item)
            and filters out null values for cleaner observability.
            """
            # Exclude 'self' from the captured positional arguments
            args_to_capture = (
                args[1:] if args and hasattr(args[0], '__class__') else args
            )

            # Only capture the first argument (the data_item)
            if not args_to_capture:
                return None

            first_arg = args_to_capture[0]
            if isinstance(first_arg, BaseModel):
                data = first_arg.model_dump()
            elif isinstance(first_arg, (dict, list)):
                data = first_arg
            elif isinstance(first_arg, (str, int, float, bool)):
                return first_arg  # Primitives don't need null filtering
            else:
                return f'<{type(first_arg).__name__}>'

            # Filter out null values
            return _filter_non_null(data)

        def _serialize_output(result: Any) -> Any:
            """Serialize result for span.set_output()."""
            if result is None:
                return None
            if isinstance(result, BaseModel):
                return result.model_dump()
            if isinstance(result, (str, int, float, bool, dict, list)):
                return result
            return f'<{type(result).__name__}>'

        def _capture_result_attributes(span, result: Any):
            """Capture function result as span attributes."""
            if not capture_result or result is None:
                return

            if isinstance(result, BaseModel):
                span.set_attribute('function.result', result.model_dump_json())
            elif (
                isinstance(result, (str, int, float, bool, dict, list))
                and len(str(result)) < 2000
            ):
                span.set_attribute('function.result', result)
            else:
                span.set_attribute('function.result_type', type(result).__name__)

        def _capture_error_attributes(span, exception: Exception):
            """Capture exception details as span attributes."""
            span.set_attribute('error.type', type(exception).__name__)
            span.set_attribute('error.message', str(exception)[:MAX_ERROR_LENGTH])

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            """Synchronous function wrapper."""
            if not hasattr(self, 'tracer'):
                # If no tracer available, execute function normally
                return func(self, *args, **kwargs)

            # Get span name dynamically (prefers self.name if available)
            effective_span_name = _get_span_name(self)

            attributes = _build_attributes((self,) + args, kwargs)
            # Pass input directly in attributes so it's set at span creation time
            # This ensures the input is captured before any child spans can interfere
            if capture_args:
                attributes['input'] = _build_input_data((self,) + args, kwargs)
            if span_type:
                attributes['span_type'] = span_type

            with self.tracer.span(effective_span_name, **attributes) as span:
                try:
                    result = func(self, *args, **kwargs)
                    _capture_result_attributes(span, result)

                    # Capture output for Langfuse/tracing visibility
                    if capture_result and hasattr(span, 'set_output'):
                        span.set_output(_serialize_output(result))

                    return result
                except Exception as e:
                    _capture_error_attributes(span, e)
                    raise

        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            """Asynchronous function wrapper."""
            if not hasattr(self, 'tracer'):
                # If no tracer available, execute function normally
                return await func(self, *args, **kwargs)

            # Get span name dynamically (prefers self.name if available)
            effective_span_name = _get_span_name(self)

            attributes = _build_attributes((self,) + args, kwargs)
            # Pass input directly in attributes so it's set at span creation time
            # This ensures the input is captured before any child spans can interfere
            if capture_args:
                attributes['input'] = _build_input_data((self,) + args, kwargs)
            if span_type:
                attributes['span_type'] = span_type

            async with self.tracer.async_span(
                effective_span_name, **attributes
            ) as span:
                try:
                    result = await func(self, *args, **kwargs)
                    _capture_result_attributes(span, result)

                    # Capture output for Langfuse/tracing visibility
                    if capture_result and hasattr(span, 'set_output'):
                        span.set_output(_serialize_output(result))

                    return result
                except Exception as e:
                    _capture_error_attributes(span, e)
                    raise

        # Return appropriate wrapper based on function type
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return wrapper

    # Handle different decorator usage patterns
    if _func is None:
        # Called with arguments: @trace(name="custom")
        return decorator
    else:
        # Handle positional string argument: @trace("custom_name")
        if isinstance(_func, str):
            name = _func
            return decorator
        # Called without arguments: @trace
        return decorator(_func)


def trace_method(
    name: Optional[str] = None,
    capture_args: bool = False,
    capture_result: bool = False,
    auto_start: bool = True,
    span_type: Optional[str] = None,
):
    """
    Method-specific tracing decorator with additional features.

    This decorator is similar to @trace but provides additional method-specific
    functionality like automatic tracer start/complete calls.

    Args:
        name: Custom span name
        capture_args: Whether to capture method arguments
        capture_result: Whether to capture method result
        auto_start: Whether to automatically call tracer.start()
        span_type: Explicit span type override (e.g., 'tool', 'evaluator', 'retriever').
            If not specified, type is auto-inferred from span name and attributes.

    Example:
        ```python
        class DataProcessor:
            def __init__(self):
                self.tracer = init_tracer('processor')

            @trace_method(name="process_data", capture_args=True, auto_start=True)
            async def process(self, data):
                # tracer.start() called automatically
                result = await process_data(data)
                # tracer.complete() called automatically
                return result
        ```
    """

    def decorator(func: Callable):
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            if not hasattr(self, 'tracer'):
                return await func(self, *args, **kwargs)

            attributes = {}
            if capture_args:
                # Exclude 'self' from captured args
                safe_args = [
                    (
                        str(arg)
                        if isinstance(arg, (str, int, float, bool))
                        and len(str(arg)) < MAX_ARG_LENGTH
                        else f'<{type(arg).__name__}>'
                    )
                    for arg in args
                ]
                attributes['method.args'] = safe_args
                attributes['method.kwargs'] = {
                    k: (
                        v
                        if isinstance(v, (str, int, float, bool))
                        and len(str(v)) < MAX_ARG_LENGTH
                        else f'<{type(v).__name__}>'
                    )
                    for k, v in kwargs.items()
                }
            if span_type:
                attributes['span_type'] = span_type

            async with self.tracer.async_span(span_name, **attributes) as span:
                try:
                    if auto_start and hasattr(self.tracer, 'start'):
                        self.tracer.start()

                    result = await func(self, *args, **kwargs)

                    if capture_result and result is not None:
                        if (
                            isinstance(result, (str, int, float, bool))
                            and len(str(result)) < MAX_RESULT_LENGTH
                        ):
                            span.set_attribute('method.result', result)
                        else:
                            span.set_attribute(
                                'method.result_type', type(result).__name__
                            )

                    if auto_start and hasattr(self.tracer, 'complete'):
                        self.tracer.complete({'result_type': type(result).__name__})

                    return result
                except Exception as e:
                    if auto_start and hasattr(self.tracer, 'fail'):
                        self.tracer.fail(str(e))
                    span.set_attribute('error.type', type(e).__name__)
                    span.set_attribute('error.message', str(e)[:MAX_ERROR_LENGTH])
                    raise

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            if not hasattr(self, 'tracer'):
                return func(self, *args, **kwargs)

            attributes = {}
            if capture_args:
                safe_args = [
                    (
                        str(arg)
                        if isinstance(arg, (str, int, float, bool))
                        and len(str(arg)) < MAX_ARG_LENGTH
                        else f'<{type(arg).__name__}>'
                    )
                    for arg in args
                ]
                attributes['method.args'] = safe_args
                attributes['method.kwargs'] = {
                    k: (
                        v
                        if isinstance(v, (str, int, float, bool))
                        and len(str(v)) < MAX_ARG_LENGTH
                        else f'<{type(v).__name__}>'
                    )
                    for k, v in kwargs.items()
                }
            if span_type:
                attributes['span_type'] = span_type

            with self.tracer.span(span_name, **attributes) as span:
                try:
                    if auto_start and hasattr(self.tracer, 'start'):
                        self.tracer.start()

                    result = func(self, *args, **kwargs)

                    if capture_result and result is not None:
                        if (
                            isinstance(result, (str, int, float, bool))
                            and len(str(result)) < MAX_RESULT_LENGTH
                        ):
                            span.set_attribute('method.result', result)
                        else:
                            span.set_attribute(
                                'method.result_type', type(result).__name__
                            )

                    if auto_start and hasattr(self.tracer, 'complete'):
                        self.tracer.complete({'result_type': type(result).__name__})

                    return result
                except Exception as e:
                    if auto_start and hasattr(self.tracer, 'fail'):
                        self.tracer.fail(str(e))
                    span.set_attribute('error.type', type(e).__name__)
                    span.set_attribute('error.message', str(e)[:MAX_ERROR_LENGTH])
                    raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def trace_function(
    name: Optional[str] = None,
    capture_args: bool = False,
    capture_result: bool = False,
    span_type: Optional[str] = None,
):
    """
    Function-level tracing decorator for standalone functions.

    This decorator is for tracing standalone functions that don't belong to a class
    with a tracer attribute. It attempts to get the current tracer from context.

    Args:
        name: Custom span name
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture function result
        span_type: Explicit span type override (e.g., 'tool', 'evaluator', 'retriever').
            If not specified, type is auto-inferred from span name and attributes.

    Example:
        ```python
        @trace_function(name="process_data", capture_args=True)
        async def process_data(data):
            # Uses tracer from current context
            return processed_data

        # Usage within traced context
        async with tracer.async_span("main_operation"):
            result = await process_data(my_data)  # Automatically traced
        ```
    """

    def decorator(func: Callable):
        from axion._core.tracing.context import (
            get_current_tracer,
            has_current_tracer,
        )

        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not has_current_tracer():
                return await func(*args, **kwargs)

            tracer = get_current_tracer()
            attributes = {}

            if capture_args:
                safe_args = [
                    (
                        str(arg)
                        if isinstance(arg, (str, int, float, bool))
                        and len(str(arg)) < MAX_ARG_LENGTH
                        else f'<{type(arg).__name__}>'
                    )
                    for arg in args
                ]
                attributes['function.args'] = safe_args
                attributes['function.kwargs'] = {
                    k: (
                        v
                        if isinstance(v, (str, int, float, bool))
                        and len(str(v)) < MAX_ARG_LENGTH
                        else f'<{type(v).__name__}>'
                    )
                    for k, v in kwargs.items()
                }
            if span_type:
                attributes['span_type'] = span_type

            async with tracer.async_span(span_name, **attributes) as span:
                try:
                    result = await func(*args, **kwargs)

                    if capture_result and result is not None:
                        if (
                            isinstance(result, (str, int, float, bool))
                            and len(str(result)) < MAX_RESULT_LENGTH
                        ):
                            span.set_attribute('function.result', result)
                        else:
                            span.set_attribute(
                                'function.result_type', type(result).__name__
                            )

                    return result
                except Exception as e:
                    span.set_attribute('error.type', type(e).__name__)
                    span.set_attribute('error.message', str(e)[:MAX_ERROR_LENGTH])
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not has_current_tracer():
                return func(*args, **kwargs)

            tracer = get_current_tracer()
            attributes = {}

            if capture_args:
                safe_args = [
                    (
                        str(arg)
                        if isinstance(arg, (str, int, float, bool))
                        and len(str(arg)) < MAX_ARG_LENGTH
                        else f'<{type(arg).__name__}>'
                    )
                    for arg in args
                ]
                attributes['function.args'] = safe_args
                attributes['function.kwargs'] = {
                    k: (
                        v
                        if isinstance(v, (str, int, float, bool))
                        and len(str(v)) < MAX_ARG_LENGTH
                        else f'<{type(v).__name__}>'
                    )
                    for k, v in kwargs.items()
                }
            if span_type:
                attributes['span_type'] = span_type

            with tracer.span(span_name, **attributes) as span:
                try:
                    result = func(*args, **kwargs)

                    if capture_result and result is not None:
                        if (
                            isinstance(result, (str, int, float, bool))
                            and len(str(result)) < MAX_RESULT_LENGTH
                        ):
                            span.set_attribute('function.result', result)
                        else:
                            span.set_attribute(
                                'function.result_type', type(result).__name__
                            )

                    return result
                except Exception as e:
                    span.set_attribute('error.type', type(e).__name__)
                    span.set_attribute('error.message', str(e)[:MAX_ERROR_LENGTH])
                    raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
