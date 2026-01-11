from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from axion._core.utils import Timer

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


class Span:
    """Unified span for both sync and async operations with automatic cleanup."""

    def __init__(
        self,
        operation_name: str,
        tracer: Callable,
        attributes: Optional[Dict[str, Any]] = None,
        auto_trace: bool = True,
    ):
        self.operation_name = operation_name
        self.tracer = tracer
        self.attributes = attributes or {}
        self.auto_trace = auto_trace
        self.timer = Timer()
        self.span_id = str(uuid4())
        self.trace_id = getattr(tracer, '_trace_id', str(uuid4()))
        self.logfire_span = None
        self._is_entered = False

    def __enter__(self):
        return self._enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._exit(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        return self._enter()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self._exit(exc_type, exc_val, exc_tb)

    def _enter(self):
        if self._is_entered:
            return self

        self._is_entered = True
        self.timer.start()

        # Update span stack
        self._push_to_stack()

        # Start Logfire span if available
        if LOGFIRE_AVAILABLE and self.tracer.enable_logfire:
            self._start_logfire_span()

        # Auto trace start
        if self.auto_trace:
            self._trace_event('start', f'Started {self.operation_name}')

        return self

    def _exit(self, exc_type, exc_val, exc_tb):
        if not self._is_entered:
            return

        self.timer.stop()
        latency = self.timer.elapsed_time

        # Handle errors or success
        if exc_type:
            self._handle_error(exc_val, latency)
        elif self.auto_trace:
            self._trace_event(
                'complete',
                f'Completed {self.operation_name}',
                {'latency': latency, **self.attributes},
            )

        # Close Logfire span
        if self.logfire_span:
            try:
                self.logfire_span.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass

        # Clean up span stack
        self._pop_from_stack()
        self._is_entered = False

    def _push_to_stack(self):
        """Add this span to the tracker's span stack."""
        if not hasattr(self.tracer, '_span_stack'):
            self.tracer._span_stack = []
        self.tracer._span_stack.append(self)
        self.tracer._current_span = self

    def _pop_from_stack(self):
        """Remove this span from the tracker's span stack."""
        if hasattr(self.tracer, '_span_stack') and self.tracer._span_stack:
            if self.tracer._span_stack[-1] == self:
                self.tracer._span_stack.pop()
                self.tracer._current_span = (
                    self.tracer._span_stack[-1] if self.tracer._span_stack else None
                )

    def _start_logfire_span(self):
        """Start Logfire span with error handling."""
        try:
            attrs = {
                'operation': self.operation_name,
                'tool_name': self.tracer.tool_metadata.name,
                'span_id': self.span_id,
                'trace_id': self.trace_id,
                **self.attributes,
            }
            self.logfire_span = logfire.span(
                f'{self.tracer.tool_metadata.name}: {self.operation_name}', **attrs
            ).__enter__()
        except Exception as e:
            self.tracer.warning(f'Failed to create Logfire span: {e}')

    def _handle_error(self, exc_val: Exception, latency: float):
        """Handle span errors."""
        error_msg = str(exc_val) if exc_val else 'Unknown error'

        if self.logfire_span:
            try:
                self.logfire_span.set_level('error')
                self.logfire_span.record_exception(exc_val)
            except Exception:
                pass

        if self.auto_trace:
            self._trace_event(
                'failed',
                f'Failed {self.operation_name}: {error_msg}',
                {'error': error_msg, 'latency': latency},
            )

    def _trace_event(self, suffix: str, message: str, metadata: Dict[str, Any] = None):
        """Add trace event with span context."""
        self.tracer._add_trace_internal(
            f'{self.operation_name}_{suffix}',
            message,
            {
                'span_id': self.span_id,
                'trace_id': self.trace_id,
                **(metadata or {}),
                **self.attributes,
            },
        )

    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value
        if self.logfire_span and hasattr(self.logfire_span, 'set_attribute'):
            try:
                self.logfire_span.set_attribute(key, value)
            except Exception:
                pass

    def add_trace(
        self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Add custom trace event."""
        self.tracer._add_trace_internal(
            event_type,
            message,
            {'span_id': self.span_id, 'trace_id': self.trace_id, **(metadata or {})},
        )

    def set_input(self, data: Any) -> None:
        """Set the input data for this span."""
        serialized = self._serialize_data(data)
        self.set_attribute('input', serialized)

    def set_output(self, data: Any) -> None:
        """Set the output data for this span."""
        serialized = self._serialize_data(data)
        self.set_attribute('output', serialized)

    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for tracing (handles Pydantic models, dicts, etc.)."""
        if data is None:
            return None
        elif hasattr(data, 'model_dump'):
            return data.model_dump()
        elif hasattr(data, 'dict'):
            return data.dict()
        elif isinstance(data, (dict, list, str, int, float, bool)):
            return data
        else:
            return str(data)
