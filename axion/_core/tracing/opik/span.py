from typing import TYPE_CHECKING, Any, Dict, Optional

from axion._core.logging import get_logger
from axion._core.uuid import uuid7

if TYPE_CHECKING:
    from axion._core.tracing.opik.tracer import OpikTracer

try:
    import opik

    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    opik = None  # type: ignore[assignment]

logger = get_logger(__name__)

__all__ = ['OpikSpan']


class OpikSpan:
    """
    Span implementation for Opik tracing.

    Wraps Opik's span context manager to provide a consistent
    interface with other tracer implementations.
    """

    def __init__(
        self,
        tracer: 'OpikTracer',
        name: str,
        attributes: Dict[str, Any],
        is_async: bool = False,
    ):
        self.tracer = tracer
        self.name = name
        self.attributes = attributes
        self.is_async = is_async
        self._opik_span = None
        self._opik_context = None
        self._opik_trace = None
        self._span_id = str(uuid7())
        self._trace_id = (
            tracer._trace_id if hasattr(tracer, '_trace_id') else str(uuid7())
        )

    @property
    def span_id(self) -> str:
        """Get the span ID."""
        return self._span_id

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self._trace_id

    def __enter__(self) -> 'OpikSpan':
        """Enter the span context."""
        if not self.tracer._client or not OPIK_AVAILABLE:
            logger.debug(f'Opik client not initialized, skipping span: {self.name}')
            return self

        try:
            # Determine span type based on attributes
            # Use 'llm' for LLM calls (when model is present), 'general' otherwise
            span_type = 'llm' if 'model' in self.attributes else 'general'

            # Extract known Opik span parameters
            known_params = {'model', 'provider', 'input', 'output'}
            internal_params = {'auto_trace', 'new_trace'}

            opik_kwargs = {}
            metadata = {}

            for k, v in self.attributes.items():
                if k in internal_params:
                    continue
                elif k in known_params:
                    opik_kwargs[k] = v
                else:
                    metadata[k] = v

            # Check if this should be a new trace (first span in a new tracer)
            # Note: span is already on stack when __enter__ is called, so check for == 1
            is_new_trace = (
                self.attributes.get('new_trace', False) or
                len(self.tracer._span_stack) == 1
            )

            if is_new_trace:
                # Flush any pending traces before starting a new one
                # This ensures clean separation between traces
                try:
                    self.tracer._client.flush()
                except Exception:
                    pass

                # Create a new trace explicitly with our trace_id
                self._opik_trace = self.tracer._client.trace(
                    id=self._trace_id,
                    name=self.name,
                    metadata=metadata if metadata else None,
                )
                # Store the trace on the tracer so nested spans can access it
                self.tracer._current_trace = self._opik_trace

                # Create the root span under this trace using trace.span()
                # This ensures spans go to the correct project
                self._opik_context = self._opik_trace.span(
                    name=self.name,
                    type=span_type,
                    metadata=metadata if metadata else None,
                )
                self._opik_span = self._opik_context.__enter__()
                logger.debug(f'Opik new trace created: {self.name} (trace_id={self._trace_id})')
            else:
                # For nested spans, use the current trace's span method
                # This ensures all spans go to the same project
                parent_trace = getattr(self.tracer, '_current_trace', None)
                if parent_trace:
                    self._opik_context = parent_trace.span(
                        name=self.name,
                        type=span_type,
                        metadata=metadata if metadata else None,
                    )
                    self._opik_span = self._opik_context.__enter__()
                else:
                    # Fallback to global context only if no parent trace
                    self._opik_context = opik.start_as_current_span(
                        name=self.name,
                        type=span_type,
                        metadata=metadata if metadata else None,
                    )
                    self._opik_span = self._opik_context.__enter__()
                logger.debug(f'Opik span created: {self.name} (type={span_type})')

            # Set additional attributes on the span
            if 'input' in opik_kwargs:
                self._opik_span.input = opik_kwargs['input']
            if 'output' in opik_kwargs:
                self._opik_span.output = opik_kwargs['output']
            if 'model' in opik_kwargs:
                self._opik_span.model = opik_kwargs['model']
            if 'provider' in opik_kwargs:
                self._opik_span.provider = opik_kwargs['provider']

        except Exception as e:
            logger.info(f'Failed to create Opik span "{self.name}": {e}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the span context."""
        if self._opik_context:
            try:
                if exc_type is not None:
                    # Record error on the span
                    if self._opik_span:
                        self._opik_span.metadata = {
                            **(self._opik_span.metadata or {}),
                            'error': str(exc_val) if exc_val else 'Unknown error',
                            'error_type': exc_type.__name__ if exc_type else 'Unknown',
                        }
                self._opik_context.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.debug(f'Failed to close Opik span: {e}')

        # Auto-flush and clear current trace when exiting the outermost span
        if len(self.tracer._span_stack) == 1 and self.tracer._client:
            try:
                self.tracer._client.flush()
                logger.debug('Opik traces auto-flushed (outermost span closed)')
            except Exception as e:
                logger.debug(f'Failed to auto-flush Opik traces: {e}')

            # Clear the current trace so the next trace starts fresh
            if self._opik_trace:
                self.tracer._current_trace = None

        return False

    async def __aenter__(self) -> 'OpikSpan':
        """Async enter - delegates to sync implementation."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async exit - delegates to sync implementation."""
        return self.__exit__(exc_type, exc_val, exc_tb)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        if self._opik_span:
            try:
                # Opik uses direct attribute assignment or metadata
                if key in ('input', 'output', 'model', 'provider'):
                    setattr(self._opik_span, key, value)
                else:
                    current_metadata = self._opik_span.metadata or {}
                    current_metadata[key] = value
                    self._opik_span.metadata = current_metadata
            except Exception as e:
                logger.debug(f'Failed to set attribute on Opik span: {e}')

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span (stored in metadata for Opik)."""
        if self._opik_span:
            try:
                event_data = {'event': name}
                if attributes:
                    event_data.update(attributes)
                current_metadata = self._opik_span.metadata or {}
                events = current_metadata.get('events', [])
                events.append(event_data)
                current_metadata['events'] = events
                self._opik_span.metadata = current_metadata
            except Exception as e:
                logger.debug(f'Failed to add event to Opik span: {e}')

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span."""
        if self._opik_span:
            try:
                current_metadata = self._opik_span.metadata or {}
                current_metadata['error'] = str(exception)
                current_metadata['error_type'] = type(exception).__name__
                self._opik_span.metadata = current_metadata
            except Exception as e:
                logger.debug(f'Failed to record exception on Opik span: {e}')

    def add_trace(
        self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add custom trace event to the tracer."""
        trace_metadata = metadata or {}
        trace_metadata['span_id'] = self.span_id
        trace_metadata['trace_id'] = self.trace_id
        self.tracer.add_trace(event_type, message, trace_metadata)

    def set_input(self, data: Any) -> None:
        """Set the input data for this span."""
        if self._opik_span:
            try:
                serialized = self._serialize_data(data)
                self._opik_span.input = serialized
            except Exception as e:
                logger.debug(f'Failed to set input on Opik span: {e}')

    def set_output(self, data: Any) -> None:
        """Set the output data for this span."""
        if self._opik_span:
            try:
                serialized = self._serialize_data(data)
                self._opik_span.output = serialized
            except Exception as e:
                logger.debug(f'Failed to set output on Opik span: {e}')

    def set_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Set token usage for LLM spans."""
        if self._opik_span:
            try:
                self._opik_span.usage = {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens,
                }
            except Exception as e:
                logger.debug(f'Failed to set usage on Opik span: {e}')

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
