from typing import TYPE_CHECKING, Any, Dict, Optional

from axion._core.logging import get_logger
from axion._core.uuid import uuid7

if TYPE_CHECKING:
    from axion._core.tracing.langfuse.tracer import LangfuseTracer

logger = get_logger(__name__)

__all__ = ['LangfuseSpan']


class LangfuseSpan:
    """
    Span implementation for Langfuse tracing.

    Wraps Langfuse's observation context manager to provide a consistent
    interface with other tracer implementations.
    """

    def __init__(
        self,
        tracer: 'LangfuseTracer',
        name: str,
        attributes: Dict[str, Any],
        is_async: bool = False,
    ):
        self.tracer = tracer
        self.name = name
        self.attributes = attributes
        self.is_async = is_async
        self._observation = None
        self._observation_context = None
        self._span_id = str(uuid7())
        self._trace_id = tracer._trace_id if hasattr(tracer, '_trace_id') else str(uuid7())

    @property
    def span_id(self) -> str:
        """Get the span ID."""
        return self._span_id

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self._trace_id

    def __enter__(self) -> 'LangfuseSpan':
        """Enter the span context."""
        if not self.tracer._client:
            logger.debug(f'Langfuse client not initialized, skipping span: {self.name}')
            return self

        try:
            # Determine observation type based on attributes
            # Use 'generation' for LLM calls (when model is present), 'span' otherwise
            as_type = 'generation' if 'model' in self.attributes else 'span'

            # Langfuse only accepts specific kwargs for start_as_current_observation
            # Known params: as_type, name, model (for generation), input, output
            # Everything else goes into metadata
            known_params = {'model', 'input', 'output'}
            internal_params = {'auto_trace'}

            langfuse_kwargs = {}
            metadata = {}

            for k, v in self.attributes.items():
                if k in internal_params:
                    continue
                elif k in known_params:
                    langfuse_kwargs[k] = v
                else:
                    # Put custom attributes in metadata
                    metadata[k] = v

            self._observation_context = self.tracer._client.start_as_current_observation(
                as_type=as_type,
                name=self.name,
                metadata=metadata if metadata else None,
                **langfuse_kwargs,
            )
            self._observation = self._observation_context.__enter__()
            logger.debug(f'Langfuse span created: {self.name} (type={as_type})')
        except Exception as e:
            logger.info(f'Failed to create Langfuse span "{self.name}": {e}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the span context."""
        if self._observation_context:
            try:
                if exc_type is not None:
                    # Record error on the observation
                    self._observation.update(
                        level='ERROR',
                        status_message=str(exc_val) if exc_val else 'Unknown error',
                    )
                self._observation_context.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.debug(f'Failed to close Langfuse span: {e}')

        # Auto-flush when exiting the outermost span
        # (span_stack length is 1 means this is the last span, about to be popped)
        if len(self.tracer._span_stack) == 1 and self.tracer._client:
            try:
                self.tracer._client.flush()
                logger.debug('Langfuse traces auto-flushed (outermost span closed)')
            except Exception as e:
                logger.debug(f'Failed to auto-flush Langfuse traces: {e}')

        return False

    async def __aenter__(self) -> 'LangfuseSpan':
        """Async enter - delegates to sync implementation."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async exit - delegates to sync implementation."""
        return self.__exit__(exc_type, exc_val, exc_tb)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        if self._observation:
            try:
                self._observation.update(**{key: value})
            except Exception as e:
                logger.debug(f'Failed to set attribute on Langfuse span: {e}')

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        if self._observation:
            try:
                event_data = {'event': name}
                if attributes:
                    event_data.update(attributes)
                self._observation.update(metadata=event_data)
            except Exception as e:
                logger.debug(f'Failed to add event to Langfuse span: {e}')

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span."""
        if self._observation:
            try:
                self._observation.update(
                    level='ERROR',
                    status_message=str(exception),
                )
            except Exception as e:
                logger.debug(f'Failed to record exception on Langfuse span: {e}')

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
        if self._observation:
            try:
                serialized = self._serialize_data(data)
                self._observation.update(input=serialized)
            except Exception as e:
                logger.debug(f'Failed to set input on Langfuse span: {e}')

    def set_output(self, data: Any) -> None:
        """Set the output data for this span."""
        if self._observation:
            try:
                serialized = self._serialize_data(data)
                self._observation.update(output=serialized)
            except Exception as e:
                logger.debug(f'Failed to set output on Langfuse span: {e}')

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
