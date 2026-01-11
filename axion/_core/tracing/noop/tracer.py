import asyncio
import functools
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

from axion._core.logging import get_logger
from axion._core.metadata.schema import BaseExecutionMetadata, ToolMetadata
from axion._core.tracing import reset_tracer_context, set_current_tracer
from axion._core.tracing.noop.span import Span
from axion._core.tracing.registry import BaseTracer, TracerRegistry
from axion._core.utils import Timer
from axion._core.uuid import uuid7

logger = get_logger(__name__)


@TracerRegistry.register('noop')
class NoOpTracer(BaseTracer):
    """
    A no-operation tracer that mirrors the LogfireTracer interface.

    This class provides a complete, non-functional implementation of the tracer
    so that it can be swapped with the real LogfireTracer without causing
    any errors. All tracing and logging methods are designed to do nothing,
    ensuring minimal performance overhead when tracing is disabled.
    """

    def __init__(
        self,
        metadata_type: str,
        tool_metadata: Optional[ToolMetadata] = None,
        enable_logfire: bool = False,
        trace_id: Optional[str] = None,
        **kwargs,
    ):
        self.metadata_type = metadata_type
        self.tool_metadata = tool_metadata or self._create_default_tool_meta()
        self.enable_logfire = False  # Always False for NoOp
        self.kwargs = kwargs
        self.logger = get_logger(
            f'axion.tracing.{self.tool_metadata.name}_{metadata_type}'
        )
        self.timer = Timer()

        self._current_span: Optional[Span] = None
        self._span_stack: List[Span] = []
        self._trace_id = trace_id or str(uuid7())

        # Metadata
        self._metadata = self._create_metadata()

        logger.debug('Tracing disabled - using NoOpTracer.')

    @staticmethod
    def _create_default_tool_meta() -> ToolMetadata:
        """Creates minimal default metadata for the no-op tracer."""
        return ToolMetadata(
            name='noop_tracer',
            description='No-operation tracer',
            owner='system',
            version='1.0.0',
        )

    def _create_metadata(self) -> BaseExecutionMetadata:
        """Create the appropriate metadata instance based on type."""
        return BaseExecutionMetadata(
            name=self.tool_metadata.name, tool_metadata=self.tool_metadata.to_dict()
        )

    @contextmanager
    def span(self, operation_name: str, **attributes):
        """
        Creates a synchronous span AND sets the tracer context for its duration.
        Uses shared context management from main tracer module.
        """
        token = set_current_tracer(self)
        span_obj = Span(operation_name, self, attributes)
        try:
            yield span_obj
        finally:
            reset_tracer_context(token)

    @asynccontextmanager
    async def async_span(self, operation_name: str, **attributes):
        """
        Creates an asynchronous span AND sets the tracer context for its duration.
        Uses shared context management from main tracer module.
        """
        token = set_current_tracer(self)
        span_obj = Span(operation_name, self, attributes)
        try:
            yield span_obj
        finally:
            reset_tracer_context(token)

    @contextmanager
    def context(self):
        """A context manager to set this tracer as the current one."""
        token = set_current_tracer(self)
        try:
            yield
        finally:
            reset_tracer_context(token)

    @asynccontextmanager
    async def acontext(self):
        """An async context manager to set this tracer as the current one."""
        token = set_current_tracer(self)
        try:
            yield
        finally:
            reset_tracer_context(token)

    def add_trace(self, event_type: str, message: str, metadata: Dict[str, Any] = None):
        """Add a trace event - no-op."""
        pass

    def start(self, **attributes):
        """Start execution tracking - no-op."""
        pass

    def complete(self, output_data: Dict[str, Any] = None, **attributes):
        """Complete execution tracking - no-op."""
        pass

    def info(self, msg: Any):
        """Log info"""
        logger.info(msg)

    def fail(self, error: str, **attributes):
        """Even when tracing is off, log the failure."""
        logger.error(f'Execution failed: {error}', extra=attributes)

    def log_performance(self, name: str, duration: float, **attributes):
        """No-op performance logging."""
        pass

    def log_table(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        columns: Optional[List[str]] = None,
        level: Union[int, str] = logging.INFO,
    ) -> None:
        """No-op table logging."""
        pass

    def log_llm_call(self, *args, **kwargs):
        """No-op LLM call logging."""
        pass

    def log_retrieval_call(self, *args, **kwargs):
        """No-op retrieval call logging."""
        pass

    def log_database_query(self, *args, **kwargs):
        """No-op database query logging."""
        pass

    def log_evaluation(self, *args, **kwargs):
        """No-op evaluation logging."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Return empty statistics."""
        return {}

    def display_statistics(self):
        """No-op statistics display."""
        pass

    def get_llm_statistics(self) -> Dict[str, Any]:
        """Return empty LLM statistics."""
        return {}

    def display_llm_statistics(self):
        """No-op LLM statistics display."""
        pass

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Return empty knowledge statistics."""
        return {}

    def display_knowledge_statistics(self):
        """No-op knowledge statistics display."""
        pass

    def get_database_statistics(self) -> Dict[str, Any]:
        """Return empty database statistics."""
        return {}

    def display_database_statistics(self):
        """No-op database statistics display."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata with current span context."""
        metadata = self._metadata.model_dump()
        if self._current_span:
            metadata['current_span_id'] = self._current_span.span_id
            metadata['trace_id'] = self.trace_id
        return metadata

    def display_traces(self):
        """No-op trace display."""
        pass

    @property
    def metadata(self) -> BaseExecutionMetadata:
        return self._metadata

    @property
    def current_span(self):
        return self._current_span

    @property
    def trace_id(self) -> str:
        if self._current_span:
            return self._current_span.trace_id
        return self._trace_id

    @property
    def handler(self):
        """Return None for handler since this is no-op."""
        return None

    @classmethod
    def create(
        cls,
        metadata_type: str,
        tool_metadata: Optional[ToolMetadata] = None,
        # configuration_params: Optional[Dict[str, Any]] = None,
        enable_logfire: bool = False,
        **kwargs,
    ) -> 'NoOpTracer':
        """Factory method to create a NoOpTracer instance."""
        return cls(metadata_type, tool_metadata, enable_logfire, **kwargs)

    @staticmethod
    def traced_operation(
        operation_name: Optional[str] = None,
        auto_trace: bool = True,
        capture_args: bool = False,
        capture_result: bool = False,
        use_class_name: bool = False,
        operation_suffix: Optional[str] = None,
        **span_attributes,
    ):
        """A no-op decorator that returns the original function untouched."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For no-op, we still need to set context but don't actually trace
                try:
                    from axion._core.tracing import get_current_tracer

                    _ = get_current_tracer()
                    return func(*args, **kwargs)
                except LookupError:
                    return func(*args, **kwargs)

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    from axion._core.tracing import get_current_tracer

                    _ = get_current_tracer()
                    return await func(*args, **kwargs)
                except LookupError:
                    return await func(*args, **kwargs)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    @staticmethod
    def trace(
        operation_suffix: str = None,
        capture_args: bool = False,
        capture_result: bool = False,
        **span_attributes,
    ):
        """No-op trace decorator that mimics LogfireTracer.trace interface."""
        return NoOpTracer.traced_operation(
            use_class_name=True,
            operation_suffix=operation_suffix,
            capture_args=capture_args,
            capture_result=capture_result,
            **span_attributes,
        )

    def __getattr__(self, name: str) -> Callable:
        """
        Catches any method call not explicitly defined and returns a
        no-op function. This ensures compatibility if new methods are
        added to LogfireTracer without being added here.
        """

        def noop(*args, **kwargs):
            if name.startswith('get_'):
                return {} if 'statistics' in name else None
            return None

        return noop


def trace(
    operation_suffix: str = None,
    capture_args: bool = False,
    capture_result: bool = False,
    **span_attributes,
):
    """
    Factory function for class-aware operation tracing (no-op version).
    """
    return NoOpTracer.traced_operation(
        use_class_name=True,
        operation_suffix=operation_suffix,
        capture_args=capture_args,
        capture_result=capture_result,
        **span_attributes,
    )
