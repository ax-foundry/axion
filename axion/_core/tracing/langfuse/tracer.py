import logging
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional, Union

from axion._core.environment import settings
from axion._core.logging import get_logger
from axion._core.metadata.schema import BaseExecutionMetadata, Status, ToolMetadata
from axion._core.tracing import reset_tracer_context, set_current_tracer
from axion._core.tracing.langfuse.span import LangfuseSpan
from axion._core.tracing.registry import BaseTracer, TracerRegistry
from axion._core.utils import Timer
from axion._core.uuid import uuid7

try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None  # type: ignore[misc, assignment]

logger = get_logger(__name__)

__all__ = ['LangfuseTracer', 'LANGFUSE_AVAILABLE']


@TracerRegistry.register('langfuse')
class LangfuseTracer(BaseTracer):
    """
    Langfuse-based tracer for LLM observability.

    This tracer integrates with Langfuse (https://langfuse.com) to provide
    detailed tracing and observability for LLM applications.

    Configuration:
        Set the following environment variables or pass to constructor:
        - LANGFUSE_PUBLIC_KEY: Your Langfuse public key
        - LANGFUSE_SECRET_KEY: Your Langfuse secret key
        - LANGFUSE_BASE_URL: API endpoint (default: https://cloud.langfuse.com)
        - LANGFUSE_TAGS: Comma-separated list of tags (e.g., "prod,v1.0")
        - LANGFUSE_ENVIRONMENT: Environment name (e.g., "production", "staging")

    Example:
        from axion._core.tracing import Tracer, configure_tracing

        # Configure via environment
        os.environ['TRACING_MODE'] = 'langfuse'
        os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-...'
        os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-...'
        os.environ['LANGFUSE_TAGS'] = 'prod,v1.0'
        os.environ['LANGFUSE_ENVIRONMENT'] = 'production'

        configure_tracing()
        tracer = Tracer('llm')
        with tracer.span('my-operation', tags=['custom-tag']):
            # ... your code
        tracer.flush()

        # Or pass tags/environment directly:
        tracer = LangfuseTracer(tags=['prod', 'v1.0'], environment='production')
        with tracer.span('my-operation'):
            # ... your code
    """

    def __init__(
        self,
        metadata_type: str = 'default',
        tool_metadata: Optional[ToolMetadata] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        trace_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        environment: Optional[str] = None,
        **kwargs,
    ):
        self.metadata_type = metadata_type
        self.tool_metadata = tool_metadata or self._create_default_tool_meta()
        self.kwargs = kwargs
        self.logger = get_logger(
            f'axion.tracing.{self.tool_metadata.name}_{metadata_type}'
        )
        self.timer = Timer()

        # Tracing state
        self._current_span: Optional[LangfuseSpan] = None
        self._span_stack: List[LangfuseSpan] = []
        self._trace_id = trace_id or str(uuid7())

        # Metadata
        self._metadata = self._create_metadata()

        # Resolve credentials from args or settings
        self._public_key = public_key or settings.langfuse_public_key
        self._secret_key = secret_key or settings.langfuse_secret_key
        self._base_url = base_url or settings.langfuse_base_url

        # Tags and environment (support env vars)
        if tags is None:
            env_tags = os.environ.get('LANGFUSE_TAGS')
            if env_tags:
                self.tags = [t.strip() for t in env_tags.split(',') if t.strip()]
            else:
                self.tags = []
        else:
            self.tags = tags

        self.environment = environment or os.environ.get('LANGFUSE_ENVIRONMENT')

        # Initialize client
        self._client: Optional[Langfuse] = None
        self._initialize_client()

    @staticmethod
    def _create_default_tool_meta() -> ToolMetadata:
        """Creates minimal default metadata for the tracer."""
        return ToolMetadata(
            name='langfuse_tracer',
            description='Langfuse-based tracer',
            owner='system',
            version='1.0.0',
        )

    def _create_metadata(self) -> BaseExecutionMetadata:
        """Create the appropriate metadata instance based on type."""
        return BaseExecutionMetadata(
            name=self.tool_metadata.name, tool_metadata=self.tool_metadata.to_dict()
        )

    def _initialize_client(self) -> None:
        """Initialize the Langfuse client."""
        if not LANGFUSE_AVAILABLE:
            logger.warning('langfuse package not installed. Run: pip install langfuse')
            return

        if not self._public_key or not self._secret_key:
            logger.warning(
                'Langfuse credentials not configured. '
                'Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables.'
            )
            return

        try:
            # Build client kwargs
            client_kwargs = {
                'public_key': self._public_key,
                'secret_key': self._secret_key,
                'host': self._base_url,
            }

            # Set environment if provided (Langfuse SDK supports this at client init)
            # Also check LANGFUSE_TRACING_ENVIRONMENT env var (Langfuse SDK standard)
            env = self.environment or os.environ.get('LANGFUSE_TRACING_ENVIRONMENT')
            if env:
                client_kwargs['environment'] = env

            self._client = Langfuse(**client_kwargs)
            logger.debug(
                f'Langfuse client initialized successfully (endpoint: {self._base_url}, '
                f'environment: {env or "default"})'
            )
        except Exception as e:
            logger.warning(f'Failed to initialize Langfuse client: {e}')
            self._client = None

    @classmethod
    def create(
        cls,
        metadata_type: str = 'default',
        tool_metadata: Optional[ToolMetadata] = None,
        **kwargs,
    ) -> 'LangfuseTracer':
        """Factory method to create a LangfuseTracer instance."""
        return cls(metadata_type=metadata_type, tool_metadata=tool_metadata, **kwargs)

    @contextmanager
    def span(self, operation_name: str, **attributes):
        """
        Creates a synchronous span AND sets the tracer context for its duration.

        Args:
            operation_name: Name of the operation being traced
            **attributes: Additional attributes to attach to the span

        Yields:
            A LangfuseSpan object
        """
        token = set_current_tracer(self)
        span_obj = LangfuseSpan(self, operation_name, attributes, is_async=False)
        self._span_stack.append(span_obj)
        self._current_span = span_obj
        try:
            with span_obj:
                yield span_obj
        finally:
            self._span_stack.pop()
            self._current_span = self._span_stack[-1] if self._span_stack else None
            reset_tracer_context(token)

    @asynccontextmanager
    async def async_span(self, operation_name: str, **attributes):
        """
        Creates an asynchronous span AND sets the tracer context for its duration.

        Args:
            operation_name: Name of the operation being traced
            **attributes: Additional attributes to attach to the span

        Yields:
            A LangfuseSpan object
        """
        token = set_current_tracer(self)
        span_obj = LangfuseSpan(self, operation_name, attributes, is_async=True)
        self._span_stack.append(span_obj)
        self._current_span = span_obj
        try:
            async with span_obj:
                yield span_obj
        finally:
            self._span_stack.pop()
            self._current_span = self._span_stack[-1] if self._span_stack else None
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

    def add_trace(
        self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a trace event."""
        trace_metadata = metadata or {}
        if self._current_span:
            trace_metadata.update(
                {
                    'span_id': self._current_span.span_id,
                    'trace_id': self._current_span.trace_id,
                }
            )
        else:
            trace_metadata.update({'span_id': None, 'trace_id': self._trace_id})

        self._metadata.add_trace(
            event_type,
            message,
            trace_metadata,
            span_id=trace_metadata.get('span_id'),
            trace_id=trace_metadata.get('trace_id'),
        )
        self.logger.log(
            logging.DEBUG, f'[{event_type}] {message}', extra=trace_metadata
        )

    def start(self, **attributes) -> None:
        """Start execution tracking."""
        self._metadata.status = Status.RUNNING
        self.timer.start()
        self._metadata.start_time = self.timer.start_timestamp
        self.add_trace('execution_start', 'Execution started', attributes)

    def complete(
        self, output_data: Optional[Dict[str, Any]] = None, **attributes
    ) -> None:
        """Complete execution tracking."""
        self.timer.stop()
        self._metadata.status = Status.COMPLETED
        self._metadata.end_time = self.timer.end_timestamp
        self._metadata.latency = self.timer.elapsed_time
        self._metadata.output_data = output_data or {}

        self.add_trace(
            'execution_complete',
            'Execution completed successfully',
            {'latency': self._metadata.latency, **attributes},
        )

    def fail(self, error: str, **attributes) -> None:
        """Handle execution failure."""
        self.timer.stop()
        self._metadata.status = Status.FAILED
        self._metadata.end_time = self.timer.end_timestamp
        self._metadata.latency = self.timer.elapsed_time
        self._metadata.error = error

        self.logger.error(f'Execution failed: {self._metadata.name} - {error}')
        self.add_trace(
            'execution_failed',
            f'Execution failed: {error}',
            {'error': error, 'latency': self._metadata.latency, **attributes},
        )

    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self._client:
            try:
                self._client.flush()
                logger.info('Langfuse traces flushed successfully')
            except Exception as e:
                logger.warning(f'Failed to flush Langfuse traces: {e}')

    def shutdown(self) -> None:
        """Gracefully shutdown the tracer."""
        self.flush()
        if self._client:
            try:
                self._client.shutdown()
                logger.debug('Langfuse client shutdown successfully')
            except Exception as e:
                logger.warning(f'Failed to shutdown Langfuse client: {e}')

    def info(self, msg: Any) -> None:
        """Log info message."""
        logger.info(msg)

    def log_performance(self, name: str, duration: float, **attributes) -> None:
        """Log a performance metric."""
        self.logger.log_performance(name, duration, **attributes)

    def log_table(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        columns: Optional[List[str]] = None,
        level: Union[int, str] = logging.INFO,
    ) -> None:
        """Log a structured table of data."""
        self.logger.log_table(data, title, columns, level)

    def log_llm_call(self, *args, **kwargs) -> None:
        """Log an LLM call to Langfuse as a generation."""
        if not self._client:
            return

        try:
            # Extract common LLM call attributes
            model = kwargs.get('model', 'unknown')
            prompt = kwargs.get('prompt', kwargs.get('input', ''))
            response = kwargs.get('response', kwargs.get('output', ''))

            # Extract token counts
            prompt_tokens = kwargs.get('prompt_tokens', 0)
            completion_tokens = kwargs.get('completion_tokens', 0)
            latency = kwargs.get('latency')
            cost = kwargs.get('cost_estimate')

            # Use parent span name if available, otherwise default to 'llm_call'
            default_name = self._current_span.name if self._current_span else 'llm_call'
            name = kwargs.get('name', default_name)

            # Create a generation observation and update with usage
            with self._client.start_as_current_observation(
                as_type='generation',
                name=name,
                model=model,
                input=prompt,
                output=response,
            ) as observation:
                # Update with usage data
                observation.update(
                    usage={
                        'input': prompt_tokens,
                        'output': completion_tokens,
                        'total': prompt_tokens + completion_tokens,
                    },
                    metadata={
                        'latency': latency,
                        'cost_estimate': cost,
                        'provider': kwargs.get('provider', 'unknown'),
                    },
                )
        except Exception as e:
            logger.info(f'Failed to log LLM call to Langfuse: {e}')

    def log_retrieval_call(self, *args, **kwargs) -> None:
        """Log a retrieval call."""
        self.add_trace('retrieval_call', 'Retrieval call', kwargs)

    def log_database_query(self, *args, **kwargs) -> None:
        """Log a database query."""
        self.add_trace('database_query', 'Database query', kwargs)

    def log_evaluation(self, *args, **kwargs) -> None:
        """Log an evaluation."""
        if not self._client:
            return

        try:
            # Extract evaluation attributes
            name = kwargs.get('name', 'evaluation')
            score = kwargs.get('score')
            comment = kwargs.get('comment', '')

            # Log as a score in Langfuse
            if score is not None:
                self._client.score(
                    name=name,
                    value=score,
                    comment=comment,
                )
        except Exception as e:
            logger.debug(f'Failed to log evaluation to Langfuse: {e}')

    def get_statistics(self) -> Dict[str, Any]:
        """Get basic execution statistics."""
        return {
            'execution_id': str(self._metadata.id),
            'status': (
                self._metadata.status.value if self._metadata.status else 'unknown'
            ),
            'start_time': self._metadata.start_time,
            'end_time': self._metadata.end_time,
            'latency': self._metadata.latency,
            'traces_count': (
                len(self._metadata.traces) if hasattr(self._metadata, 'traces') else 0
            ),
        }

    def display_statistics(self) -> None:
        """Display basic execution statistics."""
        stats = self.get_statistics()
        if not stats:
            self.info('No statistics available')
            return

        table = [
            {
                'metric': 'Execution ID',
                'value': stats['execution_id'][:8] + '...',
                'details': f'Status: {stats["status"]}',
            },
            {
                'metric': 'Execution Time',
                'value': (
                    f'{stats.get("latency", 0):.3f}s' if stats.get('latency') else 'N/A'
                ),
                'details': f'Traces: {stats.get("traces_count", 0)}',
            },
        ]
        self.log_table(table, title='Execution Statistics')

    def get_llm_statistics(self) -> Dict[str, Any]:
        """Get LLM statistics - for backward compatibility."""
        return self.get_statistics()

    def display_llm_statistics(self) -> None:
        """Display LLM statistics - for backward compatibility."""
        self.display_statistics()

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get Knowledge statistics - for backward compatibility."""
        return self.get_statistics()

    def display_knowledge_statistics(self) -> None:
        """Display Knowledge statistics - for backward compatibility."""
        self.display_statistics()

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get Database statistics - for backward compatibility."""
        return self.get_statistics()

    def display_database_statistics(self) -> None:
        """Display Database statistics - for backward compatibility."""
        self.display_statistics()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata with current span context."""
        metadata = self._metadata.model_dump()
        if self._current_span:
            metadata['current_span_id'] = self._current_span.span_id
            metadata['trace_id'] = self.trace_id
        return metadata

    def display_traces(self) -> None:
        """Display traces in a formatted table."""
        if not self._metadata.traces:
            self.logger.info('No traces recorded')
            return
        trace_data = [
            {
                'timestamp': trace.timestamp.strftime('%H:%M:%S.%f')[:-3],
                'event_type': trace.event_type,
                'message': trace.message[:50],
                'span_id': trace.span_id or 'N/A',
            }
            for trace in self._metadata.traces
        ]
        self.log_table(trace_data, title='Execution Traces')

    @property
    def metadata(self) -> BaseExecutionMetadata:
        """Get the metadata instance."""
        return self._metadata

    @property
    def current_span(self) -> Optional[LangfuseSpan]:
        """Get the current active span."""
        return self._current_span

    @property
    def trace_id(self) -> str:
        """Get the current trace ID."""
        if self._current_span:
            return self._current_span.trace_id
        return self._trace_id

    @property
    def handler(self):
        """Return None for handler since Langfuse doesn't use the handler pattern."""
        return None

    def __getattr__(self, name: str) -> Any:
        """
        Fallback for undefined methods to ensure forward compatibility.

        This allows the tracer to gracefully handle calls to methods that may
        exist in other tracer implementations but aren't implemented here.
        """

        def noop(*args, **kwargs):
            if name.startswith('get_'):
                return {} if 'statistics' in name else None
            return None

        return noop
