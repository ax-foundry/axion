import asyncio
import functools
import logging
import os
import pprint
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

from axion._core.environment import TracingMode, settings
from axion._core.logging import RichLogger, get_logger
from axion._core.metadata.schema import (
    BaseExecutionMetadata,
    DBExecutionMetadata,
    KnowledgeExecutionMetadata,
    LLMExecutionMetadata,
    Status,
    ToolMetadata,
)
from axion._core.tracing import reset_tracer_context, set_current_tracer
from axion._core.tracing.handlers import (
    MAX_ARG_LENGTH,
    MAX_ERROR_LENGTH,
    MAX_RESULT_LENGTH,
    BaseTraceHandler,
    DefaultTraceHandler,
)
from axion._core.tracing.handlers.knowledge_handler import KnowledgeTraceHandler
from axion._core.tracing.handlers.llm_handler import (
    EvaluationTraceHandler,
    LLMTraceHandler,
)
from axion._core.tracing.logfire.span import Span
from axion._core.utils import Timer
from axion._core.uuid import uuid7
from pydantic import BaseModel

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

logger = get_logger(__name__)


class LogfireTracer:
    """
    Tracer that combines tracing, logging, and metadata collection.
    Uses composition with a logger instance instead of inheritance.
    """

    def __init__(
        self,
        metadata_type: str,
        tool_metadata: Optional[ToolMetadata] = None,
        enable_logfire: bool = True,
        trace_id: Optional[str] = None,
        **kwargs,
    ):
        self.metadata_type = metadata_type
        self.tool_metadata = tool_metadata or self._infer_default_tool_meta()
        self.enable_logfire = enable_logfire
        self.kwargs = kwargs
        self.logger: RichLogger = get_logger(
            f'axion.tracing.{self.tool_metadata.name}_{metadata_type}'
        )
        self.timer = Timer()

        # Tracing state
        self._current_span: Optional[Span] = None
        self._span_stack: List[Span] = []
        self._trace_id = trace_id or str(uuid7())

        # Metadata and handler
        self._metadata = self._create_metadata()
        self._handler = self._create_metadata_handler()

        self.enable_logfire = LOGFIRE_AVAILABLE and settings.tracing_mode in {
            TracingMode.LOGFIRE_LOCAL,
            TracingMode.LOGFIRE_HOSTED,
            TracingMode.LOGFIRE_OTEL,
        }
        if self.enable_logfire:
            self._configure_logfire()

    def _infer_default_tool_meta(self) -> ToolMetadata:
        """Infer default metadata from the calling module (not recommended for production)."""
        import inspect

        frame = inspect.currentframe().f_back.f_back
        module = frame.f_globals.get('__name__', 'unknown')
        return ToolMetadata(
            name=f'tracer_{module}',
            description='Auto-generated tracer metadata',
            owner='unknown',
            version='1.0.0',
        )

    def _configure_logfire(self):
        """Configure Logfire based on the explicit `settings.tracing_mode`."""
        try:
            config = {
                'service_name': settings.logfire_service_name
                or self.tool_metadata.name,
                'distributed_tracing': settings.logfire_distributed_tracing,
                'console': settings.logfire_console_logging,
            }

            # Local Development Mode
            if settings.tracing_mode == TracingMode.LOGFIRE_LOCAL:
                self.logger.debug(
                    'Configuring Logfire for Local Development UI (`logfire dev`).'
                )
                logfire.configure(**config)

            # Hosted Cloud Mode
            elif settings.tracing_mode == TracingMode.LOGFIRE_HOSTED:
                if not settings.logfire_token:
                    self.logger.warning(
                        "tracing_mode is 'logfire_hosted' but no token was provided. Disabling Logfire."
                    )
                    self.enable_logfire = False
                    return

                self.logger.debug('Configuring Logfire for Hosted Cloud UI.')
                config.update(
                    {
                        'token': settings.logfire_token,
                        'send_to_logfire': True,
                    }
                )
                if settings.logfire_project_name:
                    config['project_name'] = settings.logfire_project_name
                logfire.configure(**config)

            # Custom OpenTelemetry Endpoint Mode
            elif settings.tracing_mode == TracingMode.LOGFIRE_OTEL:
                if not settings.otel_endpoint:
                    self.logger.warning(
                        "tracing_mode is 'logfire_otel' but no endpoint was provided. Disabling Logfire."
                    )
                    self.enable_logfire = False
                    return

                self.logger.debug(
                    f'Configuring Logfire for custom OTEL endpoint: {settings.otel_endpoint}'
                )
                os.environ['OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'] = (
                    settings.otel_endpoint
                )
                logfire.configure(**config)

            self.logger.debug(
                f"Logfire successfully configured for service: {config['service_name']}"
            )

        except Exception as e:
            self.logger.warning(f'Failed to configure Logfire: {e}')
            self.enable_logfire = False

    def _create_metadata(self) -> BaseExecutionMetadata:
        """Create the appropriate metadata instance based on type."""
        metadata_classes = {
            'base': BaseExecutionMetadata,
            'llm': LLMExecutionMetadata,
            'knowledge': KnowledgeExecutionMetadata,
            'database': DBExecutionMetadata,
        }
        metadata_class = metadata_classes.get(self.metadata_type, BaseExecutionMetadata)
        return metadata_class(
            name=self.tool_metadata.name, tool_metadata=self.tool_metadata.to_dict()
        )

    def _create_metadata_handler(self) -> BaseTraceHandler:
        """Create the appropriate handler based on metadata type."""
        handlers = {
            'llm': LLMTraceHandler,
            'knowledge': KnowledgeTraceHandler,
            'evaluation': EvaluationTraceHandler,
        }
        handler_class = handlers.get(self.metadata_type, DefaultTraceHandler)
        return handler_class(self)

    @contextmanager
    def span(self, operation_name: str, **attributes):
        """
        Creates a synchronous span AND sets the tracer context for its duration.
        Uses shared context management from main tracer module.
        """
        token = set_current_tracer(self)
        span_obj = Span(operation_name, self, attributes)
        try:
            with span_obj as s:
                yield s
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
            span_obj.__enter__()
            yield span_obj
        except Exception as e:
            span_obj.__exit__(type(e), e, e.__traceback__)
            raise
        else:
            span_obj.__exit__(None, None, None)
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
        self._add_trace_internal(event_type, message, trace_metadata)

    def _add_trace_internal(
        self, event_type: str, message: str, metadata: Dict[str, Any]
    ):
        """Internal method to add trace to metadata and log it."""
        self._metadata.add_trace(
            event_type,
            message,
            metadata,
            span_id=metadata.get('span_id'),
            trace_id=metadata.get('trace_id'),
        )
        self.logger.log(logging.DEBUG, f'[{event_type}] {message}', extra=metadata)

    @staticmethod
    def _get_log_level_for_event(event_type: str) -> int:
        """Get appropriate log level for event type."""
        if 'fail' in event_type or 'error' in event_type:
            return logging.ERROR
        return logging.DEBUG

    def start(self, **attributes):
        """Start execution tracking."""
        self._metadata.status = Status.RUNNING
        self.timer.start()
        self._metadata.start_time = self.timer.start_timestamp
        self.add_trace('execution_start', 'Execution started', attributes)

    def info(self, msg: Any):
        """Log info"""
        logger.info(msg)

    def log_performance(self, name, duration, **attributes):
        """Logs a performance metric with an optional set of attributes."""
        return self.logger.log_performance(name, duration, **attributes)

    def log_table(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        columns: Optional[List[str]] = None,
        level: Union[int, str] = logging.INFO,
    ) -> None:
        """Logs a structured table of data for easier readability and debugging."""
        return self.logger.log_table(data, title, columns, level)

    def complete(self, output_data: Dict[str, Any] = None, **attributes):
        """Complete execution tracking."""
        self.timer.stop()
        self._metadata.status = Status.COMPLETED
        self._metadata.end_time = self.timer.end_timestamp
        self._metadata.latency = self.timer.elapsed_time
        self._metadata.output_data = output_data or {}

        self.log_performance(self._metadata.name, self._metadata.latency, **attributes)
        self.add_trace(
            'execution_complete',
            'Execution completed successfully',
            {'latency': self._metadata.latency, **attributes},
        )

    def fail(self, error: str, **attributes):
        """Handle execution failure."""
        self.timer.stop()
        self._metadata.status = Status.FAILED
        self._metadata.end_time = self.timer.end_timestamp
        self._metadata.latency = self.timer.elapsed_time
        self._metadata.error = error

        # Log to Logfire if available
        if self.enable_logfire and LOGFIRE_AVAILABLE:
            try:
                logfire.error(
                    f'Execution failed: {self._metadata.name}',
                    execution_id=str(self._metadata.id),
                    error_message=error,
                    latency=self._metadata.latency,
                    **attributes,
                )
            except Exception:
                pass

        self.logger.error_highlight(
            f'Execution failed: {self._metadata.name} - {error}'
        )
        self.add_trace(
            'execution_failed',
            f'Execution failed: {error}',
            {'error': error, 'latency': self._metadata.latency, **attributes},
        )

    def log_llm_call(self, *args, **kwargs):
        """Log an LLM call - delegates to handler."""
        if hasattr(self._handler, 'log_llm_call'):
            return self._handler.log_llm_call(*args, **kwargs)
        raise AttributeError(
            f'Trace handler {type(self._handler).__name__} does not support LLM calls'
        )

    def log_retrieval_call(self, *args, **kwargs):
        """Log a retrieval call - delegates to handler."""
        if hasattr(self._handler, 'log_retrieval_call'):
            return self._handler.log_retrieval_call(*args, **kwargs)
        raise AttributeError(
            f'Trace handler {type(self._handler).__name__} does not support retrieval calls'
        )

    def log_database_query(self, *args, **kwargs):
        """Log a database query - delegates to handler."""
        if hasattr(self._handler, 'log_database_query'):
            return self._handler.log_database_query(*args, **kwargs)
        raise AttributeError(
            f'Trace handler {type(self._handler).__name__} does not support database queries'
        )

    def log_evaluation(self, *args, **kwargs):
        """Log an evaluation - delegates to evaluation handler."""
        if hasattr(self._handler, 'log_evaluation'):
            return self._handler.log_evaluation(*args, **kwargs)
        raise AttributeError(
            f'Trace handler {type(self._handler).__name__} does not support evaluation calls'
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from the current handler."""
        return self._handler.get_statistics()

    def display_statistics(self):
        """Display statistics using the current handler."""
        return self._handler.display_statistics()

    def get_llm_statistics(self) -> Dict[str, Any]:
        """Get LLM statistics - for backward compatibility."""
        if hasattr(self._handler, 'get_statistics'):
            return self._handler.get_statistics()
        return {}

    def display_llm_statistics(self):
        """Display LLM statistics - for backward compatibility."""
        if hasattr(self._handler, 'display_statistics'):
            return self._handler.display_statistics()

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get Knowledge statistics - for backward compatibility."""
        if hasattr(self._handler, 'get_statistics'):
            return self._handler.get_statistics()
        return {}

    def display_knowledge_statistics(self):
        """Display Knowledge statistics - for backward compatibility."""
        if hasattr(self._handler, 'display_statistics'):
            return self._handler.display_statistics()

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get Database statistics - for backward compatibility."""
        if hasattr(self._handler, 'get_statistics'):
            return self._handler.get_statistics()
        return {}

    def display_database_statistics(self):
        """Display Database statistics - for backward compatibility."""
        if hasattr(self._handler, 'display_statistics'):
            return self._handler.display_statistics()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata with current span context."""
        metadata = self._metadata.model_dump()
        if self._current_span:
            metadata['current_span_id'] = self._current_span.span_id
            metadata['trace_id'] = self.trace_id
        return metadata

    def display_traces(self):
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
        if hasattr(self.logger, 'log_table'):
            self.logger.log_table(trace_data, title='Execution Traces')
        else:
            self.logger.info(f'Execution Traces:\n{pprint.pformat(trace_data)}')

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
    def handler(self) -> BaseTraceHandler:
        return self._handler

    @classmethod
    def create(
        cls,
        metadata_type: str,
        tool_metadata: Optional[ToolMetadata] = None,
        **kwargs,
    ) -> 'LogfireTracer':
        """Factory method to create a LogfireTracer."""
        return cls(metadata_type, tool_metadata, **kwargs)

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
        """
        Decorator to automatically trace function execution with optional argument/result capture.
        """

        def build_operation_name(obj, func: Callable) -> str:
            if operation_name:
                return operation_name
            if use_class_name:
                suffix = operation_suffix or func.__name__
                return f'{obj.__class__.__name__}_{suffix}'
            return func.__name__

        def build_attributes(obj, func: Callable, args, kwargs) -> Dict[str, Any]:
            attrs = dict(span_attributes)

            if use_class_name or operation_name is None:
                attrs['class_name'] = obj.__class__.__name__
                attrs['method_name'] = func.__name__
                if use_class_name and operation_suffix:
                    attrs['operation_suffix'] = operation_suffix

            if capture_args:
                safe_args = []
                for arg in args:
                    if isinstance(arg, BaseModel):
                        safe_args.append(arg.model_dump_json())
                    elif (
                        isinstance(arg, (str, int, float, bool))
                        and len(str(arg)) < MAX_ARG_LENGTH
                    ):
                        safe_args.append(str(arg))
                    else:
                        safe_args.append(f'<{type(arg).__name__}>')

                safe_kwargs = {
                    k: (
                        v
                        if isinstance(v, (str, int, float, bool))
                        and len(str(v)) < MAX_ARG_LENGTH
                        else f'<{type(v).__name__}>'
                    )
                    for k, v in kwargs.items()
                }

                attrs['function_args'] = safe_args
                attrs['function_kwargs'] = safe_kwargs

            return attrs

        def capture_result_attributes(span, result: Any):
            if isinstance(result, BaseModel):
                span.set_attribute('result', result.model_dump_json())
            elif (
                isinstance(result, (str, int, float, bool, dict))
                and len(str(result)) < MAX_RESULT_LENGTH
            ):
                span.set_attribute('result', result)
            else:
                span.set_attribute('result_type', type(result).__name__)

        def handle_exception(span, exc: Exception):
            span.set_attribute('error_type', type(exc).__name__)
            span.set_attribute('error_message', str(exc)[:MAX_ERROR_LENGTH])

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Use shared context management to get current tracer
                from axion._core.tracing import get_current_tracer

                tracer = get_current_tracer()

                # The rest of the logic is the same, but it uses 'tracer'
                self_obj = args[0] if args else None
                op_name = build_operation_name(self_obj, func)
                attrs = build_attributes(self_obj, func, args[1:], kwargs)

                with tracer.span(op_name, auto_trace=auto_trace, **attrs) as span:
                    try:
                        result = func(*args, **kwargs)
                        if capture_result and result is not None:
                            capture_result_attributes(span, result)
                        return result
                    except Exception as e:
                        handle_exception(span, e)
                        raise

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Use shared context management to get current tracer
                from axion._core.tracing import get_current_tracer

                tracer = get_current_tracer()

                # The rest of the logic is the same
                self_obj = args[0] if args else None
                op_name = build_operation_name(self_obj, func)
                attrs = build_attributes(self_obj, func, args[1:], kwargs)

                async with tracer.async_span(
                    op_name, auto_trace=auto_trace, **attrs
                ) as span:
                    try:
                        result = await func(*args, **kwargs)
                        if capture_result and result is not None:
                            capture_result_attributes(span, result)
                        return result
                    except Exception as e:
                        handle_exception(span, e)
                        raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    @staticmethod
    def trace(
        operation_suffix: str = None,
        capture_args: bool = False,
        capture_result: bool = False,
        **span_attributes,
    ):
        """
        Convenience decorator for class-aware operation tracing.

        Usage:
        @Tracer.trace(name="execute", capture_args=True)
        async def execute(self, input_data):
            # Creates span named "{ClassName}_execute"
            pass

        @Tracer.trace() # Uses method name
        async def process(self, data):
            # Creates span named "{ClassName}_process"
            pass

        Args:
            operation_suffix: Suffix to append to class name (defaults to method name)
            capture_args: Whether to capture function arguments
            capture_result: Whether to capture function result
            **span_attributes: Additional attributes to set on the span
        """
        return LogfireTracer.traced_operation(
            use_class_name=True,
            operation_suffix=operation_suffix,
            capture_args=capture_args,
            capture_result=capture_result,
            **span_attributes,
        )


def trace(
    operation_suffix: str = None,
    capture_args: bool = False,
    capture_result: bool = False,
    **span_attributes,
):
    """
    Factory function for class-aware operation tracing.

    This is a convenience wrapper around LogfireTracer.traced_operation
    that enables class-aware naming by default.

    Usage:
    @trace(name="execute", capture_args=True)
    async def execute(self, input_data):
        # Creates span named "{ClassName}_execute"
        pass

    @trace() # Uses method name
    async def process(self, data):
        # Creates span named "{ClassName}_process"
        pass

    @trace(capture_result=True, custom_attr="value")
    def compute(self, x, y):
        # Creates span named "{ClassName}_compute" with custom attributes
        return x + y

    Args:
        operation_suffix: Suffix to append to class name (defaults to method name)
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture function result
        **span_attributes: Additional attributes to set on the span

    Returns:
        Decorator function that can be applied to methods
    """
    return LogfireTracer.traced_operation(
        use_class_name=True,
        operation_suffix=operation_suffix,
        capture_args=capture_args,
        capture_result=capture_result,
        **span_attributes,
    )
