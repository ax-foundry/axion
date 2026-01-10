from abc import ABC
from typing import Any, Dict, List, Optional, Protocol

from axion._core.logging import get_logger
from axion._core.metadata.schema import BaseExecutionMetadata

logger = get_logger(__name__)

MAX_ARG_LENGTH = 500
MAX_RESULT_LENGTH = 2000
MAX_ERROR_LENGTH = 200


class TracerProtocol(Protocol):
    """Protocol defining the interface that handlers can use from the tracer."""

    def add_trace(
        self, event_type: str, message: str, metadata: Dict[str, Any] = None
    ) -> None:
        """Add a trace event to the tracer."""
        ...

    @staticmethod
    def log_performance(operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics."""
        ...

    @staticmethod
    def log_table(data: List[Dict[str, Any]], title: str = None) -> None:
        """Log data in table format."""
        ...

    @staticmethod
    def info(message: str) -> None:
        """Log info message."""
        ...

    def error_highlight(self, message: str) -> None:
        """Log error with highlighting."""
        ...

    @property
    def metadata(self) -> BaseExecutionMetadata:
        """Get the metadata instance."""
        ...

    @property
    def current_span(self):
        """Get the current active span."""
        ...

    @property
    def trace_id(self) -> str:
        """Get the current trace ID."""
        ...

    def span(self):
        pass


class BaseTraceHandler(ABC):
    """Base handler for metadata-specific functionality."""

    def __init__(self, tracer: TracerProtocol):
        self.tracer = tracer

    def get_span_context(self) -> tuple[Optional[str], Optional[str]]:
        """Get current span and trace context."""
        if self.tracer.current_span:
            return self.tracer.current_span.span_id, self.tracer.current_span.trace_id
        return None, self.tracer.trace_id

    def get_statistics(self) -> Dict[str, Any]:
        """Get handler-specific statistics."""
        return {}

    def display_statistics(self) -> None:
        """Display handler-specific statistics."""
        pass


class DefaultTraceHandler(BaseTraceHandler):
    """Default handler for base metadata types."""

    def get_statistics(self) -> Dict[str, Any]:
        """Get basic execution statistics."""
        return {
            'execution_id': str(self.tracer.metadata.id),
            'status': (
                self.tracer.metadata.status.value
                if self.tracer.metadata.status
                else 'unknown'
            ),
            'start_time': self.tracer.metadata.start_time,
            'end_time': self.tracer.metadata.end_time,
            'latency': self.tracer.metadata.latency,
            'traces_count': (
                len(self.tracer.metadata.traces)
                if hasattr(self.tracer.metadata, 'traces')
                else 0
            ),
        }

    @staticmethod
    def log_performance(operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics."""
        logger.log_performance(operation, duration, **kwargs)

    @staticmethod
    def log_table(data: List[Dict[str, Any]], title: str = None) -> None:
        """Log data in table format."""
        logger.log_table(data, title)

    @staticmethod
    def info(message: str) -> None:
        """Log info message."""
        logger.info(message)

    def display_statistics(self) -> None:
        """Display basic execution statistics."""
        stats = self.get_statistics()
        if not stats:
            self.tracer.info('No statistics available')
            return

        def fmt(num, precision=0, suffix=''):
            if num is None:
                return 'N/A'
            return f'{num:,.{precision}f}{suffix}'

        table = [
            {
                'metric': 'Execution ID',
                'value': stats['execution_id'][:8] + '...',
                'details': f"Status: {stats['status']}",
            },
            {
                'metric': 'Execution Time',
                'value': (
                    fmt(stats.get('latency'), 3, 's') if stats.get('latency') else 'N/A'
                ),
                'details': f"Traces: {stats.get('traces_count', 0)}",
            },
        ]

        self.tracer.log_table(table, title='Execution Statistics')
