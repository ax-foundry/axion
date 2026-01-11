from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, List, Optional, Type

from axion._core.logging import get_logger

logger = get_logger(__name__)

__all__ = ['BaseTracer', 'TracerRegistry']


class BaseTracer(ABC):
    """
    Abstract base class for all tracer implementations.

    This defines the core interface that all tracers (NoOp, Logfire, Langfuse, etc.)
    must implement. Users can create custom tracers by subclassing this and
    registering with @TracerRegistry.register('name').

    Example:
        @TracerRegistry.register('my_custom_tracer')
        class MyCustomTracer(BaseTracer):
            # implement abstract methods
            ...
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        metadata_type: str = 'default',
        tool_metadata: Optional[Any] = None,
        **kwargs,
    ) -> 'BaseTracer':
        """
        Factory method to create tracer instances.

        Args:
            metadata_type: Type of metadata (e.g., 'llm', 'knowledge', 'evaluation')
            tool_metadata: Optional metadata about the tool being traced
            **kwargs: Additional implementation-specific arguments

        Returns:
            A new tracer instance
        """
        pass

    @abstractmethod
    @contextmanager
    def span(self, operation_name: str, **attributes):
        """
        Create a synchronous span context manager.

        Args:
            operation_name: Name of the operation being traced
            **attributes: Additional attributes to attach to the span

        Yields:
            A span object that can be used to add attributes or events
        """
        pass

    @abstractmethod
    @asynccontextmanager
    async def async_span(self, operation_name: str, **attributes):
        """
        Create an asynchronous span context manager.

        Args:
            operation_name: Name of the operation being traced
            **attributes: Additional attributes to attach to the span

        Yields:
            A span object that can be used to add attributes or events
        """
        pass

    @abstractmethod
    def start(self, **attributes) -> None:
        """Start execution tracking."""
        pass

    @abstractmethod
    def complete(self, output_data: Optional[Dict[str, Any]] = None, **attributes) -> None:
        """Complete execution tracking."""
        pass

    @abstractmethod
    def fail(self, error: str, **attributes) -> None:
        """Handle execution failure."""
        pass

    @abstractmethod
    def add_trace(
        self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a trace event."""
        pass

    def flush(self) -> None:
        """
        Flush any pending traces to the backend.

        Override in implementations that buffer traces (e.g., Langfuse).
        Default implementation is a no-op.
        """
        pass

    def shutdown(self) -> None:
        """
        Gracefully shutdown the tracer.

        Override in implementations that need cleanup.
        Default implementation calls flush().
        """
        self.flush()

    def info(self, msg: Any) -> None:
        """Log info message."""
        logger.info(msg)


class TracerRegistry:
    """
    Registry for tracer implementations using decorator pattern.

    This class provides a simple way to register and retrieve tracer implementations
    by name, following the same pattern as LLMRegistry.

    Example:
        @TracerRegistry.register('my_tracer')
        class MyTracer(BaseTracer):
            ...

        # Later, retrieve the tracer class
        TracerClass = TracerRegistry.get('my_tracer')
        tracer = TracerClass.create(metadata_type='llm')
    """

    _registry: Dict[str, Type[BaseTracer]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseTracer]], Type[BaseTracer]]:
        """
        Decorator to register a tracer implementation.

        Args:
            name: The name to register the tracer under (e.g., 'noop', 'logfire', 'langfuse')

        Returns:
            Decorator function

        Example:
            @TracerRegistry.register('custom')
            class CustomTracer(BaseTracer):
                ...
        """

        def decorator(tracer_class: Type[BaseTracer]) -> Type[BaseTracer]:
            if name in cls._registry:
                logger.debug(
                    f"Tracer '{name}' already registered. Overwriting with {tracer_class.__name__}"
                )
            cls._registry[name] = tracer_class
            logger.debug(f"Tracer '{name}' registered: {tracer_class.__name__}")
            return tracer_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseTracer]:
        """
        Get a tracer class by name.

        Args:
            name: The registered name of the tracer

        Returns:
            The tracer class

        Raises:
            ValueError: If the tracer is not registered
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Tracer '{name}' not registered. Available tracers: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered tracer provider names.

        Returns:
            List of registered tracer names
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a tracer is registered.

        Args:
            name: The name to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry

    @classmethod
    def display(cls) -> None:
        """Display the tracer registry in a formatted table."""
        if not cls._registry:
            logger.info('No tracers registered')
            return

        table_data = [
            {'name': name, 'class': tracer_class.__name__}
            for name, tracer_class in cls._registry.items()
        ]
        logger.log_table(table_data, title='Registered Tracers')
