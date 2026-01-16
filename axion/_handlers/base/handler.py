from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, Generic, List, Optional, Union

from axion._core.error import InvalidConfig
from axion._core.logging import get_logger
from axion._core.metadata.schema import BaseExecutionMetadata, ToolMetadata
from axion._core.schema import Callbacks, InputModel, OutputModel
from axion._core.tracing import init_tracer
from axion._core.tracing.handlers import BaseTraceHandler

logger = get_logger(__name__)


class BaseHandler(ABC, Generic[InputModel, OutputModel]):
    """
    Abstract base class for AXION handlers with unified tracing and metadata tracking.

    This class provides the foundation for all handler implementations in AXION,
    establishing a consistent interface and lifecycle for processing requests. It handles
    input validation, unified metadata tracking, execution, and output formatting.

    Attributes:
        input_model (Type[InputModel]): Model class for validating and parsing input data.
        output_model (Type[OutputModel]): Model class for structuring handler outputs.
        name (str): Handler name, defaults to snake_case of class name if not provided.
        description (str): Human-readable description of handler functionality.
        task (str): Category or classification of the handler's purpose.
        owner (str): Owner or team responsible for the handler.
        version (str): Semantic version of the handler implementation.
        handler_type (str): Name of the prompt template used by this handler.
        max_retries (int): Maximum number of execution retry attempts before failing.
        retry_delay (float): Delay in seconds between retry attempts.
        verbose (bool): Whether to enable verbose logging during execution.

    Usage:
        Subclass BaseHandler and implement the execute() method to create custom handlers:

        ```python
        class MyCustomHandler(BaseHandler):
            input_model = MyInputModel
            output_model = MyOutputModel
            description = "Processes custom inputs"
            owner = "AI Team"
            handler_type = "base"

            @Tracer.trace(name="custom_execute", capture_args=True)
            async def execute(self, data, callbacks=None):
                # Automatic unified span for entire execution
                async with self._generation_context(callbacks):
                    with self.span("custom_processing") as span:
                        processed_input = self.process_input(data)
                        # Custom processing logic
                        result = self.output_model(...)
                        return self.process_output(result, data)
        ```
    Lifecycle:
        1. Input validation and parsing via input_model
        2. Optional input preprocessing via process_input()
        3. Execution of handler logic via execute() (wrapped in unified spans)
        4. Optional output postprocessing via process_output()
        5. Unified metadata collection throughout the process
    """

    _metadata: BaseExecutionMetadata = None
    tracer: BaseTraceHandler = None

    input_model: Optional[InputModel] = None
    output_model: Optional[OutputModel] = None

    _name: str
    description: str = None
    task: str = None
    owner: str = None
    version: str = '1.0.0'
    handler_type: str = 'base'

    max_retries: int = 3
    retry_delay: float = 0.5
    validate_base: bool = False
    verbose: bool = False

    def __init__(self, **kwargs):
        """
        Initialize the handler with configuration and tracing support.

        Args:
            **kwargs: Additional configuration parameters that match class attributes.
                     Special kwargs:
                     - tracer: Tracer object
                     - enable_logfire: bool = True
        """

        self.enable_logfire = kwargs.pop('enable_logfire', True)
        self.metadata_type = self.handler_type
        self.tracer = kwargs.get('tracer')

        self._set_kwargs(**kwargs)
        if self.validate_base:
            self._validate_required_fields(['description', 'owner'])
        if not self.tracer:
            self.set_tracer()

    def _set_kwargs(self, **kwargs) -> None:
        """Set instance attributes from provided keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _validate_required_fields(self, required_fields: list) -> None:
        """
        Validate that required configuration fields are provided.

        Args:
            required_fields (list): List of attribute names that must be defined.
        """
        missing_fields = [
            field for field in required_fields if not getattr(self, field, None)
        ]
        if missing_fields:
            raise InvalidConfig(f'Missing required fields: {", ".join(missing_fields)}')

    def get_tool_meta(self) -> ToolMetadata:
        """Get tool metadata from object."""
        return ToolMetadata(
            name=getattr(self, 'name', None) or self.__class__.__name__,
            description=self.description,
            task=self.task,
            owner=self.owner,
            version=self.version,
        )

    def set_tracer(self) -> None:
        """
        Set up unified tracer for tracking and observability.

        Initializes the simplified Tracer with environment-based configuration.
        """
        tool_metadata = self.get_tool_meta()

        self.tracer = init_tracer(
            metadata_type=self.metadata_type,
            tool_metadata=tool_metadata,
            tracer=self.tracer,
        )

    def format_input_data(
        self, input_data: Union[InputModel, Dict[str, str], str]
    ) -> Union[InputModel, Dict[str, str], str]:
        """
        Format and validate input data to match the expected input model.

        Args:
            input_data (Union[InputModel, Dict[str, str], str]): Raw input data in various formats.

        Returns:
            InputModel: Validated and structured input data.
        """
        if isinstance(input_data, str):
            raise AssertionError(
                f'Must Pass {self.input_model.__name__} or Dictionary Mapping.'
            )
        if isinstance(input_data, dict):
            input_data = self.input_model(**input_data)
        return input_data

    def span(self, operation_name: str, **attributes):
        """
        Convenience function to create a unified span for operation tracking.

        Args:
            operation_name: Name of the operation being tracked.
            **attributes: Additional attributes to attach to the span.
        """
        return self.tracer.span(operation_name, **attributes)

    def async_span(self, operation_name: str, **attributes):
        """
        Convenience function to create an async unified span for operation tracking.

        Args:
            operation_name: Name of the operation being tracked.
            **attributes: Additional attributes to attach to the span.
        """
        return self.tracer.async_span(operation_name, **attributes)

    def add_trace(
        self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Convince function to cdd a trace event (use sparingly - spans with auto_trace handle most cases).

        Args:
            event_type: Type of event (e.g., 'validation_passed', 'config_loaded')
            message: Human-readable message describing the event
            metadata: Optional additional metadata for the event
        """
        self.tracer.add_trace(event_type, message, metadata)

    @asynccontextmanager
    async def _generation_context(self, callbacks: Callbacks = None):
        """
        Context manager for handling generation lifecycle and callbacks.

        Args:
            callbacks: Callbacks for tracking progress.
        """
        callbacks = callbacks or []
        try:
            self.tracer.start(handler_type=self.__class__.__name__)

            # Execute callback hooks
            for cb in callbacks:
                if hasattr(cb, 'on_generation_start'):
                    await cb.on_generation_start()

            yield

        except Exception as e:
            self.tracer.fail(str(e))

            for cb in callbacks:
                if hasattr(cb, 'on_generation_error'):
                    await cb.on_generation_error(error=e)
            raise

        finally:
            self.tracer.complete()

            for cb in callbacks:
                if hasattr(cb, 'on_generation_end'):
                    await cb.on_generation_end()

    @abstractmethod
    async def execute(
        self,
        data: Union[InputModel, Dict[str, Any]],
        callbacks: List[Any] = None,
    ) -> OutputModel:
        """
        Execute the handler's core logic with the provided input data.

        This abstract method must be implemented by all subclasses to define
        the specific processing logic of the handler. The unified tracing system
        will automatically track the execution.

        Args:
            data (Union[InputModel, Dict[str, Any]]): Input data to process.
            callbacks (List[Any], optional): Callback functions for execution tracking.

        Returns:
            OutputModel: The structured result of the handler's processing.
        """
        pass

    def process_input(self, input_data: InputModel) -> InputModel:
        """
        Process input data before main execution.

        This method can be overridden to perform custom preprocessing on the input
        data before it is passed to the execute method.
        """
        return input_data

    def process_output(self, output_data: Any, input_data: InputModel = None) -> Any:
        """
        Process output data after main execution.

        This method can be overridden to perform custom postprocessing on the output
        data after it is generated by the execute method.
        """
        return output_data

    def display_traces(self):
        """Display all captured trace events in a formatted table."""
        self.tracer.display_traces()

    def get_traces(self) -> List[Dict[str, Any]]:
        """Get all captured trace events."""
        return [trace.model_dump() for trace in self.tracer.metadata.traces]

    def display_execution_statistics(self):
        """Display execution statistics based on the tracer's meta_type."""
        method_name = f'display_{self.tracer.metadata_type}_statistics'
        display_method = getattr(self.tracer, method_name, None)

        if callable(display_method):
            display_method()
        else:
            logger.info('No specialized statistics available for this handler type.')

    def get_execution_metadata(self) -> Dict[str, Any]:
        """
        Retrieve the collected execution metadata from unified tracker.

        Returns:
            Dict[str, Any]: Dictionary of execution metadata.
        """
        return self.tracer.get_metadata()

    def display_execution_metadata(self) -> None:
        """Display execution metadata in a human-readable format."""
        try:
            from axion._core.display import display_execution_metadata

            display_execution_metadata(self.get_execution_metadata())
        except ImportError:
            # Fallback to simple display
            metadata = self.get_execution_metadata()
            logger.info(f'Execution Metadata: {metadata}')

    def display_prompt(self) -> None:
        """
        Display the prompt template used by this handler.

        This method should be implemented by subclasses to show the prompt
        template in a human-readable format.
        """
        if self.handler_type:
            logger.info(f'Handler uses prompt: {self.handler_type}')
        else:
            logger.info('No prompt template specified for this handler')

    def get_span_context(self) -> tuple[Optional[str], Optional[str]]:
        """Get current span and trace IDs."""
        return self.tracer.get_current_span_context()

    def get_tracing_status(self) -> Dict[str, Any]:
        """Get current tracing configuration status."""
        status = {
            'handler_name': self.name,
            'handler_type': self.__class__.__name__,
            'metadata_type': self.metadata_type,
            'logfire_enabled': self.tracer.enable_logfire,
            'current_span_active': self.tracer._current_span is not None,
        }

        # Add tracer-specific status if available
        if hasattr(self.tracer, 'check_logfire_status'):
            status['logfire_status'] = self.tracer.check_logfire_status()

        return status

    @property
    def name(self):
        """Getter for the 'name' property."""
        return getattr(self, '_name', None) or self.__class__.__name__

    @name.setter
    def name(self, value):
        """Setter for the 'name' property."""
        self._name = value

    @name.deleter
    def name(self):
        """Deleter for the 'name' property."""
        self._name = None

    @property
    def metadata(self) -> BaseExecutionMetadata:
        """Get the execution metadata object."""
        return self.tracer.metadata

    @property
    def current_span(self):
        """Get the current active span."""
        return self.tracer.current_span

    @property
    def execution_time(self) -> Optional[float]:
        """Get execution time from metadata tracker."""
        if hasattr(self.tracer, 'metadata'):
            return getattr(self.tracer.metadata, 'latency', None)
        return None

    @property
    def execution_id(self) -> str:
        """Get the current execution ID."""
        return str(self.tracer.metadata.id)

    def __repr__(self):
        """Get a string representation of the handler."""
        return f'{self.__class__.__name__}(name={self.name}, metadata_type={self.metadata_type})'

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with unified error handling."""
        if exc_type:
            self.tracer.fail(f'Handler failed: {exc_val}')
        else:
            if hasattr(self.tracer.metadata, 'status'):
                if self.tracer.metadata.status != 'completed':
                    self.tracer.complete()
