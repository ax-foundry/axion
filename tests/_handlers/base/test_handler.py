import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from axion._core.error import InvalidConfig
from axion._core.metadata.schema import BaseExecutionMetadata, ToolMetadata
from axion._core.schema import OutputModel
from axion._core.tracing.handlers import BaseTraceHandler
from axion._handlers.base.handler import BaseHandler
from pydantic import BaseModel


class MockInputModel(BaseModel):
    """Mock input model for testing."""

    field1: str
    field2: int


class MockOutputModel(BaseModel):
    """Mock output model for testing."""

    result: str


class MockTraceHandler(BaseTraceHandler):
    """Mock trace handler for testing."""

    def __init__(self, **kwargs):
        self.metadata = MagicMock(spec=BaseExecutionMetadata)
        self.metadata.id = 'test-execution-id'
        self.metadata.latency = 1.5
        self.metadata.status = 'running'
        self.metadata.traces = []
        self.metadata_type = kwargs.get('metadata_type', 'base')
        self.enable_logfire = kwargs.get('enable_logfire', True)
        self._current_span = None

    @classmethod
    def create(cls, metadata_type: str = 'default', tool_metadata=None, **kwargs):
        """Factory method to create tracer instances."""
        return cls(metadata_type=metadata_type, **kwargs)

    def span(self, operation_name: str, **attributes):
        return MagicMock()

    @asynccontextmanager
    async def async_span(self, operation_name: str, **attributes):
        span_mock = MagicMock()
        span_mock.set_attribute = MagicMock()
        self._current_span = span_mock
        try:
            yield span_mock
        finally:
            self._current_span = None

    def add_trace(self, event_type: str, message: str, metadata: Dict[str, Any] = None):
        pass

    def start(self, **kwargs):
        self.metadata.status = 'running'

    def complete(self):
        self.metadata.status = 'completed'

    def fail(self, error_message: str):
        self.metadata.status = 'failed'

    def get_metadata(self) -> Dict[str, Any]:
        return {'execution_id': 'test-id', 'status': 'completed'}

    def display_traces(self):
        pass

    def get_current_span_context(self):
        return ('trace-id', 'span-id')

    @property
    def current_span(self):
        return self._current_span


class MockHandler(BaseHandler):
    """Mock implementation of BaseHandler for testing."""

    description = 'Test description'
    owner = 'Test Owner'
    handler_type = 'test'
    input_model = MockInputModel
    output_model = MockOutputModel
    validate_base = True

    async def execute(self, data, callbacks=None) -> OutputModel:
        async with self._generation_context(callbacks):
            with self.span('test_processing') as span:
                span.set_attribute('processing', True)
                return MockOutputModel(result='mock_result')


class MockHandlerNoValidation(BaseHandler):
    """Mock handler without validation for testing."""

    description = 'Test description'
    owner = 'Test Owner'
    handler_type = 'test'
    input_model = MockInputModel
    output_model = MockOutputModel
    validate_base = False

    async def execute(self, data, callbacks=None) -> OutputModel:
        return MockOutputModel(result='mock_result')


class TestBaseHandlerInitialization:
    """Test BaseHandler initialization."""

    @patch('axion._handlers.base.handler.init_tracer')
    def test_handler_initialization_with_defaults(self, mock_init_tracer):
        """Test successful initialization with default values."""
        mock_tracer = MockTraceHandler()
        mock_init_tracer.return_value = mock_tracer

        handler = MockHandlerNoValidation()

        assert handler.description == 'Test description'
        assert handler.owner == 'Test Owner'
        assert handler.handler_type == 'test'
        assert handler.version == '1.0.0'
        assert handler.max_retries == 3
        assert handler.retry_delay == 0.5
        assert not handler.verbose
        assert handler.enable_logfire
        assert handler.tracer == mock_tracer

    @patch('axion._handlers.base.handler.init_tracer')
    def test_handler_initialization_with_kwargs(self, mock_init_tracer):
        """Test initialization with custom kwargs."""
        mock_tracer = MockTraceHandler()
        mock_init_tracer.return_value = mock_tracer

        handler = MockHandlerNoValidation(
            version='2.0.0',
            max_retries=5,
            verbose=True,
            enable_logfire=False,
            custom_attr='custom_value',
        )

        assert handler.version == '2.0.0'
        assert handler.max_retries == 5
        assert handler.verbose
        assert handler.enable_logfire is False
        # Custom attributes not in class should be ignored
        assert not hasattr(handler, 'custom_attr')

    def test_missing_required_fields_validation(self):
        """Test validation when required fields are missing."""

        class InvalidHandler(BaseHandler):
            handler_type = 'invalid'
            validate_base = True

            # Missing description and owner

            async def execute(self, data, callbacks=None):
                return MockOutputModel(result='invalid_result')

        with pytest.raises(
            InvalidConfig, match='Missing required fields: description, owner'
        ):
            InvalidHandler()

    @patch('axion._handlers.base.handler.init_tracer')
    def test_provided_tracer(self, mock_init_tracer):
        """Test initialization with provided tracer."""
        custom_tracer = MockTraceHandler()

        handler = MockHandlerNoValidation(tracer=custom_tracer)

        # Should not call init_tracer when tracer is provided
        mock_init_tracer.assert_not_called()
        assert handler.tracer == custom_tracer


class TestBaseHandlerMetadata:
    """Test metadata-related functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('axion._handlers.base.handler.init_tracer') as mock_init:
            mock_init.return_value = MockTraceHandler()
            self.handler = MockHandlerNoValidation(name='test_handler')

    def test_get_tool_meta(self):
        """Test tool metadata creation."""
        tool_meta = self.handler.get_tool_meta()

        assert isinstance(tool_meta, ToolMetadata)
        assert tool_meta.name == 'test_handler'
        assert tool_meta.description == 'Test description'
        assert tool_meta.owner == 'Test Owner'
        assert tool_meta.version == '1.0.0'

    def test_get_tool_meta_default_name(self):
        """Test tool metadata with default class name."""
        with patch('axion._handlers.base.handler.init_tracer') as mock_init:
            mock_init.return_value = MockTraceHandler()
            handler = MockHandlerNoValidation()  # No name provided

        tool_meta = handler.get_tool_meta()
        assert tool_meta.name == 'MockHandlerNoValidation'

    def test_name_property(self):
        """Test name property getter/setter/deleter."""
        assert self.handler.name == 'test_handler'

        self.handler.name = 'new_name'
        assert self.handler.name == 'new_name'

        del self.handler.name
        assert (
            self.handler.name == 'MockHandlerNoValidation'
        )  # Falls back to class name

    def test_metadata_property(self):
        """Test metadata property."""
        metadata = self.handler.metadata
        assert metadata == self.handler.tracer.metadata

    def test_execution_time_property(self):
        """Test execution time property."""
        assert self.handler.execution_time == 1.5

    def test_execution_id_property(self):
        """Test execution ID property."""
        assert self.handler.execution_id == 'test-execution-id'

    def test_current_span_property(self):
        """Test current span property."""
        span = self.handler.current_span
        assert span == self.handler.tracer.current_span


class TestBaseHandlerInputOutput:
    """Test input/output processing."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('axion._handlers.base.handler.init_tracer') as mock_init:
            mock_init.return_value = MockTraceHandler()
            self.handler = MockHandlerNoValidation()

    def test_format_input_data_with_dict(self):
        """Test formatting input data from dictionary."""
        input_dict = {'field1': 'test', 'field2': 42}
        formatted = self.handler.format_input_data(input_dict)

        assert isinstance(formatted, MockInputModel)
        assert formatted.field1 == 'test'
        assert formatted.field2 == 42

    def test_format_input_data_with_model(self):
        """Test formatting input data that's already a model."""
        input_model = MockInputModel(field1='test', field2=42)
        formatted = self.handler.format_input_data(input_model)

        assert formatted == input_model

    def test_format_input_data_with_string_raises_error(self):
        """Test that string input raises AssertionError."""
        with pytest.raises(
            AssertionError, match='Must Pass MockInputModel or Dictionary Mapping'
        ):
            self.handler.format_input_data('invalid string input')

    def test_process_input_default(self):
        """Test default input processing (no-op)."""
        input_data = MockInputModel(field1='test', field2=42)
        processed = self.handler.process_input(input_data)

        assert processed == input_data

    def test_process_output_default(self):
        """Test default output processing (no-op)."""
        output_data = {'result': 'test_result'}
        input_data = MockInputModel(field1='test', field2=42)
        processed = self.handler.process_output(output_data, input_data)

        assert processed == output_data


class TestBaseHandlerTracing:
    """Test tracing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('axion._handlers.base.handler.init_tracer') as mock_init:
            self.mock_tracer = MockTraceHandler()
            mock_init.return_value = self.mock_tracer
            self.handler = MockHandlerNoValidation()

    def test_span_creation(self):
        """Test span creation."""
        span = self.handler.span('test_operation', attr1='value1')
        assert span is not None

    def test_async_span_creation(self):
        """Test async span creation."""
        async_span = self.handler.async_span('test_operation', attr1='value1')
        assert async_span is not None

    def test_add_trace(self):
        """Test adding trace events."""
        self.handler.add_trace('test_event', 'Test message', {'key': 'value'})
        # Should not raise any exceptions

    def test_get_traces(self):
        """Test getting trace events."""
        traces = self.handler.get_traces()
        assert isinstance(traces, list)

    def test_display_traces(self):
        """Test displaying traces."""
        self.handler.display_traces()
        # Should not raise any exceptions

    def test_get_span_context(self):
        """Test getting span context."""
        trace_id, span_id = self.handler.get_span_context()
        assert trace_id == 'trace-id'
        assert span_id == 'span-id'

    def test_get_tracing_status(self):
        """Test getting tracing status."""
        status = self.handler.get_tracing_status()

        assert isinstance(status, dict)
        assert 'handler_name' in status
        assert 'handler_type' in status
        assert 'metadata_type' in status
        assert 'logfire_enabled' in status
        assert 'current_span_active' in status


class TestBaseHandlerExecution:
    """Test execution-related functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('axion._handlers.base.handler.init_tracer') as mock_init:
            mock_init.return_value = MockTraceHandler()
            self.handler = MockHandler()

    @pytest.mark.asyncio
    async def test_generation_context_success(self):
        """Test successful generation context."""
        callbacks = [AsyncMock()]
        callbacks[0].on_generation_start = AsyncMock()
        callbacks[0].on_generation_end = AsyncMock()

        async with self.handler._generation_context(callbacks):
            await asyncio.sleep(0.01)

        callbacks[0].on_generation_start.assert_called_once()
        callbacks[0].on_generation_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_generation_context_failure(self):
        """Test generation context with exception."""
        callbacks = [AsyncMock()]
        callbacks[0].on_generation_error = AsyncMock()
        callbacks[0].on_generation_end = AsyncMock()

        with pytest.raises(ValueError):
            async with self.handler._generation_context(callbacks):
                raise ValueError('Test error')

        callbacks[0].on_generation_error.assert_called_once()
        callbacks[0].on_generation_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_method(self):
        """Test abstract execute method implementation."""
        input_data = MockInputModel(field1='test', field2=42)
        result = await self.handler.execute(input_data)

        assert isinstance(result, MockOutputModel)
        assert result.result == 'mock_result'

    def test_get_execution_metadata(self):
        """Test getting execution metadata."""
        metadata = self.handler.get_execution_metadata()
        assert isinstance(metadata, dict)
        assert metadata == {'execution_id': 'test-id', 'status': 'completed'}

    @patch('axion._core.display.display_execution_metadata')
    def test_display_execution_metadata_with_display_module(self, mock_display):
        """Test display execution metadata with display module available."""
        self.handler.display_execution_metadata()
        mock_display.assert_called_once()


class TestBaseHandlerContextManager:
    """Test context manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('axion._handlers.base.handler.init_tracer') as mock_init:
            mock_init.return_value = MockTraceHandler()
            self.handler = MockHandlerNoValidation()

    def test_context_manager_success(self):
        """Test context manager with successful execution."""
        with self.handler as h:
            assert h == self.handler

        # Should complete successfully
        assert self.handler.tracer.metadata.status == 'completed'

    def test_context_manager_failure(self):
        """Test context manager with exception."""
        with pytest.raises(ValueError):
            with self.handler as _:
                raise ValueError('Test error')

        # Should mark as failed
        assert self.handler.tracer.metadata.status == 'failed'


class TestBaseHandlerRepr:
    """Test string representation."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('axion._handlers.base.handler.init_tracer') as mock_init:
            mock_init.return_value = MockTraceHandler()
            self.handler = MockHandlerNoValidation(name='test_handler')

    def test_repr(self):
        """Test __repr__ method."""
        repr_output = repr(self.handler)
        expected = 'MockHandlerNoValidation(name=test_handler, metadata_type='
        assert expected in repr_output


class TestBaseHandlerAbstract:
    """Test abstract method enforcement."""

    def test_abstract_execute_method(self):
        """Test that BaseHandler cannot be instantiated without implementing execute."""

        class IncompleteHandler(BaseHandler):
            description = 'Test'
            owner = 'Test'
            handler_type = 'test'
            # Missing execute method implementation

        with pytest.raises(TypeError):
            IncompleteHandler()
