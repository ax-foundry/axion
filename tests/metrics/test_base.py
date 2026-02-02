from unittest.mock import MagicMock, patch

import pytest

from axion.dataset import DatasetItem
from axion.error import MetricRegistryError, MetricValidationError
from axion.metrics.base import (
    BaseMetric,
    MetricRegistry,
    metric,
    metric_registry,
)
from axion.metrics.schema import MetricConfig, MetricEvaluationResult


class DummyMetric:
    config = MetricConfig(
        key='dummy_metric',
        name='Dummy Metric',
        description='A dummy metric for testing.',
        tags=['test', 'dummy'],
        required_fields=['actual_output', 'expected_output'],
    )

    def __init__(self):
        pass


class IncompatibleMetric:
    config = MetricConfig(
        key='incompatible',
        name='Incompatible Metric',
        description='Missing required fields.',
        tags=['dummy'],
        required_fields=['missing_field'],
    )


@pytest.fixture
def registry():
    # This ensures a clean registry for each test function
    return MetricRegistry()


def setup_function():
    # Clear the global registry before each test that uses it
    metric_registry._registry.clear()


def test_register_and_get_metric(registry):
    registry.register(DummyMetric)
    retrieved = registry.get('dummy_metric')
    assert retrieved == DummyMetric


def test_register_without_config_raises(registry):
    class InvalidMetric:
        pass

    with pytest.raises(MetricRegistryError):
        registry.register(InvalidMetric)


def test_register_overwrite_existing_key(registry):
    """Tests that registering a metric with an existing key overwrites it."""
    registry.register(DummyMetric)

    class NewDummyMetric:
        config = MetricConfig(
            key='dummy_metric', name='New Dummy', description='...', required_fields=[]
        )

    # Using patch to check for the warning log
    with patch('axion.metrics.base.logger.warning') as mock_warning:
        registry.register(NewDummyMetric)
        mock_warning.assert_called_once_with(
            "Metric with key 'dummy_metric' is already registered. Overwriting"
        )

    assert registry.get('dummy_metric') is NewDummyMetric


def test_get_missing_metric_raises(registry):
    with pytest.raises(KeyError):
        registry.get('missing')


def test_find_by_name_description_tag(registry):
    registry.register(DummyMetric)
    results = registry.find('dummy')
    assert DummyMetric in results


def test_get_compatible_metrics(registry):
    registry.register(DummyMetric)
    registry.register(IncompatibleMetric)

    item = MagicMock()
    item.model_dump.return_value = {'actual_output': 'x', 'expected_output': 'y'}

    compatible = registry.get_compatible_metrics(item)
    assert DummyMetric in compatible
    assert IncompatibleMetric not in compatible


def test_iter_and_len(registry):
    registry.register(DummyMetric)
    items = list(iter(registry))
    assert DummyMetric in items
    assert len(registry) == 1


def test_metric_decorator_registers_metric():
    @metric(
        name='Test Metric',
        description='A test metric.',
        required_fields=['input', 'output'],
        tags=['test'],
    )
    class TestMetric(BaseMetric):
        def compute(self, item, **kwargs):
            return 1.0

    # Should be registered in the global registry
    assert 'test_metric' in metric_registry._registry.keys()
    registered = metric_registry.get('test_metric')
    assert registered is TestMetric

    # Validate config
    assert hasattr(TestMetric, 'config')
    assert isinstance(TestMetric.config, MetricConfig)
    assert TestMetric.config.key == 'test_metric'
    assert TestMetric.config.name == 'Test Metric'
    assert TestMetric.config.required_fields == ['input', 'output']
    assert TestMetric.config.tags == ['test']
    assert TestMetric.config.score_range == (0, 1)
    assert TestMetric.config.default_threshold == 0.5


def test_metric_decorator_autogenerates_key():
    """Tests that the key is generated from the name if not provided."""

    @metric(name='My Awesome Metric', description='...', required_fields=[])
    class MyAwesomeMetric(BaseMetric):
        pass

    assert 'my_awesome_metric' in metric_registry._registry
    assert MyAwesomeMetric.config.key == 'my_awesome_metric'


def test_metric_decorator_rejects_invalid_class():
    with pytest.raises(TypeError):

        @metric(
            name='Invalid Metric', description='Should fail.', required_fields=['input']
        )
        class NotAMetric:
            pass


class DummyMetricEx(BaseMetric):
    config = MetricConfig(
        key='dummy_metric',
        name='Dummy Metric',
        description='A dummy metric.',
        required_fields=['expected_output'],
        optional_fields=['actual_output'],
        default_threshold=0.7,
        tags=['test'],
        score_range=(0, 1),
    )

    async def execute(self, item, callbacks=None):
        return MetricEvaluationResult(score=1.0, explanation='All good')


@pytest.fixture
def dataset_item():
    return DatasetItem(query='q', expected_output='expected', actual_output='actual')


def test_metric_initialization_and_config(dataset_item):
    metric = DummyMetricEx()
    assert metric.name == 'Dummy Metric'
    assert metric.threshold == 0.7
    assert metric.config.required_fields == ['expected_output']


def test_metric_name_and_description_fallback_logic():
    """Tests the fallback logic for name and description properties."""

    # 1. Fallback to class name
    class MetricNoConfig(BaseMetric):
        pass

    metric1 = MetricNoConfig()
    assert metric1.name == 'MetricNoConfig'
    assert metric1.description == 'MetricNoConfig'

    # 2. Fallback to config
    class MetricWithConfig(BaseMetric):
        config = MetricConfig(
            name='Config Name', description='Config Desc', key='k', required_fields=[]
        )

    metric2 = MetricWithConfig()
    assert metric2.name == 'Config Name'
    assert metric2.description == 'Config Desc'

    # 3. Instance-level overrides all
    metric3 = MetricWithConfig(
        metric_name='Instance Name', metric_description='Instance Desc'
    )
    assert metric3.name == 'Instance Name'
    assert metric3.description == 'Instance Desc'


def test_metric_threshold_property():
    """Tests that instance threshold overrides config threshold."""
    metric_no_override = DummyMetricEx()
    assert metric_no_override.threshold == 0.7  # From config

    metric_with_override = DummyMetricEx(threshold=0.9)
    assert metric_with_override.threshold == 0.9  # From instance


def test_required_and_optional_fields_properties():
    """Tests the fallback and setter for required/optional fields."""
    metric = DummyMetricEx()
    # Falls back to config
    assert metric.required_fields == ['expected_output']
    assert metric.optional_fields == ['actual_output']

    # Setter overrides
    metric.required_fields = ['new_req']
    metric.optional_fields = ['new_opt']
    assert metric.required_fields == ['new_req']
    assert metric.optional_fields == ['new_opt']


def test_set_instruction_and_examples(dataset_item):
    metric = DummyMetricEx()

    metric.set_instruction('New instruction')
    assert metric.instruction == 'New instruction'

    result = MetricEvaluationResult(score=1.0, explanation='test')
    example = (dataset_item, result)
    metric.set_examples([example])
    assert metric.examples == [example]

    metric.add_examples([example])
    assert metric.examples == [example, example]


def test_repr_method():
    metric = DummyMetricEx()
    repr_str = repr(metric)
    assert 'DummyMetricEx' in repr_str
    assert 'Dummy Metric' in repr_str


def test_validation_failure_raises():
    item = DatasetItem()  # Missing required field 'expected_output'
    metric = DummyMetricEx()
    with pytest.raises(MetricValidationError):
        # _validate is called inside execute, so we call it directly here for a unit test
        metric._validate_required_metric_fields(item)


def test_metric_gets_default_llm():
    """Tests that a metric without explicit LLM gets the default from registry."""

    class SimpleMetric(BaseMetric):
        pass

    metric = SimpleMetric()
    from axion.llm_registry import LiteLLMWrapper

    # Should get a real LLM from registry
    assert isinstance(metric.llm, LiteLLMWrapper)
    assert metric.model_name is not None
    assert metric.llm_provider is not None


def test_embedding_model_skipped_if_not_required():
    """Tests that embed_model is None if requires_embeddings is False."""

    class MyMetric(BaseMetric):
        requires_embeddings = False

    metric = MyMetric()
    assert metric.embed_model is None
    assert metric.embed_model_name is None


def test_get_evaluation_fields_logic(dataset_item):
    """Tests the logic for selecting fields for evaluation."""
    # 1. Tests full
    metric1 = DummyMetricEx()
    fields = metric1.get_evaluation_fields(dataset_item)
    assert 'query' in fields
    assert 'actual_output' in fields
    assert 'expected_output' in fields

    # 2. Tests required and optional
    metric2 = DummyMetricEx(
        required_fields=['query'], optional_fields=['actual_output']
    )
    fields = metric2.get_evaluation_fields(dataset_item)
    assert 'query' in fields
    assert 'actual_output' in fields


def test_compute_cost_estimate():
    """Tests the cost estimation logic with sub-models."""
    metric = DummyMetricEx()
    metric.cost_estimate = 10.0

    sub_metric1 = DummyMetricEx()
    sub_metric1.cost_estimate = 5.0

    sub_metric2 = DummyMetricEx()
    sub_metric2.cost_estimate = 2.5

    metric.compute_cost_estimate([sub_metric1, sub_metric2])
    assert metric.cost_estimate == 17.5  # 10.0 + 5.0 + 2.5


# =====================
# Field Mapping Tests
# =====================


class TestFieldMapping:
    """Tests for the field mapping functionality."""

    def test_field_mapping_initialization(self):
        """Tests that field_mapping is properly initialized."""
        # Without field_mapping
        metric1 = DummyMetricEx()
        assert metric1._field_mapping == {}

        # With field_mapping
        mapping = {'actual_output': 'additional_output.summary'}
        metric2 = DummyMetricEx(field_mapping=mapping)
        assert metric2._field_mapping == mapping

    def test_get_field_without_mapping(self):
        """Tests get_field returns direct attribute when no mapping exists."""
        item = DatasetItem(
            query='test query',
            actual_output='direct output',
            expected_output='expected',
        )
        metric = DummyMetricEx()

        assert metric.get_field(item, 'actual_output') == 'direct output'
        assert metric.get_field(item, 'query') == 'test query'
        assert metric.get_field(item, 'expected_output') == 'expected'

    def test_get_field_with_mapping_from_additional_output(self):
        """Tests get_field resolves mapped paths from additional_output."""
        item = DatasetItem(
            query='test query',
            additional_output={'summary': 'Hello from summary'},
        )
        metric = DummyMetricEx(
            field_mapping={'actual_output': 'additional_output.summary'}
        )

        assert metric.get_field(item, 'actual_output') == 'Hello from summary'
        # Non-mapped fields still work normally
        assert metric.get_field(item, 'query') == 'test query'

    def test_get_field_with_mapping_from_additional_input(self):
        """Tests get_field resolves mapped paths from additional_input."""
        item = DatasetItem(
            query='test query',
            additional_input={'reference': 'expected reference text'},
        )
        metric = DummyMetricEx(
            field_mapping={'expected_output': 'additional_input.reference'}
        )

        assert metric.get_field(item, 'expected_output') == 'expected reference text'

    def test_get_field_with_default_when_path_not_found(self):
        """Tests get_field returns default when mapped path doesn't exist."""
        item = DatasetItem(query='test query')
        metric = DummyMetricEx(
            field_mapping={'actual_output': 'additional_output.nonexistent'}
        )

        assert metric.get_field(item, 'actual_output') is None
        assert metric.get_field(item, 'actual_output', 'fallback') == 'fallback'

    def test_get_field_nested_dict_path(self):
        """Tests get_field resolves deeply nested paths."""
        item = DatasetItem(
            query='test',
            additional_output={'nested': {'deep': {'value': 'found it'}}},
        )
        metric = DummyMetricEx(
            field_mapping={'actual_output': 'additional_output.nested.deep.value'}
        )

        assert metric.get_field(item, 'actual_output') == 'found it'

    def test_resolve_path_direct_attribute(self):
        """Tests _resolve_path for direct attributes."""
        item = DatasetItem(query='test query', actual_output='test output')
        metric = DummyMetricEx()

        assert metric._resolve_path(item, 'query') == 'test query'
        assert metric._resolve_path(item, 'actual_output') == 'test output'

    def test_resolve_path_nested_dict(self):
        """Tests _resolve_path for nested dictionary access."""
        item = DatasetItem(
            query='test',
            additional_output={'key1': {'key2': 'nested value'}},
        )
        metric = DummyMetricEx()

        assert (
            metric._resolve_path(item, 'additional_output.key1.key2') == 'nested value'
        )
        assert metric._resolve_path(item, 'additional_output.key1') == {
            'key2': 'nested value'
        }

    def test_resolve_path_returns_default_on_missing(self):
        """Tests _resolve_path returns default when path is missing."""
        item = DatasetItem(query='test')
        metric = DummyMetricEx()

        assert metric._resolve_path(item, 'nonexistent') is None
        assert metric._resolve_path(item, 'nonexistent', 'default') == 'default'
        assert (
            metric._resolve_path(item, 'additional_output.missing', 'default')
            == 'default'
        )

    def test_get_mapped_fields(self):
        """Tests get_mapped_fields returns all fields with resolved values."""
        item = DatasetItem(
            query='test query',
            expected_output='expected value',
            additional_output={'summary': 'mapped actual'},
        )
        metric = DummyMetricEx(
            required_fields=['actual_output', 'expected_output'],
            optional_fields=['query'],
            field_mapping={'actual_output': 'additional_output.summary'},
        )

        fields = metric.get_mapped_fields(item)

        assert fields['actual_output'] == 'mapped actual'
        assert fields['expected_output'] == 'expected value'
        assert fields['query'] == 'test query'

    def test_validation_with_field_mapping_success(self):
        """Tests validation passes when mapped fields exist."""
        item = DatasetItem(
            query='test',
            additional_output={'summary': 'Hello world'},
            additional_input={'ref': 'Hello'},
        )
        metric = DummyMetricEx(
            required_fields=['actual_output', 'expected_output'],
            field_mapping={
                'actual_output': 'additional_output.summary',
                'expected_output': 'additional_input.ref',
            },
        )

        # Should not raise
        metric._validate_required_metric_fields(item)

    def test_validation_with_field_mapping_failure(self):
        """Tests validation fails when mapped fields are missing."""
        item = DatasetItem(
            query='test',
            additional_output={},  # No 'summary' key
        )
        metric = DummyMetricEx(
            required_fields=['actual_output'],
            field_mapping={'actual_output': 'additional_output.summary'},
        )

        with pytest.raises(MetricValidationError) as exc_info:
            metric._validate_required_metric_fields(item)

        # Error message should include the mapped path info
        assert 'actual_output' in str(exc_info.value)
        assert 'additional_output.summary' in str(exc_info.value)


@pytest.mark.asyncio
async def test_field_mapping_integration():
    """Integration test: metric execution with field mapping."""
    item = DatasetItem(
        query='test',
        additional_output={'summary': 'Hello world'},
        additional_input={'ref': 'Hello'},
    )

    class TestMappingMetric(BaseMetric):
        config = MetricConfig(
            key='test_mapping_metric',
            name='Test Mapping Metric',
            description='Tests field mapping in execute',
            required_fields=['actual_output', 'expected_output'],
            default_threshold=0.5,
        )

        async def execute(self, item, callbacks=None, **kwargs):
            actual = self.get_field(item, 'actual_output')
            expected = self.get_field(item, 'expected_output')

            if actual is None or expected is None:
                return MetricEvaluationResult(score=0.0)

            is_contained = expected.strip() in actual
            return MetricEvaluationResult(
                score=1.0 if is_contained else 0.0,
                explanation=f'actual={actual}, expected={expected}',
            )

    metric = TestMappingMetric(
        field_mapping={
            'actual_output': 'additional_output.summary',
            'expected_output': 'additional_input.ref',
        }
    )

    result = await metric.execute(item)
    assert result.score == 1.0  # 'Hello' is in 'Hello world'
    assert 'Hello world' in result.explanation
    assert 'Hello' in result.explanation
