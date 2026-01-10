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


def test_non_llm_metric_uses_mock_llm():
    """Tests that a metric with default instructions gets a MockLLM."""

    class NonLLMMetric(BaseMetric):
        # Does not override self.instruction
        pass

    metric = NonLLMMetric()
    from axion.llm_registry import MockLLM

    assert isinstance(metric.llm, MockLLM)


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
