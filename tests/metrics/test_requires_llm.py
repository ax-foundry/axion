import litellm
import pytest

from axion.dataset import DatasetItem
from axion.llm_registry import MockLLM
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric


@metric(
    name='No LLM Metric',
    key='no_llm_metric',
    description='Scores without reaching a chat model.',
    required_fields=['actual_output'],
    optional_fields=[],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['test'],
)
class NoLLMMetric(BaseMetric):
    requires_llm = False

    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        return MetricEvaluationResult(score=1.0)


@metric(
    name='LLM Metric',
    key='llm_metric',
    description='Reaches a chat model.',
    required_fields=['actual_output'],
    optional_fields=[],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['test'],
)
class LLMMetric(BaseMetric):
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        return MetricEvaluationResult(score=1.0)


@pytest.fixture
def litellm_globals():
    """Restore the LiteLLM module globals the handler writes to."""
    api_base, api_key = litellm.api_base, litellm.api_key
    litellm.api_base, litellm.api_key = 'sentinel-base', 'sentinel-key'
    yield
    litellm.api_base, litellm.api_key = api_base, api_key


@pytest.fixture
def credentialed_settings(monkeypatch):
    """Settings that would configure LiteLLM, so skipping it is observable."""
    from axion._handlers.llm import handler

    monkeypatch.setattr(handler.settings, 'api_base_url', 'https://configured', False)
    monkeypatch.setattr(handler.settings, 'openai_api_key', 'configured-key', False)


class TestRequiresLLM:
    def test_defaults_to_true(self):
        assert BaseMetric.requires_llm is True
        assert LLMMetric.requires_llm is True

    def test_constructs_without_resolving_a_model(self, monkeypatch):
        """No registry lookup, so no credentials and no default model are needed."""
        from axion.metrics import base

        def fail(*args, **kwargs):
            raise AssertionError('the model registry was consulted')

        monkeypatch.setattr(base, 'LLMRegistry', fail)
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)

        instance = NoLLMMetric()

        assert isinstance(instance.llm, MockLLM)
        assert instance.model_name is None
        assert instance.llm_provider is None

    def test_leaves_the_litellm_globals_alone(
        self, litellm_globals, credentialed_settings
    ):
        NoLLMMetric()

        assert litellm.api_base == 'sentinel-base'
        assert litellm.api_key == 'sentinel-key'

    def test_a_metric_that_does_use_an_llm_still_configures_it(
        self, litellm_globals, credentialed_settings
    ):
        """The control: the skip is what `requires_llm = False` buys, not the default."""
        LLMMetric(llm=MockLLM())

        assert litellm.api_base == 'https://configured'
        assert litellm.api_key == 'configured-key'

    @pytest.mark.asyncio
    async def test_executes(self):
        result = await NoLLMMetric().execute(DatasetItem(actual_output='anything'))

        assert result.score == 1.0
