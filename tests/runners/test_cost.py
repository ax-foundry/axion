"""Tests for the cost extraction utilities."""

import pytest
from axion.runners.cost import (
    CostAggregator,
    CostExtractor,
    CostResult,
    TokenUsage,
    extract_cost,
    extract_cost_result,
)


class TestTokenUsage:
    """Test suite for TokenUsage dataclass."""

    def test_init_defaults(self):
        """Test TokenUsage initialization with defaults."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.model == ''

    def test_init_with_values(self):
        """Test TokenUsage initialization with values."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, model='gpt-4o')
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.model == 'gpt-4o'

    def test_total_tokens(self):
        """Test total_tokens property."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_compute_cost_with_custom_pricing(self):
        """Test compute_cost with custom token pricing."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = usage.compute_cost(
            cost_per_input_token=0.001, cost_per_output_token=0.002
        )
        assert cost == 2.0  # 1000 * 0.001 + 500 * 0.002

    def test_compute_cost_with_model(self):
        """Test compute_cost using LLMCostEstimator."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500, model='gpt-4o')
        cost = usage.compute_cost()
        assert cost > 0  # Should get a non-zero cost from LLMCostEstimator

    def test_compute_cost_without_model(self):
        """Test compute_cost without model returns 0."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = usage.compute_cost()
        assert cost == 0.0

    def test_addition(self):
        """Test TokenUsage addition."""
        usage1 = TokenUsage(input_tokens=100, output_tokens=50, model='gpt-4o')
        usage2 = TokenUsage(input_tokens=200, output_tokens=100)
        combined = usage1 + usage2
        assert combined.input_tokens == 300
        assert combined.output_tokens == 150
        assert combined.model == 'gpt-4o'  # Takes first non-empty model


class TestCostResult:
    """Test suite for CostResult dataclass."""

    def test_init_defaults(self):
        """Test CostResult initialization with defaults."""
        result = CostResult()
        assert result.cost_estimate == 0.0
        assert result.token_usage is None
        assert result.source == 'unknown'
        assert result.details == {}

    def test_init_with_values(self):
        """Test CostResult initialization with values."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        result = CostResult(
            cost_estimate=0.05,
            token_usage=usage,
            source='ragas_llm',
            details={'model': 'gpt-4o'},
        )
        assert result.cost_estimate == 0.05
        assert result.token_usage is usage
        assert result.source == 'ragas_llm'
        assert result.details == {'model': 'gpt-4o'}

    def test_addition(self):
        """Test CostResult addition."""
        usage1 = TokenUsage(input_tokens=100, output_tokens=50)
        usage2 = TokenUsage(input_tokens=200, output_tokens=100)
        result1 = CostResult(
            cost_estimate=0.05, token_usage=usage1, source='ragas', details={'a': 1}
        )
        result2 = CostResult(
            cost_estimate=0.03, token_usage=usage2, source='deepeval', details={'b': 2}
        )
        combined = result1 + result2
        assert combined.cost_estimate == 0.08
        assert combined.token_usage.input_tokens == 300
        assert combined.token_usage.output_tokens == 150
        assert combined.source == 'ragas+deepeval'
        assert combined.details == {'a': 1, 'b': 2}


class TestCostExtractor:
    """Test suite for CostExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CostExtractor()

    def test_extract_direct_cost(self):
        """Test extraction from direct cost_estimate attribute."""

        class MetricWithCost:
            cost_estimate = 0.05

        result = self.extractor.extract(MetricWithCost())
        assert result.cost_estimate == 0.05
        assert result.source == 'direct'

    def test_extract_from_llm(self):
        """Test extraction from metric.llm.cost_estimate (Ragas pattern)."""

        class MockLLM:
            cost_estimate = 0.03

        class MetricWithLLM:
            llm = MockLLM()

        result = self.extractor.extract(MetricWithLLM())
        assert result.cost_estimate == 0.03
        assert result.source == 'ragas_llm'

    def test_extract_from_llm_internal(self):
        """Test extraction from metric.llm._cost_estimate."""

        class MockLLM:
            _cost_estimate = 0.04

        class MetricWithLLM:
            llm = MockLLM()

        result = self.extractor.extract(MetricWithLLM())
        assert result.cost_estimate == 0.04
        assert result.source == 'ragas_llm_internal'

    def test_extract_from_model(self):
        """Test extraction from metric.model.cost_estimate (DeepEval pattern)."""

        class MockModel:
            cost_estimate = 0.06

        class MetricWithModel:
            model = MockModel()

        result = self.extractor.extract(MetricWithModel())
        assert result.cost_estimate == 0.06
        assert result.source == 'deepeval_model'

    def test_extract_from_model_internal(self):
        """Test extraction from metric.model._cost_estimate."""

        class MockModel:
            _cost_estimate = 0.07

        class MetricWithModel:
            model = MockModel()

        result = self.extractor.extract(MetricWithModel())
        assert result.cost_estimate == 0.07
        assert result.source == 'deepeval_model_internal'

    def test_extract_from_evaluation_model(self):
        """Test extraction from metric.evaluation_model.cost_estimate."""

        class MockEvalModel:
            cost_estimate = 0.08

        class MetricWithEvalModel:
            evaluation_model = MockEvalModel()

        result = self.extractor.extract(MetricWithEvalModel())
        assert result.cost_estimate == 0.08
        assert result.source == 'evaluation_model'

    def test_extract_fallback_default(self):
        """Test fallback to default value."""

        class MetricNoCost:
            pass

        result = self.extractor.extract(MetricNoCost())
        assert result.cost_estimate == 0.0
        assert result.source == 'default'

    def test_extract_with_custom_default(self):
        """Test extraction with custom default value."""

        class MetricNoCost:
            pass

        result = self.extractor.extract(MetricNoCost(), default=0.01)
        assert result.cost_estimate == 0.01
        assert result.source == 'default'

    def test_extract_none_metric(self):
        """Test extraction from None metric."""
        result = self.extractor.extract(None)
        assert result.cost_estimate == 0.0
        assert result.source == 'none'

    def test_extract_priority_order(self):
        """Test that direct cost takes priority over llm cost."""

        class MockLLM:
            cost_estimate = 0.03

        class MetricWithBoth:
            cost_estimate = 0.05  # Direct should win
            llm = MockLLM()

        result = self.extractor.extract(MetricWithBoth())
        assert result.cost_estimate == 0.05
        assert result.source == 'direct'

    def test_extract_token_usage(self):
        """Test token usage extraction."""

        class MockLLM:
            cost_estimate = 0.05
            input_tokens = 100
            output_tokens = 50
            model = 'gpt-4o'

        class MetricWithTokens:
            llm = MockLLM()

        result = self.extractor.extract(MetricWithTokens())
        assert result.cost_estimate == 0.05
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 50
        assert result.token_usage.model == 'gpt-4o'


class TestCostAggregator:
    """Test suite for CostAggregator class."""

    def test_add_metrics(self):
        """Test adding multiple metrics."""

        class Metric1:
            cost_estimate = 0.05

        class Metric2:
            cost_estimate = 0.03

        aggregator = CostAggregator()
        aggregator.add(Metric1())
        aggregator.add(Metric2())

        assert aggregator.total_cost == 0.08
        assert len(aggregator.results) == 2

    def test_add_result(self):
        """Test adding pre-computed results."""
        aggregator = CostAggregator()
        result = CostResult(cost_estimate=0.05, source='test')
        aggregator.add_result(result)

        assert aggregator.total_cost == 0.05
        assert len(aggregator.results) == 1

    def test_total_tokens(self):
        """Test token aggregation."""

        class Metric:
            class LLM:
                cost_estimate = 0.05
                input_tokens = 100
                output_tokens = 50
                model = 'gpt-4o'

            llm = LLM()

        aggregator = CostAggregator()
        aggregator.add(Metric())
        aggregator.add(Metric())

        total = aggregator.total_tokens
        assert total.input_tokens == 200
        assert total.output_tokens == 100

    def test_compute_total_cost_custom_pricing(self):
        """Test custom pricing computation."""

        class Metric:
            class LLM:
                cost_estimate = 0.05
                input_tokens = 1000
                output_tokens = 500
                model = 'gpt-4o'

            llm = LLM()

        aggregator = CostAggregator()
        aggregator.add(Metric())

        # Custom pricing
        custom_cost = aggregator.compute_total_cost(
            cost_per_input_token=0.001, cost_per_output_token=0.002
        )
        assert custom_cost == 2.0  # 1000 * 0.001 + 500 * 0.002

    def test_summary(self):
        """Test summary generation."""

        class Metric1:
            cost_estimate = 0.05

        class Metric2:
            class Model:
                cost_estimate = 0.03

            model = Model()

        aggregator = CostAggregator()
        aggregator.add(Metric1())
        aggregator.add(Metric2())

        summary = aggregator.summary()
        assert summary['total_cost'] == 0.08
        assert summary['num_metrics'] == 2
        assert 'direct' in summary['costs_by_source']
        assert 'deepeval_model' in summary['costs_by_source']


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_extract_cost_function(self):
        """Test extract_cost convenience function."""

        class Metric:
            cost_estimate = 0.05

        cost = extract_cost(Metric())
        assert cost == 0.05

    def test_extract_cost_with_default(self):
        """Test extract_cost with default value."""

        class MetricNoCost:
            pass

        cost = extract_cost(MetricNoCost(), default=0.01)
        assert cost == 0.01

    def test_extract_cost_result_function(self):
        """Test extract_cost_result convenience function."""

        class Metric:
            cost_estimate = 0.05

        result = extract_cost_result(Metric())
        assert isinstance(result, CostResult)
        assert result.cost_estimate == 0.05
        assert result.source == 'direct'


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_cost(self):
        """Test handling of zero cost."""

        class MetricZeroCost:
            cost_estimate = 0.0

        extractor = CostExtractor()
        # Zero cost should fallback to next strategy
        result = extractor.extract(MetricZeroCost())
        assert result.source == 'default'  # Zero is not positive, so it falls back

    def test_negative_cost(self):
        """Test handling of negative cost."""

        class MetricNegativeCost:
            cost_estimate = -0.05

        extractor = CostExtractor()
        result = extractor.extract(MetricNegativeCost())
        assert result.source == 'default'  # Negative should fallback

    def test_none_cost(self):
        """Test handling of None cost."""

        class MetricNoneCost:
            cost_estimate = None

        extractor = CostExtractor()
        result = extractor.extract(MetricNoneCost())
        assert result.source == 'default'

    def test_string_cost(self):
        """Test handling of string cost value."""

        class MetricStringCost:
            cost_estimate = '0.05'

        extractor = CostExtractor()
        result = extractor.extract(MetricStringCost())
        assert result.source == 'default'  # String is not int/float

    def test_exception_in_extraction(self):
        """Test that exceptions in extraction are handled."""

        class MetricWithProperty:
            @property
            def cost_estimate(self):
                raise RuntimeError('Property access error')

        extractor = CostExtractor()
        result = extractor.extract(MetricWithProperty())
        assert result.source == 'default'  # Should fallback on error
