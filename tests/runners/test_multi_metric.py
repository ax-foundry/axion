"""Tests for multi-metric explosion functionality."""

from dataclasses import dataclass
from typing import List

import pytest

from axion._core.tracing import clear_tracing_config, configure_tracing
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.schema import MetricEvaluationResult, SubMetricResult
from axion.runners.metric import AxionRunner, MetricRunner
from axion.schema import EvaluationResult, MetricScore, TestResult

clear_tracing_config()
configure_tracing('noop')


# =====================
# Test Fixtures
# =====================


@dataclass
class MockSignals:
    """Mock signals for multi-metric testing."""

    engagement_score: float = 0.7
    frustration_score: float = 0.2
    sentiment_score: float = 0.85


class MockMultiMetric:
    """Mock multi-metric for testing explosion."""

    __module__ = 'axion.metrics.something'

    def __init__(
        self,
        aggregate_score: float = 0.6,
        is_multi_metric: bool = True,
        include_parent_score: bool = True,
        sub_metric_prefix: bool = True,
    ):
        self.aggregate_score = aggregate_score
        self.is_multi_metric = is_multi_metric
        self.include_parent_score = include_parent_score
        self.sub_metric_prefix = sub_metric_prefix
        self.threshold = 0.5
        self.cost_estimate = 0.01
        self.name = 'SlackConversationAnalyzer'
        self._signals = MockSignals()

    async def execute(self, input_data: DatasetItem, **kwargs):
        return MetricEvaluationResult(
            score=self.aggregate_score,
            explanation='Multi-metric analysis complete',
            signals=self._signals,
            metadata={'source': 'test'},
        )

    def get_sub_metrics(self, result: MetricEvaluationResult) -> List[SubMetricResult]:
        signals = result.signals
        if not signals:
            return []

        return [
            SubMetricResult(
                name='engagement',
                score=signals.engagement_score,
                group='behavioral',
                explanation='Engagement level analysis',
            ),
            SubMetricResult(
                name='frustration',
                score=signals.frustration_score,
                group='sentiment',
                threshold=0.3,  # Custom threshold
            ),
            SubMetricResult(
                name='sentiment',
                score=signals.sentiment_score,
                group='sentiment',
            ),
        ]


class MockSingleMetric:
    """Mock single-metric (non-multi-metric) for testing."""

    __module__ = 'axion.metrics.something'

    def __init__(self, score_value: float = 0.8):
        self.score_value = score_value
        self.is_multi_metric = False
        self.threshold = 0.5
        self.cost_estimate = 0.01
        self.name = 'SimpleFaithfulness'

    async def execute(self, input_data: DatasetItem, **kwargs):
        return MetricEvaluationResult(
            score=self.score_value,
            explanation='Single metric result',
        )


@pytest.fixture
def sample_dataset_item():
    """Create a sample DatasetItem for testing."""
    return DatasetItem(
        id='test_item_1',
        query='What is the weather?',
        actual_output='The weather is sunny.',
    )


@pytest.fixture
def sample_dataset_items():
    """Create multiple DatasetItems for testing."""
    return [
        DatasetItem(id='item_1', query='Query 1', actual_output='Output 1'),
        DatasetItem(id='item_2', query='Query 2', actual_output='Output 2'),
    ]


# =====================
# SubMetricResult Tests
# =====================


class TestSubMetricResult:
    """Tests for SubMetricResult model."""

    def test_submetric_result_creation(self):
        """Test creating a SubMetricResult with all fields."""
        sub = SubMetricResult(
            name='engagement',
            score=0.75,
            explanation='High engagement detected',
            threshold=0.5,
            group='behavioral',
            metadata={'custom': 'data'},
        )
        assert sub.name == 'engagement'
        assert sub.score == 0.75
        assert sub.explanation == 'High engagement detected'
        assert sub.threshold == 0.5
        assert sub.group == 'behavioral'
        assert sub.metadata == {'custom': 'data'}

    def test_submetric_result_minimal(self):
        """Test creating a SubMetricResult with minimal fields."""
        sub = SubMetricResult(name='test')
        assert sub.name == 'test'
        assert sub.score is None
        assert sub.explanation is None
        assert sub.threshold is None
        assert sub.group is None
        assert sub.metadata == {}

    def test_submetric_result_serialization(self):
        """Test SubMetricResult serialization."""
        sub = SubMetricResult(
            name='engagement',
            score=0.75,
            group='behavioral',
        )
        data = sub.model_dump()
        assert data['name'] == 'engagement'
        assert data['score'] == 0.75
        assert data['group'] == 'behavioral'


# =====================
# AxionRunner Multi-Metric Tests
# =====================


class TestAxionRunnerMultiMetric:
    """Tests for AxionRunner multi-metric explosion."""

    @pytest.mark.asyncio
    async def test_multi_metric_explosion_with_prefix(self, sample_dataset_item):
        """Test multi-metric explosion with prefix (default)."""
        metric = MockMultiMetric(
            aggregate_score=0.6,
            include_parent_score=True,
            sub_metric_prefix=True,
        )
        runner = AxionRunner(metric=metric, metric_name='SlackConversationAnalyzer')

        result = await runner.execute(sample_dataset_item)

        # Should return a list of MetricScores
        assert isinstance(result, list)
        assert len(result) == 4  # 1 parent + 3 sub-metrics

        # Check parent score
        parent = result[0]
        assert parent.name == 'SlackConversationAnalyzer'
        assert parent.score == 0.6
        assert parent.type == 'aggregate'
        assert parent.parent is None

        # Check sub-metrics have prefix
        engagement = result[1]
        assert engagement.name == 'SlackConversationAnalyzer_engagement'
        assert engagement.score == 0.7
        assert engagement.type == 'sub_metric'
        assert engagement.parent == 'SlackConversationAnalyzer'
        assert engagement.metadata['group'] == 'behavioral'

        frustration = result[2]
        assert frustration.name == 'SlackConversationAnalyzer_frustration'
        assert frustration.score == 0.2
        assert frustration.threshold == 0.3  # Custom threshold from sub-metric

        sentiment = result[3]
        assert sentiment.name == 'SlackConversationAnalyzer_sentiment'
        assert sentiment.score == 0.85

    @pytest.mark.asyncio
    async def test_multi_metric_explosion_without_prefix(self, sample_dataset_item):
        """Test multi-metric explosion without prefix (standalone metrics)."""
        metric = MockMultiMetric(
            aggregate_score=0.6,
            include_parent_score=False,  # No parent
            sub_metric_prefix=False,  # Standalone metrics
        )
        runner = AxionRunner(metric=metric, metric_name='SlackConversationAnalyzer')

        result = await runner.execute(sample_dataset_item)

        # Should return only sub-metrics (no parent)
        assert isinstance(result, list)
        assert len(result) == 3  # 3 sub-metrics only

        # Check sub-metrics are standalone
        engagement = result[0]
        assert engagement.name == 'engagement'  # No prefix
        assert engagement.type == 'metric'  # Not sub_metric
        assert engagement.parent is None  # No parent reference
        assert engagement.metadata['source_metric'] == 'SlackConversationAnalyzer'

    @pytest.mark.asyncio
    async def test_multi_metric_with_parent_no_sub_prefix(self, sample_dataset_item):
        """Test including parent but not prefixing sub-metrics."""
        metric = MockMultiMetric(
            aggregate_score=0.6,
            include_parent_score=True,
            sub_metric_prefix=False,
        )
        runner = AxionRunner(metric=metric, metric_name='SlackConversationAnalyzer')

        result = await runner.execute(sample_dataset_item)

        assert isinstance(result, list)
        assert len(result) == 4

        # Parent should exist
        parent = result[0]
        assert parent.name == 'SlackConversationAnalyzer'
        assert parent.type == 'aggregate'

        # Sub-metrics should be standalone
        engagement = result[1]
        assert engagement.name == 'engagement'
        assert engagement.type == 'metric'
        assert engagement.parent is None

    @pytest.mark.asyncio
    async def test_single_metric_unchanged(self, sample_dataset_item):
        """Test that single metrics (is_multi_metric=False) return single MetricScore."""
        metric = MockSingleMetric(score_value=0.8)
        runner = AxionRunner(metric=metric, metric_name='SimpleFaithfulness')

        result = await runner.execute(sample_dataset_item)

        # Should return a single MetricScore, not a list
        assert isinstance(result, MetricScore)
        assert result.name == 'SimpleFaithfulness'
        assert result.score == 0.8

    @pytest.mark.asyncio
    async def test_multi_metric_empty_sub_metrics(self, sample_dataset_item):
        """Test multi-metric that returns empty sub-metrics list."""

        class MockEmptyMultiMetric(MockMultiMetric):
            def get_sub_metrics(self, result):
                return []  # Empty list

        metric = MockEmptyMultiMetric()
        runner = AxionRunner(metric=metric, metric_name='EmptyMultiMetric')

        result = await runner.execute(sample_dataset_item)

        # Should fall back to single MetricScore
        assert isinstance(result, MetricScore)
        assert result.name == 'EmptyMultiMetric'

    @pytest.mark.asyncio
    async def test_sub_metrics_get_score_category_when_have_score(
        self, sample_dataset_item
    ):
        """Test that sub-metrics with scores get SCORE category even if parent is ANALYSIS."""

        class MockAnalysisMultiMetric(MockMultiMetric):
            """Mock ANALYSIS metric that produces sub-metrics with scores."""

            def __init__(self):
                super().__init__()
                self.metric_category = MetricCategory.ANALYSIS  # Parent is ANALYSIS

        metric = MockAnalysisMultiMetric()
        runner = AxionRunner(metric=metric, metric_name='AnalysisParent')

        result = await runner.execute(sample_dataset_item)

        assert isinstance(result, list)
        assert len(result) == 4  # 1 parent + 3 sub-metrics

        # Parent should be ANALYSIS
        parent = result[0]
        assert parent.metric_category == 'analysis'

        # Sub-metrics with scores should be SCORE category
        engagement = result[1]
        assert engagement.score == 0.7  # Has a score
        assert engagement.metric_category == 'score'  # Should be SCORE, not ANALYSIS
        assert engagement.threshold is not None  # Should have threshold
        assert engagement.passed is not None  # Should have pass/fail

        frustration = result[2]
        assert frustration.score == 0.2  # Has a score
        assert frustration.metric_category == 'score'

        sentiment = result[3]
        assert sentiment.score == 0.85  # Has a score
        assert sentiment.metric_category == 'score'

    @pytest.mark.asyncio
    async def test_explicit_metric_category_takes_precedence(self, sample_dataset_item):
        """Test that explicitly set metric_category on SubMetricResult takes precedence."""

        class MockMixedCategoryMetric(MockMultiMetric):
            """Mock metric with sub-metrics that have explicit categories."""

            def get_sub_metrics(self, result):
                signals = result.signals
                return [
                    SubMetricResult(
                        name='engagement',
                        score=signals.engagement_score,
                        group='behavioral',
                        # No explicit category - should default to SCORE (has score)
                    ),
                    SubMetricResult(
                        name='recommendation',
                        score=0.8,  # Has a score, but...
                        metric_category=MetricCategory.ANALYSIS,  # Explicitly ANALYSIS
                        group='qualitative',
                    ),
                    SubMetricResult(
                        name='sentiment',
                        score=signals.sentiment_score,
                        metric_category=MetricCategory.SCORE,  # Explicitly SCORE
                        group='sentiment',
                    ),
                ]

        metric = MockMixedCategoryMetric()
        runner = AxionRunner(metric=metric, metric_name='MixedCategories')

        result = await runner.execute(sample_dataset_item)

        assert isinstance(result, list)
        assert len(result) == 4  # 1 parent + 3 sub-metrics

        # engagement: no explicit category, has score -> SCORE
        engagement = result[1]
        assert engagement.name == 'MixedCategories_engagement'
        assert engagement.metric_category == 'score'
        assert engagement.threshold is not None
        assert engagement.passed is not None

        # recommendation: explicit ANALYSIS, even though it has a score
        recommendation = result[2]
        assert recommendation.name == 'MixedCategories_recommendation'
        assert recommendation.score == 0.8  # Has a score
        assert recommendation.metric_category == 'analysis'  # But explicit ANALYSIS
        assert recommendation.threshold is None  # No threshold for ANALYSIS
        assert recommendation.passed is None  # No pass/fail for ANALYSIS

        # sentiment: explicit SCORE
        sentiment = result[3]
        assert sentiment.name == 'MixedCategories_sentiment'
        assert sentiment.metric_category == 'score'
        assert sentiment.threshold is not None


# =====================
# MetricRunner Integration Tests
# =====================


class TestMetricRunnerMultiMetric:
    """Tests for MetricRunner handling multi-metric results."""

    @pytest.mark.asyncio
    async def test_metric_runner_with_multi_metric(self, sample_dataset_items):
        """Test MetricRunner correctly aggregates multi-metric results."""
        multi_metric = MockMultiMetric(
            include_parent_score=True, sub_metric_prefix=True
        )
        single_metric = MockSingleMetric(score_value=0.9)

        runner = MetricRunner(
            metrics=[multi_metric, single_metric],
            max_concurrent=2,
        )

        results = await runner.execute_batch(sample_dataset_items, show_progress=False)

        assert len(results) == 2  # 2 dataset items

        # Each item should have 4 scores from multi-metric + 1 from single metric
        for test_result in results:
            assert len(test_result.score_results) == 5

            # Check multi-metric scores are present
            names = [s.name for s in test_result.score_results]
            assert 'SlackConversationAnalyzer' in names
            assert 'SlackConversationAnalyzer_engagement' in names
            assert 'SlackConversationAnalyzer_frustration' in names
            assert 'SlackConversationAnalyzer_sentiment' in names
            assert 'SimpleFaithfulness' in names

    @pytest.mark.asyncio
    async def test_metric_runner_with_standalone_multi_metric(
        self, sample_dataset_items
    ):
        """Test MetricRunner with standalone multi-metric (no prefix)."""
        multi_metric = MockMultiMetric(
            include_parent_score=False,
            sub_metric_prefix=False,
        )

        runner = MetricRunner(
            metrics=[multi_metric],
            max_concurrent=2,
        )

        results = await runner.execute_batch(sample_dataset_items, show_progress=False)

        assert len(results) == 2

        for test_result in results:
            # Only 3 standalone sub-metrics (no parent)
            assert len(test_result.score_results) == 3

            names = [s.name for s in test_result.score_results]
            assert 'engagement' in names
            assert 'frustration' in names
            assert 'sentiment' in names
            assert 'SlackConversationAnalyzer' not in names  # No parent


# =====================
# EvaluationResult expand_multi_metrics Tests
# =====================


class TestExpandMultiMetrics:
    """Tests for EvaluationResult.expand_multi_metrics()."""

    def test_expand_multi_metrics_basic(self):
        """Test basic expansion of multi-metric results."""
        # Create evaluation result with a metric that has nested data
        score = MetricScore(
            name='ConversationAnalyzer',
            score=0.6,
            metadata={
                'analysis_result': {
                    'engagement': 0.7,
                    'frustration': 0.2,
                }
            },
        )

        test_result = TestResult(
            test_case=DatasetItem(id='item_1', query='test'),
            score_results=[score],
        )

        eval_result = EvaluationResult(
            run_id='run_1',
            evaluation_name='Test',
            timestamp='2024-01-01',
            results=[test_result],
        )

        # Define expansion function
        def expand_analyzer(s: MetricScore) -> List[MetricScore]:
            data = s.metadata.get('analysis_result', {})
            return [
                MetricScore(
                    name='engagement',
                    score=data.get('engagement'),
                    parent=s.name,
                    type='sub_metric',
                    source=s.source,
                ),
                MetricScore(
                    name='frustration',
                    score=data.get('frustration'),
                    parent=s.name,
                    type='sub_metric',
                    source=s.source,
                ),
            ]

        expanded = eval_result.expand_multi_metrics(
            expansion_map={'ConversationAnalyzer': expand_analyzer}
        )

        # Original result should be unchanged
        assert len(eval_result.results[0].score_results) == 1

        # Expanded result should have 3 scores (1 original + 2 expanded)
        assert len(expanded.results[0].score_results) == 3

        names = [s.name for s in expanded.results[0].score_results]
        assert 'ConversationAnalyzer' in names
        assert 'engagement' in names
        assert 'frustration' in names

    def test_expand_multi_metrics_in_place(self):
        """Test in-place expansion."""
        score = MetricScore(
            name='TestMetric',
            score=0.5,
            metadata={'sub_score': 0.8},
        )

        test_result = TestResult(
            test_case=DatasetItem(id='item_1', query='test'),
            score_results=[score],
        )

        eval_result = EvaluationResult(
            run_id='run_1',
            evaluation_name='Test',
            timestamp='2024-01-01',
            results=[test_result],
        )

        def expand_test(s: MetricScore) -> List[MetricScore]:
            return [
                MetricScore(
                    name='sub_test',
                    score=s.metadata.get('sub_score'),
                )
            ]

        result = eval_result.expand_multi_metrics(
            expansion_map={'TestMetric': expand_test},
            in_place=True,
        )

        # Should return same instance
        assert result is eval_result

        # Should have expanded scores
        assert len(eval_result.results[0].score_results) == 2

    def test_expand_multi_metrics_no_expansion_map(self):
        """Test expansion with no expansion map."""
        score = MetricScore(name='TestMetric', score=0.5)
        test_result = TestResult(
            test_case=DatasetItem(id='item_1', query='test'),
            score_results=[score],
        )

        eval_result = EvaluationResult(
            run_id='run_1',
            evaluation_name='Test',
            timestamp='2024-01-01',
            results=[test_result],
        )

        expanded = eval_result.expand_multi_metrics()

        # Should be unchanged
        assert len(expanded.results[0].score_results) == 1
        assert expanded.results[0].score_results[0].name == 'TestMetric'

    def test_expand_multi_metrics_handles_errors(self):
        """Test expansion gracefully handles errors in expansion functions."""
        score = MetricScore(name='TestMetric', score=0.5)
        test_result = TestResult(
            test_case=DatasetItem(id='item_1', query='test'),
            score_results=[score],
        )

        eval_result = EvaluationResult(
            run_id='run_1',
            evaluation_name='Test',
            timestamp='2024-01-01',
            results=[test_result],
        )

        def failing_expand(s: MetricScore) -> List[MetricScore]:
            raise ValueError('Expansion failed!')

        expanded = eval_result.expand_multi_metrics(
            expansion_map={'TestMetric': failing_expand}
        )

        # Should keep original score when expansion fails
        assert len(expanded.results[0].score_results) == 1
        assert expanded.results[0].score_results[0].name == 'TestMetric'
