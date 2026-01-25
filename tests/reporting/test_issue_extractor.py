"""
Tests for the IssueExtractor module.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from axion.metrics.signal_normalizer import NormalizedSignal
from axion.reporting.issue_extractor import (
    METRIC_ADAPTERS,
    ExtractedIssue,
    IssueExtractionResult,
    IssueExtractor,
    LLMSummaryInput,
)
from axion.schema import EvaluationResult, MetricScore, TestResult


@dataclass
class MockDatasetItem:
    """Mock DatasetItem for testing."""

    id: str = 'test_item_1'
    query: str = 'What is the capital of France?'
    actual_output: str = 'The capital of France is Paris.'
    expected_output: str = 'Paris'


def _make_signal(
    name: str,
    value: Any,
    score: float = None,
    group: str = 'overall',
    headline_display: bool = False,
    metric_name: str = 'test_metric',
) -> NormalizedSignal:
    """Helper to create NormalizedSignal for tests."""
    return NormalizedSignal(
        metric_name=metric_name,
        group=group,
        name=name,
        value=value,
        score=score,
        headline_display=headline_display,
        raw={'name': name, 'value': value, 'score': score},
    )


class TestExtractedIssue:
    """Tests for ExtractedIssue dataclass."""

    def test_create_basic_issue(self):
        """Test creating a basic ExtractedIssue."""
        issue = ExtractedIssue(
            test_case_id='test_1',
            metric_name='faithfulness',
            signal_group='claim_0',
            signal_name='faithfulness_verdict',
            value='CONTRADICTORY',
            score=0.0,
        )

        assert issue.test_case_id == 'test_1'
        assert issue.metric_name == 'faithfulness'
        assert issue.signal_group == 'claim_0'
        assert issue.signal_name == 'faithfulness_verdict'
        assert issue.value == 'CONTRADICTORY'
        assert issue.score == 0.0
        assert issue.description is None
        assert issue.reasoning is None
        assert issue.item_context == {}
        assert issue.source_path == ''
        assert issue.raw_signal == {}

    def test_create_issue_with_all_fields(self):
        """Test creating an ExtractedIssue with all fields."""
        issue = ExtractedIssue(
            test_case_id='test_1',
            metric_name='faithfulness',
            signal_group='claim_0',
            signal_name='faithfulness_verdict',
            value='CONTRADICTORY',
            score=0.0,
            description='Verdict on claim support',
            reasoning='The claim contradicts the context',
            item_context={'query': 'test query'},
            source_path='results[0].score_results[0].signals.claim_0',
            raw_signal={'name': 'faithfulness_verdict', 'value': 'CONTRADICTORY'},
        )

        assert issue.description == 'Verdict on claim support'
        assert issue.reasoning == 'The claim contradicts the context'
        assert issue.item_context == {'query': 'test query'}
        assert 'results[0]' in issue.source_path


class TestIssueExtractionResult:
    """Tests for IssueExtractionResult dataclass."""

    def test_create_empty_result(self):
        """Test creating an empty IssueExtractionResult."""
        result = IssueExtractionResult(
            run_id='run_123',
            evaluation_name='Test Evaluation',
            total_test_cases=10,
            total_signals_analyzed=50,
            issues_found=0,
            issues_by_metric={},
            issues_by_type={},
            all_issues=[],
        )

        assert result.run_id == 'run_123'
        assert result.evaluation_name == 'Test Evaluation'
        assert result.total_test_cases == 10
        assert result.total_signals_analyzed == 50
        assert result.issues_found == 0
        assert result.all_issues == []


class TestMetricSignalAdapter:
    """Tests for MetricSignalAdapter and predefined adapters."""

    def test_faithfulness_adapter_exists(self):
        """Test that faithfulness adapter is properly defined."""
        adapter = METRIC_ADAPTERS.get('faithfulness')
        assert adapter is not None
        assert adapter.metric_key == 'faithfulness'
        assert 'faithfulness_verdict' in adapter.headline_signals
        assert 'CONTRADICTORY' in adapter.issue_values.get('faithfulness_verdict', [])
        assert 'reasoning' in adapter.context_signals

    def test_answer_criteria_adapter_exists(self):
        """Test that answer_criteria adapter is properly defined."""
        adapter = METRIC_ADAPTERS.get('answer_criteria')
        assert adapter is not None
        assert 'is_covered' in adapter.headline_signals
        assert False in adapter.issue_values.get('is_covered', [])

    def test_answer_relevancy_adapter_exists(self):
        """Test that answer_relevancy adapter is properly defined."""
        adapter = METRIC_ADAPTERS.get('answer_relevancy')
        assert adapter is not None
        assert 'is_relevant' in adapter.headline_signals
        assert 'no' in adapter.issue_values.get('verdict', [])


class TestIssueExtractor:
    """Tests for IssueExtractor class."""

    def test_init_defaults(self):
        """Test IssueExtractor initialization with defaults."""
        extractor = IssueExtractor()

        assert extractor.score_threshold == 0.0
        assert extractor.include_nan is False
        assert extractor.include_context_fields == [
            'query',
            'actual_output',
            'expected_output',
        ]
        assert extractor.metric_filters is None
        assert extractor.max_issues is None
        assert extractor.sample_rate is None

    def test_init_custom_params(self):
        """Test IssueExtractor initialization with custom parameters."""
        extractor = IssueExtractor(
            score_threshold=0.5,
            include_nan=True,
            include_context_fields=['query'],
            metric_filters=['faithfulness'],
            max_issues=100,
            sample_rate=0.5,
        )

        assert extractor.score_threshold == 0.5
        assert extractor.include_nan is True
        assert extractor.include_context_fields == ['query']
        assert extractor.metric_filters == ['faithfulness']
        assert extractor.max_issues == 100
        assert extractor.sample_rate == 0.5

    def test_is_issue_score_at_threshold(self):
        """Test _is_issue_score with score at threshold."""
        extractor = IssueExtractor(score_threshold=0.0)

        assert extractor._is_issue_score(0.0) is True
        assert extractor._is_issue_score(0.5) is False
        assert extractor._is_issue_score(1.0) is False

    def test_is_issue_score_below_threshold(self):
        """Test _is_issue_score with score below threshold."""
        extractor = IssueExtractor(score_threshold=0.5)

        assert extractor._is_issue_score(0.0) is True
        assert extractor._is_issue_score(0.4) is True
        assert extractor._is_issue_score(0.5) is True
        assert extractor._is_issue_score(0.6) is False

    def test_is_issue_score_nan_excluded(self):
        """Test _is_issue_score excludes NaN by default."""
        extractor = IssueExtractor(include_nan=False)

        assert extractor._is_issue_score(float('nan')) is False
        assert extractor._is_issue_score(None) is False

    def test_is_issue_score_nan_included(self):
        """Test _is_issue_score includes NaN when configured."""
        extractor = IssueExtractor(include_nan=True)

        assert extractor._is_issue_score(float('nan')) is True
        assert extractor._is_issue_score(None) is True

    def test_is_issue_score_non_numeric(self):
        """Test _is_issue_score with non-numeric values."""
        extractor = IssueExtractor()

        assert extractor._is_issue_score('not a number') is False
        assert extractor._is_issue_score([1, 2, 3]) is False

    def test_should_sample_no_sampling(self):
        """Test _should_sample with no sampling configured."""
        extractor = IssueExtractor()

        assert extractor._should_sample('any_id') is True

    def test_should_sample_full_sampling(self):
        """Test _should_sample with 100% sampling."""
        extractor = IssueExtractor(sample_rate=1.0)

        assert extractor._should_sample('any_id') is True

    def test_should_sample_zero_sampling(self):
        """Test _should_sample with 0% sampling."""
        extractor = IssueExtractor(sample_rate=0.0)

        assert extractor._should_sample('any_id') is False

    def test_should_sample_deterministic(self):
        """Test _should_sample is deterministic for same ID."""
        extractor = IssueExtractor(sample_rate=0.5)

        # Same ID should always return the same result
        result1 = extractor._should_sample('test_id_123')
        result2 = extractor._should_sample('test_id_123')
        assert result1 == result2

    def test_should_sample_handles_non_string_id(self):
        """Test _should_sample handles non-string IDs."""
        extractor = IssueExtractor(sample_rate=0.5)

        # Should not raise an error
        result = extractor._should_sample(12345)
        assert isinstance(result, bool)

    def test_get_adapter_for_metric(self):
        """Test _get_adapter_for_metric returns correct adapter."""
        extractor = IssueExtractor()

        adapter = extractor._get_adapter_for_metric('Faithfulness')
        assert adapter is not None
        assert adapter.metric_key == 'faithfulness'

        adapter = extractor._get_adapter_for_metric('answer_criteria')
        assert adapter is not None
        assert adapter.metric_key == 'answer_criteria'

        adapter = extractor._get_adapter_for_metric('unknown_metric')
        assert adapter is None

    def test_get_adapter_for_metric_with_hyphens(self):
        """Test _get_adapter_for_metric handles hyphens."""
        extractor = IssueExtractor()

        adapter = extractor._get_adapter_for_metric('answer-criteria')
        assert adapter is not None
        assert adapter.metric_key == 'answer_criteria'

    def test_is_headline_signal_with_flag(self):
        """Test _is_headline_signal with headline_display flag."""
        extractor = IssueExtractor()

        signal = _make_signal('some_signal', 'value', headline_display=True)
        assert extractor._is_headline_signal(signal, None) is True

        signal = _make_signal('some_signal', 'value', headline_display=False)
        assert extractor._is_headline_signal(signal, None) is False

    def test_is_headline_signal_with_adapter(self):
        """Test _is_headline_signal with adapter configuration."""
        extractor = IssueExtractor()
        adapter = METRIC_ADAPTERS['faithfulness']

        signal = _make_signal('faithfulness_verdict', 'CONTRADICTORY')
        assert extractor._is_headline_signal(signal, adapter) is True

        signal = _make_signal('claim_text', 'Some claim')
        assert extractor._is_headline_signal(signal, adapter) is False

    def test_is_issue_value_with_adapter(self):
        """Test _is_issue_value with adapter configuration."""
        extractor = IssueExtractor()
        adapter = METRIC_ADAPTERS['faithfulness']

        assert (
            extractor._is_issue_value('faithfulness_verdict', 'CONTRADICTORY', adapter)
            is True
        )
        assert (
            extractor._is_issue_value('faithfulness_verdict', 'NO_EVIDENCE', adapter)
            is True
        )
        assert (
            extractor._is_issue_value(
                'faithfulness_verdict', 'FULLY_SUPPORTED', adapter
            )
            is False
        )

    def test_is_issue_value_case_insensitive(self):
        """Test _is_issue_value is case insensitive for strings."""
        extractor = IssueExtractor()
        adapter = METRIC_ADAPTERS['faithfulness']

        # Should match regardless of case
        assert (
            extractor._is_issue_value('faithfulness_verdict', 'contradictory', adapter)
            is True
        )
        assert (
            extractor._is_issue_value('faithfulness_verdict', 'Contradictory', adapter)
            is True
        )

    def test_extract_item_context(self):
        """Test _extract_item_context extracts correct fields."""
        extractor = IssueExtractor()
        item = MockDatasetItem()

        context = extractor._extract_item_context(item)

        assert 'query' in context
        assert context['query'] == 'What is the capital of France?'
        assert 'actual_output' in context
        assert 'expected_output' in context

    def test_extract_item_context_truncates_long_values(self):
        """Test _extract_item_context truncates long strings."""
        extractor = IssueExtractor()
        item = MockDatasetItem(actual_output='x' * 600)

        context = extractor._extract_item_context(item)

        assert len(context['actual_output']) == 503  # 500 + '...'
        assert context['actual_output'].endswith('...')

    def test_extract_item_context_none(self):
        """Test _extract_item_context with None test case."""
        extractor = IssueExtractor()

        context = extractor._extract_item_context(None)

        assert context == {}

    def test_find_reasoning_signal(self):
        """Test _find_reasoning_signal finds reasoning."""
        extractor = IssueExtractor()

        group_signals = [
            _make_signal('verdict', 'CONTRADICTORY'),
            _make_signal('reasoning', 'The claim contradicts the context'),
            _make_signal('score', 0.0),
        ]

        reasoning = extractor._find_reasoning_signal(group_signals, None)
        assert reasoning == 'The claim contradicts the context'

    def test_find_reasoning_signal_with_reason(self):
        """Test _find_reasoning_signal finds 'reason' field."""
        extractor = IssueExtractor()

        group_signals = [
            _make_signal('is_covered', False),
            _make_signal('reason', 'Aspect not addressed'),
        ]

        reasoning = extractor._find_reasoning_signal(group_signals, None)
        assert reasoning == 'Aspect not addressed'

    def test_find_reasoning_signal_not_found(self):
        """Test _find_reasoning_signal returns None when not found."""
        extractor = IssueExtractor()

        group_signals = [
            _make_signal('verdict', 'CONTRADICTORY'),
            _make_signal('score', 0.0),
        ]

        reasoning = extractor._find_reasoning_signal(group_signals, None)
        assert reasoning is None


class TestIssueExtractorExtraction:
    """Tests for IssueExtractor extraction methods."""

    def _create_metric_score_with_signals(
        self,
        name: str,
        signals: Dict[str, List[Dict[str, Any]]],
        score: float = 0.5,
    ) -> MetricScore:
        """Helper to create MetricScore with signals."""
        return MetricScore(name=name, score=score, signals=signals)

    def _create_test_result(
        self,
        test_case: Any,
        score_results: List[MetricScore],
    ) -> TestResult:
        """Helper to create TestResult."""
        return TestResult(test_case=test_case, score_results=score_results)

    def _create_evaluation_result(
        self,
        results: List[TestResult],
        run_id: str = 'test_run',
        evaluation_name: str = 'Test Evaluation',
    ) -> EvaluationResult:
        """Helper to create EvaluationResult."""
        return EvaluationResult(
            run_id=run_id,
            evaluation_name=evaluation_name,
            timestamp='2024-01-01T00:00:00',
            results=results,
        )

    def test_extract_from_metric_score_no_signals(self):
        """Test extraction from MetricScore with no signals."""
        extractor = IssueExtractor()
        metric_score = MetricScore(name='test_metric', score=0.5, signals=None)

        issues = extractor.extract_from_metric_score(
            metric_score, 'test_1', MockDatasetItem(), 0, 0
        )

        assert issues == []

    def test_extract_from_metric_score_with_issues(self):
        """Test extraction from MetricScore with issues."""
        extractor = IssueExtractor()

        signals = {
            'claim_0': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'CONTRADICTORY',
                    'score': 0.0,
                    'headline_display': True,
                },
                {
                    'name': 'reasoning',
                    'value': 'Claim contradicts context',
                    'score': float('nan'),
                },
            ]
        }
        metric_score = self._create_metric_score_with_signals('Faithfulness', signals)

        issues = extractor.extract_from_metric_score(
            metric_score, 'test_1', MockDatasetItem(), 0, 0
        )

        assert len(issues) == 1
        assert issues[0].metric_name == 'Faithfulness'
        assert issues[0].signal_name == 'faithfulness_verdict'
        assert issues[0].value == 'CONTRADICTORY'
        assert issues[0].score == 0.0
        assert issues[0].reasoning == 'Claim contradicts context'

    def test_extract_from_metric_score_with_metric_filter(self):
        """Test extraction respects metric filters."""
        extractor = IssueExtractor(metric_filters=['other_metric'])

        signals = {
            'claim_0': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'CONTRADICTORY',
                    'score': 0.0,
                    'headline_display': True,
                },
            ]
        }
        metric_score = self._create_metric_score_with_signals('Faithfulness', signals)

        issues = extractor.extract_from_metric_score(
            metric_score, 'test_1', MockDatasetItem(), 0, 0
        )

        assert issues == []  # Filtered out

    def test_extract_from_metric_score_non_headline_ignored(self):
        """Test non-headline signals are ignored."""
        extractor = IssueExtractor()

        signals = {
            'claim_0': [
                {
                    'name': 'claim_text',
                    'value': 'Some claim text',
                    'score': 0.0,
                    'headline_display': False,
                },
            ]
        }
        metric_score = self._create_metric_score_with_signals('Faithfulness', signals)

        issues = extractor.extract_from_metric_score(
            metric_score, 'test_1', MockDatasetItem(), 0, 0
        )

        assert issues == []

    def test_extract_from_metric_score_unknown_metric_fallback(self):
        """Test extraction from unknown metric falls back to score-based."""
        extractor = IssueExtractor()

        signals = {
            'item_0': [
                {
                    'name': 'some_signal',
                    'value': 'bad_value',
                    'score': 0.0,
                    # No headline_display flag
                },
            ]
        }
        metric_score = self._create_metric_score_with_signals('UnknownMetric', signals)

        issues = extractor.extract_from_metric_score(
            metric_score, 'test_1', MockDatasetItem(), 0, 0
        )

        # Should extract based on score=0.0 when no adapter exists
        assert len(issues) == 1
        assert issues[0].metric_name == 'UnknownMetric'

    def test_extract_from_test_result(self):
        """Test extraction from TestResult."""
        extractor = IssueExtractor()

        signals = {
            'claim_0': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'CONTRADICTORY',
                    'score': 0.0,
                    'headline_display': True,
                },
            ]
        }
        metric_score = self._create_metric_score_with_signals('Faithfulness', signals)
        test_result = self._create_test_result(MockDatasetItem(), [metric_score])

        issues = extractor.extract_from_test_result(test_result, 0)

        assert len(issues) == 1
        assert issues[0].test_case_id == 'test_item_1'

    def test_extract_from_test_result_with_sampling(self):
        """Test extraction respects sampling."""
        # Use 0% sampling to exclude all
        extractor = IssueExtractor(sample_rate=0.0)

        signals = {
            'claim_0': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'CONTRADICTORY',
                    'score': 0.0,
                    'headline_display': True,
                },
            ]
        }
        metric_score = self._create_metric_score_with_signals('Faithfulness', signals)
        test_result = self._create_test_result(MockDatasetItem(), [metric_score])

        issues = extractor.extract_from_test_result(test_result, 0)

        assert issues == []

    def test_extract_from_evaluation(self):
        """Test extraction from full EvaluationResult."""
        extractor = IssueExtractor()

        signals = {
            'claim_0': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'CONTRADICTORY',
                    'score': 0.0,
                    'headline_display': True,
                },
            ]
        }
        metric_score = self._create_metric_score_with_signals('Faithfulness', signals)
        test_result = self._create_test_result(MockDatasetItem(), [metric_score])
        eval_result = self._create_evaluation_result([test_result])

        extraction_result = extractor.extract_from_evaluation(eval_result)

        assert extraction_result.run_id == 'test_run'
        assert extraction_result.evaluation_name == 'Test Evaluation'
        assert extraction_result.total_test_cases == 1
        assert extraction_result.issues_found == 1
        assert len(extraction_result.all_issues) == 1
        assert 'Faithfulness' in extraction_result.issues_by_metric

    def test_extract_from_evaluation_with_max_issues(self):
        """Test extraction respects max_issues limit."""
        extractor = IssueExtractor(max_issues=1)

        # Create multiple issues
        signals1 = {
            'claim_0': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'CONTRADICTORY',
                    'score': 0.0,
                    'headline_display': True,
                },
            ]
        }
        signals2 = {
            'claim_0': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'NO_EVIDENCE',
                    'score': 0.0,
                    'headline_display': True,
                },
            ]
        }
        metric_score1 = self._create_metric_score_with_signals('Faithfulness', signals1)
        metric_score2 = self._create_metric_score_with_signals('Faithfulness', signals2)

        test_result1 = self._create_test_result(
            MockDatasetItem(id='test_1'), [metric_score1]
        )
        test_result2 = self._create_test_result(
            MockDatasetItem(id='test_2'), [metric_score2]
        )
        eval_result = self._create_evaluation_result([test_result1, test_result2])

        extraction_result = extractor.extract_from_evaluation(eval_result)

        assert extraction_result.issues_found == 1

    def test_extract_groups_issues_by_metric(self):
        """Test extraction groups issues by metric correctly."""
        extractor = IssueExtractor()

        signals_faith = {
            'claim_0': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'CONTRADICTORY',
                    'score': 0.0,
                    'headline_display': True,
                },
            ]
        }
        signals_rel = {
            'statement_0': [
                {
                    'name': 'is_relevant',
                    'value': False,
                    'score': 0.0,
                    'headline_display': True,
                },
            ]
        }

        metric_score1 = self._create_metric_score_with_signals(
            'Faithfulness', signals_faith
        )
        metric_score2 = self._create_metric_score_with_signals(
            'Answer Relevancy', signals_rel
        )

        test_result = self._create_test_result(
            MockDatasetItem(), [metric_score1, metric_score2]
        )
        eval_result = self._create_evaluation_result([test_result])

        extraction_result = extractor.extract_from_evaluation(eval_result)

        assert 'Faithfulness' in extraction_result.issues_by_metric
        assert 'Answer Relevancy' in extraction_result.issues_by_metric
        assert len(extraction_result.issues_by_metric['Faithfulness']) == 1
        assert len(extraction_result.issues_by_metric['Answer Relevancy']) == 1


class TestIssueExtractorOutput:
    """Tests for IssueExtractor output methods."""

    def _create_sample_extraction_result(self) -> IssueExtractionResult:
        """Create a sample extraction result for testing."""
        issue1 = ExtractedIssue(
            test_case_id='test_1',
            metric_name='Faithfulness',
            signal_group='claim_0',
            signal_name='faithfulness_verdict',
            value='CONTRADICTORY',
            score=0.0,
            reasoning='The claim contradicts the context',
            item_context={
                'query': 'What is Python?',
                'actual_output': 'Python is a programming language.',
            },
        )
        issue2 = ExtractedIssue(
            test_case_id='test_2',
            metric_name='Answer Relevancy',
            signal_group='statement_0',
            signal_name='is_relevant',
            value=False,
            score=0.0,
            reasoning='Statement is not relevant to the query',
            item_context={'query': 'How to cook pasta?'},
        )

        return IssueExtractionResult(
            run_id='test_run',
            evaluation_name='Test Evaluation',
            total_test_cases=10,
            total_signals_analyzed=50,
            issues_found=2,
            issues_by_metric={
                'Faithfulness': [issue1],
                'Answer Relevancy': [issue2],
            },
            issues_by_type={
                'Faithfulness:faithfulness_verdict': [issue1],
                'Answer Relevancy:is_relevant': [issue2],
            },
            all_issues=[issue1, issue2],
        )

    def test_to_llm_input(self):
        """Test to_llm_input generates correct structured input."""
        extractor = IssueExtractor()
        extraction_result = self._create_sample_extraction_result()

        llm_input = extractor.to_llm_input(extraction_result)

        assert isinstance(llm_input, LLMSummaryInput)
        assert llm_input.evaluation_name == 'Test Evaluation'
        assert llm_input.total_test_cases == 10
        assert llm_input.issues_found == 2
        assert 'Faithfulness' in llm_input.issues_by_metric
        assert len(llm_input.detailed_issues) == 2

    def test_to_llm_input_respects_max_issues(self):
        """Test to_llm_input respects max_issues parameter."""
        extractor = IssueExtractor()
        extraction_result = self._create_sample_extraction_result()

        llm_input = extractor.to_llm_input(extraction_result, max_issues=1)

        assert len(llm_input.detailed_issues) == 1

    def test_to_llm_input_includes_context(self):
        """Test to_llm_input includes issue context."""
        extractor = IssueExtractor()
        extraction_result = self._create_sample_extraction_result()

        llm_input = extractor.to_llm_input(extraction_result)

        # Check first issue has context
        first_issue = llm_input.detailed_issues[0]
        assert 'context' in first_issue
        assert 'query' in first_issue['context']

    def test_to_prompt_text(self):
        """Test to_prompt_text generates valid prompt."""
        extractor = IssueExtractor()
        extraction_result = self._create_sample_extraction_result()

        prompt = extractor.to_prompt_text(extraction_result)

        assert isinstance(prompt, str)
        assert '## Evaluation Issues Summary' in prompt
        assert 'Test Evaluation' in prompt
        assert 'Faithfulness' in prompt
        assert 'CONTRADICTORY' in prompt
        assert '## Task' in prompt
        assert 'Critical Failure Patterns' in prompt

    def test_to_prompt_text_respects_max_issues(self):
        """Test to_prompt_text respects max_issues parameter."""
        extractor = IssueExtractor()
        extraction_result = self._create_sample_extraction_result()

        prompt = extractor.to_prompt_text(extraction_result, max_issues=1)

        # Should mention remaining issues
        assert '... and 1 more issues' in prompt

    def test_to_prompt_text_includes_reasoning(self):
        """Test to_prompt_text includes reasoning when available."""
        extractor = IssueExtractor()
        extraction_result = self._create_sample_extraction_result()

        prompt = extractor.to_prompt_text(extraction_result)

        assert 'The claim contradicts the context' in prompt


class TestIssueExtractorIntegration:
    """Integration tests for IssueExtractor with realistic data."""

    def test_end_to_end_faithfulness_extraction(self):
        """Test end-to-end extraction from faithfulness-like signals."""
        extractor = IssueExtractor()

        # Create realistic faithfulness signals
        signals = {
            'overall': [
                {
                    'name': 'final_score',
                    'value': 0.3,
                    'score': 0.3,
                    'headline_display': True,
                },
                {'name': 'total_claims', 'value': 3, 'score': 3.0},
            ],
            'claim_0: "Python supports version 3.6..."': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'FULLY_SUPPORTED',
                    'score': 1.0,
                    'headline_display': True,
                },
                {'name': 'reasoning', 'value': 'Claim is supported by context'},
            ],
            'claim_1: "The product is free..."': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'CONTRADICTORY',
                    'score': 0.0,
                    'headline_display': True,
                },
                {
                    'name': 'reasoning',
                    'value': 'Context states the product requires a license',
                },
            ],
            'claim_2: "It works on Windows..."': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'NO_EVIDENCE',
                    'score': 0.0,
                    'headline_display': True,
                },
                {'name': 'reasoning', 'value': 'No mention of Windows in context'},
            ],
        }

        metric_score = MetricScore(name='Faithfulness', score=0.3, signals=signals)
        test_result = TestResult(
            test_case=MockDatasetItem(
                id='test_faithfulness_1',
                query='Tell me about the product',
                actual_output='The product is free and works on Windows with Python 3.6.',
                expected_output='The product requires a license.',
            ),
            score_results=[metric_score],
        )
        eval_result = EvaluationResult(
            run_id='faith_run',
            evaluation_name='Faithfulness Test',
            timestamp='2024-01-01T00:00:00',
            results=[test_result],
        )

        extraction_result = extractor.extract_from_evaluation(eval_result)

        # Should find 2 issues (CONTRADICTORY and NO_EVIDENCE)
        assert extraction_result.issues_found == 2
        assert all(
            issue.metric_name == 'Faithfulness'
            for issue in extraction_result.all_issues
        )

        # Check issue values
        issue_values = [issue.value for issue in extraction_result.all_issues]
        assert 'CONTRADICTORY' in issue_values
        assert 'NO_EVIDENCE' in issue_values

        # Check reasoning is captured
        for issue in extraction_result.all_issues:
            assert issue.reasoning is not None

    def test_end_to_end_answer_criteria_extraction(self):
        """Test end-to-end extraction from answer_criteria-like signals."""
        extractor = IssueExtractor()

        signals = {
            'overall': [
                {
                    'name': 'concept_coverage_score',
                    'value': 0.5,
                    'score': 0.5,
                    'headline_display': True,
                },
            ],
            'aspect_Security_Features': [
                {
                    'name': 'is_covered',
                    'value': False,
                    'score': 0.0,
                    'headline_display': True,
                },
                {'name': 'aspect', 'value': 'Security Features'},
                {
                    'name': 'concepts_missing',
                    'value': ['encryption', 'authentication'],
                },
                {
                    'name': 'reason',
                    'value': 'Response does not address security features',
                },
            ],
            'aspect_Performance': [
                {
                    'name': 'is_covered',
                    'value': True,
                    'score': 1.0,
                    'headline_display': True,
                },
                {'name': 'aspect', 'value': 'Performance'},
            ],
        }

        metric_score = MetricScore(name='Answer Criteria', score=0.5, signals=signals)
        test_result = TestResult(
            test_case=MockDatasetItem(id='test_criteria_1'),
            score_results=[metric_score],
        )
        eval_result = EvaluationResult(
            run_id='criteria_run',
            evaluation_name='Criteria Test',
            timestamp='2024-01-01T00:00:00',
            results=[test_result],
        )

        extraction_result = extractor.extract_from_evaluation(eval_result)

        # Should find 1 issue (is_covered=False for Security Features)
        assert extraction_result.issues_found == 1
        assert (
            extraction_result.all_issues[0].signal_group == 'aspect_Security_Features'
        )
        assert extraction_result.all_issues[0].value is False
        assert 'security features' in extraction_result.all_issues[0].reasoning.lower()

    def test_multi_metric_extraction(self):
        """Test extraction from multiple metrics in same test case."""
        extractor = IssueExtractor()

        faith_signals = {
            'claim_0': [
                {
                    'name': 'faithfulness_verdict',
                    'value': 'CONTRADICTORY',
                    'score': 0.0,
                    'headline_display': True,
                },
            ]
        }
        rel_signals = {
            'statement_0': [
                {
                    'name': 'is_relevant',
                    'value': False,
                    'score': 0.0,
                    'headline_display': True,
                },
            ]
        }

        metric_score1 = MetricScore(
            name='Faithfulness', score=0.0, signals=faith_signals
        )
        metric_score2 = MetricScore(
            name='Answer Relevancy', score=0.0, signals=rel_signals
        )

        test_result = TestResult(
            test_case=MockDatasetItem(id='test_multi'),
            score_results=[metric_score1, metric_score2],
        )
        eval_result = EvaluationResult(
            run_id='multi_run',
            evaluation_name='Multi Metric Test',
            timestamp='2024-01-01T00:00:00',
            results=[test_result],
        )

        extraction_result = extractor.extract_from_evaluation(eval_result)

        assert extraction_result.issues_found == 2
        assert len(extraction_result.issues_by_metric) == 2
        assert 'Faithfulness' in extraction_result.issues_by_metric
        assert 'Answer Relevancy' in extraction_result.issues_by_metric

        # Generate prompt and verify both metrics are included
        prompt = extractor.to_prompt_text(extraction_result)
        assert 'Faithfulness' in prompt
        assert 'Answer Relevancy' in prompt


class TestIssueGroup:
    """Tests for IssueGroup dataclass."""

    def test_create_issue_group(self):
        """Test creating an IssueGroup."""
        from axion.reporting.issue_extractor import IssueGroup

        group = IssueGroup(
            metric_name='Faithfulness',
            signal_name='faithfulness_verdict',
            total_count=5,
            unique_values=['CONTRADICTORY', 'NO_EVIDENCE'],
            representative_issues=[],
            affected_test_cases=['test_1', 'test_2'],
        )

        assert group.metric_name == 'Faithfulness'
        assert group.total_count == 5
        assert len(group.unique_values) == 2
        assert group.llm_summary is None


class TestIssueExtractorGrouping:
    """Tests for IssueExtractor grouping functionality."""

    def _create_extraction_result_with_many_issues(self) -> IssueExtractionResult:
        """Create an extraction result with multiple similar issues for grouping."""
        issues = []

        # 5 faithfulness issues (same signal, different values)
        for i in range(5):
            issues.append(
                ExtractedIssue(
                    test_case_id=f'test_faith_{i}',
                    metric_name='Faithfulness',
                    signal_group=f'claim_{i}',
                    signal_name='faithfulness_verdict',
                    value='CONTRADICTORY' if i % 2 == 0 else 'NO_EVIDENCE',
                    score=0.0,
                    reasoning=f'Reasoning for claim {i}',
                    item_context={
                        'query': f'Query {i}',
                        'actual_output': f'Output {i}',
                    },
                )
            )

        # 3 answer relevancy issues
        for i in range(3):
            issues.append(
                ExtractedIssue(
                    test_case_id=f'test_rel_{i}',
                    metric_name='Answer Relevancy',
                    signal_group=f'statement_{i}',
                    signal_name='is_relevant',
                    value=False,
                    score=0.0,
                    reasoning=f'Statement {i} is irrelevant',
                    item_context={'query': f'Query {i}'},
                )
            )

        # 2 answer criteria issues
        for i in range(2):
            issues.append(
                ExtractedIssue(
                    test_case_id=f'test_crit_{i}',
                    metric_name='Answer Criteria',
                    signal_group=f'aspect_{i}',
                    signal_name='is_covered',
                    value=False,
                    score=0.0,
                    reasoning=f'Aspect {i} not covered',
                    item_context={'query': f'Query {i}'},
                )
            )

        return IssueExtractionResult(
            run_id='test_run',
            evaluation_name='Grouping Test',
            total_test_cases=10,
            total_signals_analyzed=100,
            issues_found=len(issues),
            issues_by_metric={
                'Faithfulness': issues[:5],
                'Answer Relevancy': issues[5:8],
                'Answer Criteria': issues[8:],
            },
            issues_by_type={
                'Faithfulness:faithfulness_verdict': issues[:5],
                'Answer Relevancy:is_relevant': issues[5:8],
                'Answer Criteria:is_covered': issues[8:],
            },
            all_issues=issues,
        )

    def test_group_issues(self):
        """Test _group_issues groups similar issues together."""
        extractor = IssueExtractor()
        extraction_result = self._create_extraction_result_with_many_issues()

        groups = extractor._group_issues(extraction_result)

        # Should have 3 groups (one per metric::signal combo)
        assert len(groups) == 3

        # Groups should be sorted by count (most common first)
        assert groups[0].total_count == 5  # Faithfulness
        assert groups[1].total_count == 3  # Answer Relevancy
        assert groups[2].total_count == 2  # Answer Criteria

    def test_group_issues_captures_unique_values(self):
        """Test that grouping captures unique failure values."""
        extractor = IssueExtractor()
        extraction_result = self._create_extraction_result_with_many_issues()

        groups = extractor._group_issues(extraction_result)

        # Faithfulness group should have 2 unique values
        faith_group = next(g for g in groups if g.metric_name == 'Faithfulness')
        assert len(faith_group.unique_values) == 2
        assert 'CONTRADICTORY' in faith_group.unique_values
        assert 'NO_EVIDENCE' in faith_group.unique_values

    def test_group_issues_limits_representatives(self):
        """Test that grouping limits representative examples."""
        extractor = IssueExtractor()
        extraction_result = self._create_extraction_result_with_many_issues()

        groups = extractor._group_issues(extraction_result, max_examples_per_group=2)

        for group in groups:
            assert len(group.representative_issues) <= 2

    def test_group_issues_selects_diverse_representatives(self):
        """Test that representative selection prefers diverse values."""
        extractor = IssueExtractor()
        extraction_result = self._create_extraction_result_with_many_issues()

        groups = extractor._group_issues(extraction_result, max_examples_per_group=2)

        # Faithfulness group should have representatives with different values
        faith_group = next(g for g in groups if g.metric_name == 'Faithfulness')
        rep_values = [str(r.value) for r in faith_group.representative_issues]
        # Should have both CONTRADICTORY and NO_EVIDENCE if possible
        assert len(set(rep_values)) == 2

    def test_to_grouped_prompt_text_without_llm(self):
        """Test to_grouped_prompt_text generates valid grouped prompt."""
        extractor = IssueExtractor()
        extraction_result = self._create_extraction_result_with_many_issues()

        prompt = extractor.to_grouped_prompt_text(extraction_result)

        assert isinstance(prompt, str)
        assert '## Evaluation Issues Summary (Grouped)' in prompt
        assert 'Issue Groups Overview' in prompt
        assert 'Faithfulness' in prompt
        assert 'faithfulness_verdict' in prompt
        assert 'Representative Examples' in prompt
        assert '## Task' in prompt

    def test_to_grouped_prompt_text_respects_max_groups(self):
        """Test to_grouped_prompt_text respects max_groups parameter."""
        extractor = IssueExtractor()
        extraction_result = self._create_extraction_result_with_many_issues()

        prompt = extractor.to_grouped_prompt_text(extraction_result, max_groups=1)

        # Should only include the first group (Faithfulness)
        assert 'Faithfulness' in prompt
        # The other groups should not be in detailed section
        lines = prompt.split('\n')
        group_headers = [line for line in lines if line.startswith('#### Group')]
        assert len(group_headers) == 1

    def test_to_grouped_prompt_text_includes_affected_tests(self):
        """Test to_grouped_prompt_text includes affected test case IDs."""
        extractor = IssueExtractor()
        extraction_result = self._create_extraction_result_with_many_issues()

        prompt = extractor.to_grouped_prompt_text(extraction_result)

        assert 'Affected Tests:' in prompt
        assert 'test_faith_' in prompt

    def test_to_grouped_prompt_is_shorter_than_full_prompt(self):
        """Test that grouped prompt is significantly shorter than full prompt."""
        extractor = IssueExtractor()
        extraction_result = self._create_extraction_result_with_many_issues()

        full_prompt = extractor.to_prompt_text(extraction_result, max_issues=100)
        grouped_prompt = extractor.to_grouped_prompt_text(extraction_result)

        # Grouped should be shorter (fewer repeated contexts)
        assert len(grouped_prompt) < len(full_prompt)


class TestIssueExtractorGroupingWithMockLLM:
    """Tests for grouped summarization with mock LLM."""

    def _create_mock_llm(self, response_text: str = 'Mock summary of the pattern.'):
        """Create a mock LLM for testing."""

        class MockResponse:
            def __init__(self, text):
                self.text = text

        class MockLLM:
            def __init__(self, response):
                self.response = response

            async def acomplete(self, prompt, **kwargs):
                return MockResponse(self.response)

            def complete(self, prompt, **kwargs):
                return MockResponse(self.response)

        return MockLLM(response_text)

    def _create_simple_extraction_result(self) -> IssueExtractionResult:
        """Create a simple extraction result for LLM testing."""
        issues = [
            ExtractedIssue(
                test_case_id='test_1',
                metric_name='Faithfulness',
                signal_group='claim_0',
                signal_name='faithfulness_verdict',
                value='CONTRADICTORY',
                score=0.0,
                reasoning='Claim contradicts context',
                item_context={'query': 'Test query'},
            ),
            ExtractedIssue(
                test_case_id='test_2',
                metric_name='Faithfulness',
                signal_group='claim_1',
                signal_name='faithfulness_verdict',
                value='CONTRADICTORY',
                score=0.0,
                reasoning='Another contradiction',
                item_context={'query': 'Another query'},
            ),
        ]

        return IssueExtractionResult(
            run_id='test_run',
            evaluation_name='LLM Test',
            total_test_cases=2,
            total_signals_analyzed=10,
            issues_found=2,
            issues_by_metric={'Faithfulness': issues},
            issues_by_type={'Faithfulness:faithfulness_verdict': issues},
            all_issues=issues,
        )

    @pytest.mark.asyncio
    async def test_to_grouped_prompt_text_async_with_llm(self):
        """Test async grouped prompt generation with LLM summaries."""
        extractor = IssueExtractor()
        extraction_result = self._create_simple_extraction_result()
        mock_llm = self._create_mock_llm(
            'Claims consistently contradict the source context.'
        )

        prompt = await extractor.to_grouped_prompt_text_async(
            extraction_result, llm=mock_llm
        )

        assert 'Pattern Summary:' in prompt
        assert 'Claims consistently contradict the source context.' in prompt

    @pytest.mark.asyncio
    async def test_generate_group_summary(self):
        """Test _generate_group_summary calls LLM correctly."""
        from axion.reporting.issue_extractor import IssueGroup

        extractor = IssueExtractor()
        mock_llm = self._create_mock_llm('Test summary response')

        group = IssueGroup(
            metric_name='Faithfulness',
            signal_name='faithfulness_verdict',
            total_count=3,
            unique_values=['CONTRADICTORY'],
            representative_issues=[
                ExtractedIssue(
                    test_case_id='test_1',
                    metric_name='Faithfulness',
                    signal_group='claim_0',
                    signal_name='faithfulness_verdict',
                    value='CONTRADICTORY',
                    score=0.0,
                    reasoning='Test reasoning',
                    item_context={'query': 'Test query'},
                )
            ],
            affected_test_cases=['test_1', 'test_2', 'test_3'],
        )

        summary = await extractor._generate_group_summary(group, mock_llm)

        assert summary == 'Test summary response'
