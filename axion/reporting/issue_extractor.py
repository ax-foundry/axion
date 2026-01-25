import asyncio
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from axion._core.logging import get_logger
from axion._core.schema import LLMRunnable
from axion.metrics.signal_normalizer import NormalizedSignal, normalize_signals
from axion.schema import EvaluationResult, MetricScore, TestResult

logger = get_logger(__name__)


@dataclass
class ExtractedIssue:
    """
    Represents a single low-score signal extracted from metric evaluation results.

    Attributes:
        test_case_id: Unique identifier for the test case
        metric_name: Name of the metric that produced this signal
        signal_group: Group name for the signal (e.g., "claim_0", "aspect_Coverage")
        signal_name: Name of the signal (e.g., "is_covered", "faithfulness_verdict")
        value: Original value (False, "CONTRADICTORY", etc.)
        score: Numeric score (0.0 for failures)
        description: Optional description of the signal
        reasoning: LLM reasoning from sibling signal if available
        item_context: Context from the test case (query, actual_output, etc.)
        source_path: Path for debugging (e.g., "results[42].score_results[0].signals.claim_0")
        raw_signal: Original signal dict for debugging
    """

    test_case_id: str
    metric_name: str
    signal_group: str
    signal_name: str
    value: Any
    score: float
    description: Optional[str] = None
    reasoning: Optional[str] = None
    item_context: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ''
    raw_signal: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IssueExtractionResult:
    """
    Aggregated result of issue extraction from an evaluation run.

    Attributes:
        run_id: Unique identifier for the evaluation run
        evaluation_name: Optional name of the evaluation
        total_test_cases: Total number of test cases analyzed
        total_signals_analyzed: Total number of signals analyzed
        issues_found: Total number of issues found
        issues_by_metric: Issues grouped by metric name
        issues_by_type: Issues grouped by signal name (issue type)
        all_issues: Flat list of all extracted issues
    """

    run_id: str
    evaluation_name: Optional[str]
    total_test_cases: int
    total_signals_analyzed: int
    issues_found: int
    issues_by_metric: Dict[str, List[ExtractedIssue]]
    issues_by_type: Dict[str, List[ExtractedIssue]]
    all_issues: List[ExtractedIssue]


@dataclass
class LLMSummaryInput:
    """
    Structured input for LLM-based issue summarization.

    Attributes:
        evaluation_name: Name of the evaluation
        total_test_cases: Total test cases analyzed
        issues_found: Total issues found
        issues_by_metric: Summary counts by metric
        issues_by_type: Summary counts by issue type
        detailed_issues: List of detailed issue dicts for the prompt
    """

    evaluation_name: Optional[str]
    total_test_cases: int
    issues_found: int
    issues_by_metric: Dict[str, int]
    issues_by_type: Dict[str, int]
    detailed_issues: List[Dict[str, Any]]


@dataclass
class IssueGroup:
    """
    Represents a group of similar issues for summarization.

    Attributes:
        metric_name: The metric that produced these issues
        signal_name: The signal name (e.g., "is_covered", "faithfulness_verdict")
        total_count: Total number of issues in this group
        unique_values: Set of unique failure values
        representative_issues: Sample issues with full context
        affected_test_cases: List of affected test case IDs
        llm_summary: Optional LLM-generated summary of the pattern
    """

    metric_name: str
    signal_name: str
    total_count: int
    unique_values: List[Any]
    representative_issues: List[ExtractedIssue]
    affected_test_cases: List[str]
    llm_summary: Optional[str] = None


@dataclass
class MetricSignalAdapter:
    """
    Adapter defining how to extract issues from a specific metric's signals.

    Attributes:
        metric_key: Metric identifier (e.g., "faithfulness")
        headline_signals: Signals that indicate pass/fail
        issue_values: Mapping of signal names to failure values
        context_signals: Sibling signals to include for context
    """

    metric_key: str
    headline_signals: List[str]
    issue_values: Dict[str, List[Any]]
    context_signals: List[str]


class SignalAdapterRegistry:
    """
    Registry for MetricSignalAdapter instances.

    Provides a centralized way to register and retrieve adapters for different metrics.
    Users can register custom adapters for their own metrics using the decorator or
    direct registration methods.

    Example using decorator:
        ```python
        @SignalAdapterRegistry.register('my_custom_metric')
        def my_adapter():
            return MetricSignalAdapter(
                metric_key='my_custom_metric',
                headline_signals=['passed'],
                issue_values={'passed': [False]},
                context_signals=['reason'],
            )
        ```

    Example using direct registration:
        ```python
        SignalAdapterRegistry.register_adapter(
            'my_custom_metric',
            MetricSignalAdapter(
                metric_key='my_custom_metric',
                headline_signals=['passed'],
                issue_values={'passed': [False]},
                context_signals=['reason'],
            )
        )
        ```
    """

    _registry: Dict[str, MetricSignalAdapter] = {}

    @classmethod
    def register(cls, metric_key: str):
        """
        Decorator to register a signal adapter for a metric.

        The decorated function should return a MetricSignalAdapter instance.

        Args:
            metric_key: The metric identifier (e.g., 'faithfulness', 'my_custom_metric')

        Returns:
            Decorator function

        Example:
            ```python
            @SignalAdapterRegistry.register('custom_metric')
            def custom_adapter():
                return MetricSignalAdapter(
                    metric_key='custom_metric',
                    headline_signals=['is_valid'],
                    issue_values={'is_valid': [False]},
                    context_signals=['reason'],
                )
            ```
        """
        normalized_key = metric_key.strip().lower().replace(' ', '_').replace('-', '_')

        def decorator(func_or_adapter):
            if isinstance(func_or_adapter, MetricSignalAdapter):
                cls._registry[normalized_key] = func_or_adapter
                logger.debug(f"Signal adapter '{normalized_key}' registered.")
                return func_or_adapter
            elif callable(func_or_adapter):
                adapter = func_or_adapter()
                if not isinstance(adapter, MetricSignalAdapter):
                    raise TypeError(
                        f'Decorated function must return MetricSignalAdapter, '
                        f'got {type(adapter).__name__}'
                    )
                cls._registry[normalized_key] = adapter
                logger.debug(f"Signal adapter '{normalized_key}' registered.")
                return func_or_adapter
            else:
                raise TypeError(
                    f'Expected MetricSignalAdapter or callable, got {type(func_or_adapter).__name__}'
                )

        return decorator

    @classmethod
    def register_adapter(
        cls,
        metric_key: str,
        adapter: MetricSignalAdapter,
    ) -> None:
        """
        Directly register a MetricSignalAdapter for a metric.

        Args:
            metric_key: The metric identifier
            adapter: The MetricSignalAdapter instance

        Example:
            ```python
            SignalAdapterRegistry.register_adapter(
                'my_metric',
                MetricSignalAdapter(
                    metric_key='my_metric',
                    headline_signals=['score'],
                    issue_values={'score': [0]},
                    context_signals=['explanation'],
                )
            )
            ```
        """
        normalized_key = metric_key.strip().lower().replace(' ', '_').replace('-', '_')
        cls._registry[normalized_key] = adapter
        logger.debug(f"Signal adapter '{normalized_key}' registered.")

    @classmethod
    def get(cls, metric_name: str) -> Optional[MetricSignalAdapter]:
        """
        Get the adapter for a metric by name.

        Args:
            metric_name: The metric name (case-insensitive, spaces/hyphens normalized)

        Returns:
            MetricSignalAdapter if found, None otherwise.
        """
        normalized_key = metric_name.strip().lower().replace(' ', '_').replace('-', '_')
        return cls._registry.get(normalized_key)

    @classmethod
    def list_adapters(cls) -> List[str]:
        """
        List all registered adapter keys.

        Returns:
            List of registered metric keys.
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters. Useful for testing."""
        cls._registry.clear()


# Register built-in adapters for common metrics
@SignalAdapterRegistry.register('faithfulness')
def _faithfulness_adapter():
    return MetricSignalAdapter(
        metric_key='faithfulness',
        headline_signals=['faithfulness_verdict'],
        issue_values={'faithfulness_verdict': ['CONTRADICTORY', 'NO_EVIDENCE']},
        context_signals=['claim_text', 'reasoning', 'verdict_score'],
    )


@SignalAdapterRegistry.register('answer_criteria')
def _answer_criteria_adapter():
    return MetricSignalAdapter(
        metric_key='answer_criteria',
        headline_signals=['is_covered', 'concept_coverage'],
        issue_values={'is_covered': [False]},
        context_signals=['aspect', 'concepts_missing', 'concepts_covered', 'reason'],
    )


@SignalAdapterRegistry.register('answer_relevancy')
def _answer_relevancy_adapter():
    return MetricSignalAdapter(
        metric_key='answer_relevancy',
        headline_signals=['is_relevant', 'verdict'],
        issue_values={'is_relevant': [False], 'verdict': ['no']},
        context_signals=['statement', 'reason', 'turn_index'],
    )


@SignalAdapterRegistry.register('answer_completeness')
def _answer_completeness_adapter():
    return MetricSignalAdapter(
        metric_key='answer_completeness',
        headline_signals=['is_covered', 'is_addressed'],
        issue_values={'is_covered': [False], 'is_addressed': [False]},
        context_signals=['aspect', 'sub_question', 'reason', 'concepts_missing'],
    )


@SignalAdapterRegistry.register('contextual_relevancy')
def _contextual_relevancy_adapter():
    return MetricSignalAdapter(
        metric_key='contextual_relevancy',
        headline_signals=['is_relevant'],
        issue_values={'is_relevant': [False]},
        context_signals=['context_text', 'reason', 'relevance_score'],
    )


@SignalAdapterRegistry.register('contextual_recall')
def _contextual_recall_adapter():
    return MetricSignalAdapter(
        metric_key='contextual_recall',
        headline_signals=['is_attributable', 'is_supported'],
        issue_values={'is_attributable': [False], 'is_supported': [False]},
        context_signals=['sentence', 'statement_text', 'reason'],
    )


@SignalAdapterRegistry.register('factual_accuracy')
def _factual_accuracy_adapter():
    return MetricSignalAdapter(
        metric_key='factual_accuracy',
        headline_signals=['is_correct', 'accuracy_score'],
        issue_values={'is_correct': [False, 0]},
        context_signals=['statement', 'reason'],
    )


@SignalAdapterRegistry.register('answer_conciseness')
def _answer_conciseness_adapter():
    return MetricSignalAdapter(
        metric_key='answer_conciseness',
        headline_signals=['conciseness_score'],
        issue_values={},  # Score-based, no specific failure values
        context_signals=['segment', 'category', 'reason', 'redundancy_count'],
    )


@SignalAdapterRegistry.register('contextual_precision')
def _contextual_precision_adapter():
    return MetricSignalAdapter(
        metric_key='contextual_precision',
        headline_signals=['is_useful', 'map_score'],
        issue_values={'is_useful': [False]},
        context_signals=['position', 'chunk_text', 'reason'],
    )


@SignalAdapterRegistry.register('contextual_utilization')
def _contextual_utilization_adapter():
    return MetricSignalAdapter(
        metric_key='contextual_utilization',
        headline_signals=['is_utilized', 'utilization_score'],
        issue_values={'is_utilized': [False]},
        context_signals=['chunk_text', 'reason', 'utilization_rate'],
    )


@SignalAdapterRegistry.register('contextual_sufficiency')
def _contextual_sufficiency_adapter():
    return MetricSignalAdapter(
        metric_key='contextual_sufficiency',
        headline_signals=['is_sufficient', 'sufficiency_score'],
        issue_values={'is_sufficient': [False]},
        context_signals=['reasoning', 'query', 'context'],
    )


@SignalAdapterRegistry.register('contextual_ranking')
def _contextual_ranking_adapter():
    return MetricSignalAdapter(
        metric_key='contextual_ranking',
        headline_signals=['ranking_score', 'is_correctly_ranked'],
        issue_values={'is_correctly_ranked': [False]},
        context_signals=['position', 'expected_position', 'chunk_text', 'reason'],
    )


@SignalAdapterRegistry.register('citation_relevancy')
def _citation_relevancy_adapter():
    return MetricSignalAdapter(
        metric_key='citation_relevancy',
        headline_signals=['relevance_verdict', 'relevance_score'],
        issue_values={'relevance_verdict': [False]},
        context_signals=['citation_text', 'relevance_reason', 'turn_index', 'original_query'],
    )


@SignalAdapterRegistry.register('pii_leakage')
def _pii_leakage_adapter():
    return MetricSignalAdapter(
        metric_key='pii_leakage',
        headline_signals=['pii_verdict', 'final_score'],
        issue_values={'pii_verdict': ['yes']},  # 'yes' means PII was found (bad)
        context_signals=['statement_text', 'reasoning', 'pii_type'],
    )


@SignalAdapterRegistry.register('tone_style_consistency')
def _tone_style_consistency_adapter():
    return MetricSignalAdapter(
        metric_key='tone_style_consistency',
        headline_signals=['consistency_score', 'is_consistent'],
        issue_values={'is_consistent': [False]},
        context_signals=['tone_detected', 'expected_tone', 'reason', 'deviation_type'],
    )


@SignalAdapterRegistry.register('persona_tone_adherence')
def _persona_tone_adherence_adapter():
    return MetricSignalAdapter(
        metric_key='persona_tone_adherence',
        headline_signals=['persona_match', 'adherence_score', 'final_composite_score'],
        issue_values={'persona_match': [False]},
        context_signals=[
            'tone_classification',
            'deviation_type',
            'positive_indicators',
            'negative_indicators',
            'violations',
        ],
    )


@SignalAdapterRegistry.register('persona_tone')
def _persona_tone_adapter():
    """Alias for persona_tone_adherence."""
    return MetricSignalAdapter(
        metric_key='persona_tone',
        headline_signals=['persona_match', 'adherence_score', 'final_composite_score'],
        issue_values={'persona_match': [False]},
        context_signals=[
            'tone_classification',
            'deviation_type',
            'positive_indicators',
            'negative_indicators',
            'violations',
        ],
    )


@SignalAdapterRegistry.register('conversation_efficiency')
def _conversation_efficiency_adapter():
    return MetricSignalAdapter(
        metric_key='conversation_efficiency',
        headline_signals=['efficiency_score', 'final_composite_score'],
        issue_values={},  # Score-based with inefficiency detection
        context_signals=[
            'type',
            'severity',
            'turns_wasted',
            'description',
            'total_wasted_turns',
            'redundancy_score',
        ],
    )


@SignalAdapterRegistry.register('conversation_flow')
def _conversation_flow_adapter():
    return MetricSignalAdapter(
        metric_key='conversation_flow',
        headline_signals=['final_score', 'coherence_score'],
        issue_values={},  # Score-based with issue detection
        context_signals=[
            'type',
            'severity',
            'penalty_contribution',
            'description',
            'turn_location',
            'coherence',
            'efficiency',
            'user_experience',
        ],
    )


@SignalAdapterRegistry.register('goal_completion')
def _goal_completion_adapter():
    return MetricSignalAdapter(
        metric_key='goal_completion',
        headline_signals=['completion_score', 'is_completed', 'goal_achieved'],
        issue_values={'is_completed': [False], 'goal_achieved': [False]},
        context_signals=['goal', 'progress', 'reason', 'blocking_issues'],
    )


@SignalAdapterRegistry.register('citation_presence')
def _citation_presence_adapter():
    return MetricSignalAdapter(
        metric_key='citation_presence',
        headline_signals=['presence_check_passed'],
        issue_values={'presence_check_passed': [False]},
        context_signals=[
            'total_assistant_messages',
            'messages_with_valid_citations',
            'passing_turn_indices',
            'mode',
        ],
    )


@SignalAdapterRegistry.register('latency')
def _latency_adapter():
    return MetricSignalAdapter(
        metric_key='latency',
        headline_signals=['latency_score', 'latency_ms'],
        issue_values={},  # Score-based, threshold determines failure
        context_signals=['performance_classification', 'threshold', 'normalization_method'],
    )


@SignalAdapterRegistry.register('tool_correctness')
def _tool_correctness_adapter():
    return MetricSignalAdapter(
        metric_key='tool_correctness',
        headline_signals=['correctness_score', 'all_tools_correct'],
        issue_values={'all_tools_correct': [False]},
        context_signals=[
            'tool_name',
            'expected_tools',
            'actual_tools',
            'missing_tools',
            'unexpected_tools',
            'parameter_mismatches',
            'reason',
        ],
    )


# Backward compatibility alias
METRIC_ADAPTERS = SignalAdapterRegistry._registry


class IssueExtractor:
    """
    Extracts low-score signals from evaluation results for LLM-based issue summarization.

    This class reads existing signal data from MetricScore objects and extracts
    issues (low-score signals) in a normalized format suitable for analysis.
    """

    def __init__(
        self,
        score_threshold: float = 0.0,
        include_nan: bool = False,
        include_context_fields: Optional[List[str]] = None,
        metric_filters: Optional[List[str]] = None,
        max_issues: Optional[int] = None,
        sample_rate: Optional[float] = None,
    ):
        """
        Initialize the IssueExtractor.

        Args:
            score_threshold: Signals with scores at or below this threshold are
                considered issues. Default 0.0 means only explicit failures.
            include_nan: Whether to include signals with NaN scores as issues.
            include_context_fields: Fields to include from test case context.
                Defaults to ['query', 'actual_output', 'expected_output'].
            metric_filters: Optional list of metric names to filter. If None,
                all metrics are processed.
            max_issues: Hard limit on number of issues to return.
            sample_rate: Deterministic sampling rate (0.0-1.0) by test_case_id.
        """
        self.score_threshold = score_threshold
        self.include_nan = include_nan
        self.include_context_fields = include_context_fields or [
            'query',
            'actual_output',
            'expected_output',
        ]
        self.metric_filters = metric_filters
        self.max_issues = max_issues
        self.sample_rate = sample_rate

    def _should_sample(self, test_case_id: str) -> bool:
        """
        Deterministic sampling by test_case_id hash for stable results.

        Args:
            test_case_id: The test case identifier

        Returns:
            True if this test case should be included based on sampling rate.
        """
        if self.sample_rate is None or self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False

        test_case_id = str(test_case_id)
        hash_value = int(hashlib.md5(test_case_id.encode()).hexdigest(), 16)
        return (hash_value % 1000) < (self.sample_rate * 1000)

    def _is_issue_score(self, score: Union[float, int, None]) -> bool:
        """
        Check if a score qualifies as an issue.

        Args:
            score: The numeric score to check

        Returns:
            True if the score is considered an issue based on threshold and nan settings.
        """
        if score is None:
            return self.include_nan

        if isinstance(score, float) and math.isnan(score):
            return self.include_nan

        if not isinstance(score, (int, float)):
            return False

        return float(score) <= self.score_threshold

    def _get_adapter_for_metric(
        self, metric_name: str
    ) -> Optional[MetricSignalAdapter]:
        """
        Get the signal adapter for a metric, if one exists.

        Args:
            metric_name: The name of the metric

        Returns:
            MetricSignalAdapter if found, None otherwise.
        """
        return SignalAdapterRegistry.get(metric_name)

    def _is_headline_signal(
        self,
        signal: NormalizedSignal,
        adapter: Optional[MetricSignalAdapter],
    ) -> bool:
        """
        Determine if a signal is a headline (pass/fail indicator) signal.

        Args:
            signal: The normalized signal
            adapter: Optional metric adapter

        Returns:
            True if this is a headline signal.
        """
        # Check headline_display flag first
        if signal.headline_display:
            return True

        # Check adapter's headline_signals list
        if adapter and signal.name in adapter.headline_signals:
            return True

        return False

    def _is_issue_value(
        self, signal_name: str, value: Any, adapter: Optional[MetricSignalAdapter]
    ) -> bool:
        """
        Check if a signal value indicates an issue based on adapter configuration.

        Args:
            signal_name: Name of the signal
            value: The signal value
            adapter: Optional metric adapter

        Returns:
            True if this value indicates an issue.
        """
        if adapter and signal_name in adapter.issue_values:
            candidate_values = adapter.issue_values[signal_name]
            if isinstance(value, str):
                normalized = value.lower()
                return any(
                    isinstance(candidate, str) and candidate.lower() == normalized
                    for candidate in candidate_values
                )
            return any(candidate == value for candidate in candidate_values)
        return False

    def _extract_item_context(self, test_case: Any) -> Dict[str, Any]:
        """
        Extract context fields from a test case item.

        Args:
            test_case: The DatasetItem or similar object

        Returns:
            Dict with context fields
        """
        context = {}
        if test_case is None:
            return context

        for field_name in self.include_context_fields:
            value = getattr(test_case, field_name, None)
            if value is not None:
                # Truncate long values for readability
                if isinstance(value, str) and len(value) > 500:
                    context[field_name] = value[:500] + '...'
                elif isinstance(value, list) and len(value) > 5:
                    context[field_name] = value[:5] + ['...']
                else:
                    context[field_name] = value

        return context

    def _find_reasoning_signal(
        self,
        group_signals: List[NormalizedSignal],
        adapter: Optional[MetricSignalAdapter],
    ) -> Optional[str]:
        """
        Find reasoning/explanation from sibling signals in the same group.

        Args:
            group_signals: List of signals in the same group
            adapter: Optional metric adapter

        Returns:
            Reasoning string if found, None otherwise.
        """
        # Look for common reasoning signal names
        reasoning_names = ['reasoning', 'reason', 'explanation']
        if adapter:
            reasoning_names.extend(
                [s for s in adapter.context_signals if 'reason' in s.lower()]
            )

        for signal in group_signals:
            if signal.name in reasoning_names:
                value = signal.value
                if value and isinstance(value, str):
                    return value

        return None

    def extract_from_metric_score(
        self,
        metric_score: MetricScore,
        test_case_id: str,
        test_case: Any,
        result_index: int,
        score_index: int,
    ) -> List[ExtractedIssue]:
        """
        Extract issues from a single MetricScore.

        Args:
            metric_score: The MetricScore to analyze
            test_case_id: ID of the test case
            test_case: The test case object for context
            result_index: Index in the results list
            score_index: Index in the score_results list

        Returns:
            List of ExtractedIssue objects found in this MetricScore.
        """
        issues: List[ExtractedIssue] = []

        # Skip if no signals
        if metric_score.signals is None:
            return issues

        metric_name = metric_score.name or 'unknown'

        # Check metric filter
        if self.metric_filters and metric_name not in self.metric_filters:
            return issues

        normalized_signals = normalize_signals(metric_name, metric_score.signals)
        if not normalized_signals:
            return issues

        adapter = self._get_adapter_for_metric(metric_name)
        item_context = self._extract_item_context(test_case)

        grouped_signals: Dict[str, List[NormalizedSignal]] = {}
        for signal in normalized_signals:
            grouped_signals.setdefault(signal.group, []).append(signal)

        for group_name, group_signals in grouped_signals.items():
            # Find reasoning for this group
            reasoning = self._find_reasoning_signal(group_signals, adapter)

            for signal in group_signals:
                score = signal.score

                # Check if this signal is an issue
                is_score_issue = self._is_issue_score(score)
                is_value_issue = self._is_issue_value(
                    signal.name, signal.value, adapter
                )
                is_headline = self._is_headline_signal(signal, adapter)

                # If no adapter exists, fall back to score-based extraction
                if adapter is None and is_score_issue:
                    is_headline = True

                # Only extract headlines that are issues
                if is_headline and (is_score_issue or is_value_issue):
                    source_path = (
                        f'results[{result_index}].score_results[{score_index}]'
                        f'.signals.{group_name}.{signal.name}'
                    )

                    issue = ExtractedIssue(
                        test_case_id=test_case_id,
                        metric_name=metric_name,
                        signal_group=group_name,
                        signal_name=signal.name,
                        value=signal.value,
                        score=score if score is not None else float('nan'),
                        description=signal.description,
                        reasoning=reasoning,
                        item_context=item_context,
                        source_path=source_path,
                        raw_signal=signal.raw,
                    )
                    issues.append(issue)

        return issues

    def extract_from_test_result(
        self, test_result: TestResult, result_index: int
    ) -> List[ExtractedIssue]:
        """
        Extract issues from a single TestResult.

        Args:
            test_result: The TestResult to analyze
            result_index: Index in the results list

        Returns:
            List of ExtractedIssue objects found in this TestResult.
        """
        issues: List[ExtractedIssue] = []

        test_case = test_result.test_case
        test_case_id = (
            str(getattr(test_case, 'id', f'test_{result_index}'))
            if test_case
            else f'test_{result_index}'
        )

        # Apply sampling
        if not self._should_sample(test_case_id):
            return issues

        for score_index, metric_score in enumerate(test_result.score_results):
            issues.extend(
                self.extract_from_metric_score(
                    metric_score,
                    test_case_id,
                    test_case,
                    result_index,
                    score_index,
                )
            )

        return issues

    def extract_from_evaluation(
        self, result: EvaluationResult
    ) -> IssueExtractionResult:
        """
        Extract all issues from an EvaluationResult.

        Args:
            result: The EvaluationResult to analyze

        Returns:
            IssueExtractionResult with all extracted issues.
        """
        all_issues: List[ExtractedIssue] = []
        total_signals = 0

        for result_index, test_result in enumerate(result.results):
            # Count signals for stats
            for metric_score in test_result.score_results:
                total_signals += len(
                    normalize_signals(
                        metric_score.name or 'unknown', metric_score.signals
                    )
                )

            # Extract issues
            issues = self.extract_from_test_result(test_result, result_index)
            all_issues.extend(issues)

            # Check max_issues limit
            if self.max_issues and len(all_issues) >= self.max_issues:
                all_issues = all_issues[: self.max_issues]
                break

        # Group issues by metric
        issues_by_metric: Dict[str, List[ExtractedIssue]] = {}
        for issue in all_issues:
            if issue.metric_name not in issues_by_metric:
                issues_by_metric[issue.metric_name] = []
            issues_by_metric[issue.metric_name].append(issue)

        # Group issues by type (signal name)
        issues_by_type: Dict[str, List[ExtractedIssue]] = {}
        for issue in all_issues:
            type_key = f'{issue.metric_name}:{issue.signal_name}'
            if type_key not in issues_by_type:
                issues_by_type[type_key] = []
            issues_by_type[type_key].append(issue)

        return IssueExtractionResult(
            run_id=result.run_id,
            evaluation_name=result.evaluation_name,
            total_test_cases=len(result.results),
            total_signals_analyzed=total_signals,
            issues_found=len(all_issues),
            issues_by_metric=issues_by_metric,
            issues_by_type=issues_by_type,
            all_issues=all_issues,
        )

    def to_llm_input(
        self, result: IssueExtractionResult, max_issues: int = 50
    ) -> LLMSummaryInput:
        """
        Convert extraction result to structured LLM input.

        Args:
            result: The IssueExtractionResult to convert
            max_issues: Maximum number of detailed issues to include

        Returns:
            LLMSummaryInput suitable for LLM processing.
        """
        # Build summary counts
        issues_by_metric_counts = {
            metric: len(issues) for metric, issues in result.issues_by_metric.items()
        }
        issues_by_type_counts = {
            issue_type: len(issues)
            for issue_type, issues in result.issues_by_type.items()
        }

        # Build detailed issues list
        detailed_issues = []
        for issue in result.all_issues[:max_issues]:
            detail = {
                'test_case_id': issue.test_case_id,
                'metric': issue.metric_name,
                'signal_group': issue.signal_group,
                'signal_name': issue.signal_name,
                'value': issue.value,
                'score': issue.score,
            }
            if issue.description:
                detail['description'] = issue.description
            if issue.reasoning:
                detail['reasoning'] = issue.reasoning
            if issue.item_context:
                detail['context'] = issue.item_context

            detailed_issues.append(detail)

        return LLMSummaryInput(
            evaluation_name=result.evaluation_name,
            total_test_cases=result.total_test_cases,
            issues_found=result.issues_found,
            issues_by_metric=issues_by_metric_counts,
            issues_by_type=issues_by_type_counts,
            detailed_issues=detailed_issues,
        )

    def to_prompt_text(
        self, result: IssueExtractionResult, max_issues: int = 50
    ) -> str:
        """
        Generate a text prompt for LLM-based issue summarization.

        Args:
            result: The IssueExtractionResult to convert
            max_issues: Maximum number of detailed issues to include

        Returns:
            Formatted prompt text.
        """
        lines = []

        # Header
        lines.append('## Evaluation Issues Summary')
        lines.append('')
        if result.evaluation_name:
            lines.append(f'**Evaluation:** {result.evaluation_name}')
        lines.append(f'**Test Cases Analyzed:** {result.total_test_cases}')
        lines.append(f'**Issues Found:** {result.issues_found}')
        lines.append('')

        # Issue breakdown by metric
        lines.append('### Issue Breakdown by Metric')
        for metric, issues in sorted(result.issues_by_metric.items()):
            # Count issue types for this metric
            type_counts: Dict[str, int] = {}
            for issue in issues:
                value_str = str(issue.value)
                if value_str not in type_counts:
                    type_counts[value_str] = 0
                type_counts[value_str] += 1

            type_summary = ', '.join(
                f'{count} {val}' for val, count in type_counts.items()
            )
            lines.append(f'- {metric}: {len(issues)} issues ({type_summary})')
        lines.append('')

        # Detailed issues
        lines.append('### Detailed Issues')
        lines.append('')

        for i, issue in enumerate(result.all_issues[:max_issues], 1):
            lines.append(f'#### Issue {i}: {issue.metric_name} - {issue.signal_name}')
            lines.append(f'- **Test Case:** {issue.test_case_id}')
            lines.append(f'- **Signal Group:** {issue.signal_group}')
            lines.append(f'- **Value:** {issue.value}')
            lines.append(f'- **Score:** {issue.score}')

            if issue.reasoning:
                lines.append(f'- **Reasoning:** "{issue.reasoning}"')

            if issue.item_context:
                for key, value in issue.item_context.items():
                    if isinstance(value, str):
                        # Truncate for readability
                        display_value = (
                            value[:300] + '...' if len(value) > 300 else value
                        )
                        lines.append(
                            f'- **{key.replace("_", " ").title()}:** "{display_value}"'
                        )

            lines.append('')

        if result.issues_found > max_issues:
            lines.append(f'*... and {result.issues_found - max_issues} more issues*')
            lines.append('')

        # Task section
        lines.append('## Task')
        lines.append('Analyze the quality issues found in this evaluation. Provide:')
        lines.append(
            '1. **Critical Failure Patterns:** What are the most common/severe issue types?'
        )
        lines.append(
            '2. **Root Cause Analysis:** What systemic problems might be causing these failures?'
        )
        lines.append(
            '3. **Recommended Improvements:** Specific actions to improve quality'
        )
        lines.append('4. **Priority Ranking:** Which issues should be addressed first?')

        return '\n'.join(lines)

    def _group_issues(
        self, result: IssueExtractionResult, max_examples_per_group: int = 2
    ) -> List[IssueGroup]:
        """
        Group similar issues together for summarization.

        Args:
            result: The IssueExtractionResult to group
            max_examples_per_group: Maximum representative examples per group

        Returns:
            List of IssueGroup objects.
        """
        groups: Dict[str, List[ExtractedIssue]] = {}

        # Group by metric_name + signal_name
        for issue in result.all_issues:
            key = f'{issue.metric_name}::{issue.signal_name}'
            if key not in groups:
                groups[key] = []
            groups[key].append(issue)

        issue_groups = []
        for key, issues in groups.items():
            metric_name, signal_name = key.split('::', 1)

            # Get unique values
            unique_values = list(set(str(i.value) for i in issues))

            # Get affected test cases
            affected_test_cases = list(set(i.test_case_id for i in issues))

            # Select representative examples (diverse values if possible)
            representatives = []
            seen_values = set()
            for issue in issues:
                val_str = str(issue.value)
                if (
                    val_str not in seen_values
                    and len(representatives) < max_examples_per_group
                ):
                    representatives.append(issue)
                    seen_values.add(val_str)
            # Fill remaining slots if we haven't hit max
            for issue in issues:
                if len(representatives) >= max_examples_per_group:
                    break
                if issue not in representatives:
                    representatives.append(issue)

            issue_groups.append(
                IssueGroup(
                    metric_name=metric_name,
                    signal_name=signal_name,
                    total_count=len(issues),
                    unique_values=unique_values,
                    representative_issues=representatives[:max_examples_per_group],
                    affected_test_cases=affected_test_cases[
                        :10
                    ],  # Limit for readability
                )
            )

        # Sort by count (most common issues first)
        issue_groups.sort(key=lambda g: g.total_count, reverse=True)
        return issue_groups

    async def _generate_group_summary(self, group: IssueGroup, llm: LLMRunnable) -> str:
        """
        Generate an LLM summary for a group of similar issues.

        Args:
            group: The IssueGroup to summarize
            llm: The LLM to use for summarization

        Returns:
            Summary string.
        """
        # Build a concise prompt for this group
        prompt_lines = [
            'Analyze this group of evaluation failures and provide a 1-2 sentence summary of the pattern.',
            '',
            f'Metric: {group.metric_name}',
            f'Signal: {group.signal_name}',
            f'Total occurrences: {group.total_count}',
            f'Failure values: {", ".join(group.unique_values)}',
            '',
            'Representative examples:',
        ]

        for i, issue in enumerate(group.representative_issues, 1):
            prompt_lines.append(f'\nExample {i}:')
            prompt_lines.append(f'  Value: {issue.value}')
            if issue.reasoning:
                prompt_lines.append(f'  Reasoning: {issue.reasoning}')
            if issue.item_context.get('query'):
                query = issue.item_context['query']
                if len(query) > 150:
                    query = query[:150] + '...'
                prompt_lines.append(f'  Query: {query}')

        prompt_lines.append('')
        prompt_lines.append(
            'Respond with ONLY a 1-2 sentence summary of the failure pattern. No preamble.'
        )

        prompt = '\n'.join(prompt_lines)

        try:
            response = await llm.acomplete(prompt)
            # Handle different response formats
            if hasattr(response, 'text'):
                return response.text.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
        except Exception as e:
            logger.warning(f'Failed to generate LLM summary for group: {e}')
            return ''

    async def to_grouped_prompt_text_async(
        self,
        result: IssueExtractionResult,
        llm: Optional[LLMRunnable] = None,
        max_groups: int = 20,
        max_examples_per_group: int = 2,
    ) -> str:
        """
        Generate a grouped prompt with optional LLM summarization (async version).

        Groups similar issues together and shows representative examples,
        reducing context size while preserving signal quality.

        Args:
            result: The IssueExtractionResult to convert
            llm: Optional LLM for generating group summaries
            max_groups: Maximum number of issue groups to include
            max_examples_per_group: Representative examples per group

        Returns:
            Formatted prompt text with grouped issues.
        """
        groups = self._group_issues(result, max_examples_per_group)[:max_groups]

        # Generate LLM summaries if LLM is provided
        if llm:
            tasks = [self._generate_group_summary(g, llm) for g in groups]
            summaries = await asyncio.gather(*tasks)
            for group, summary in zip(groups, summaries):
                group.llm_summary = summary

        return self._format_grouped_prompt(result, groups)

    def to_grouped_prompt_text(
        self,
        result: IssueExtractionResult,
        llm: Optional[LLMRunnable] = None,
        max_groups: int = 20,
        max_examples_per_group: int = 2,
    ) -> str:
        """
        Generate a grouped prompt with optional LLM summarization.

        Groups similar issues together and shows representative examples,
        reducing context size while preserving signal quality.

        Args:
            result: The IssueExtractionResult to convert
            llm: Optional LLM for generating group summaries
            max_groups: Maximum number of issue groups to include
            max_examples_per_group: Representative examples per group

        Returns:
            Formatted prompt text with grouped issues.
        """
        groups = self._group_issues(result, max_examples_per_group)[:max_groups]

        # Generate LLM summaries if LLM is provided
        if llm:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, use nest_asyncio pattern or skip
                    logger.warning(
                        'Running in async context. Use to_grouped_prompt_text_async() instead.'
                    )
                else:
                    tasks = [self._generate_group_summary(g, llm) for g in groups]
                    summaries = loop.run_until_complete(asyncio.gather(*tasks))
                    for group, summary in zip(groups, summaries):
                        group.llm_summary = summary
            except RuntimeError:
                # No event loop, create one
                summaries = asyncio.run(
                    asyncio.gather(
                        *[self._generate_group_summary(g, llm) for g in groups]
                    )
                )
                for group, summary in zip(groups, summaries):
                    group.llm_summary = summary

        return self._format_grouped_prompt(result, groups)

    def _format_grouped_prompt(
        self, result: IssueExtractionResult, groups: List[IssueGroup]
    ) -> str:
        """
        Format the grouped prompt text.

        Args:
            result: The original extraction result
            groups: The issue groups to format

        Returns:
            Formatted prompt string.
        """
        lines = []

        # Header
        lines.append('## Evaluation Issues Summary (Grouped)')
        lines.append('')
        if result.evaluation_name:
            lines.append(f'**Evaluation:** {result.evaluation_name}')
        lines.append(f'**Test Cases Analyzed:** {result.total_test_cases}')
        lines.append(f'**Total Issues Found:** {result.issues_found}')
        lines.append(f'**Issue Groups:** {len(groups)}')
        lines.append('')

        # Quick overview
        lines.append('### Issue Groups Overview')
        lines.append('')
        lines.append('| Metric | Signal | Count | Values |')
        lines.append('|--------|--------|-------|--------|')
        for group in groups:
            values_str = ', '.join(group.unique_values[:3])
            if len(group.unique_values) > 3:
                values_str += f' (+{len(group.unique_values) - 3} more)'
            lines.append(
                f'| {group.metric_name} | {group.signal_name} | {group.total_count} | {values_str} |'
            )
        lines.append('')

        # Detailed groups
        lines.append('### Detailed Issue Groups')
        lines.append('')

        for i, group in enumerate(groups, 1):
            lines.append(f'#### Group {i}: {group.metric_name} - {group.signal_name}')
            lines.append(f'- **Total Issues:** {group.total_count}')
            lines.append(f'- **Failure Values:** {", ".join(group.unique_values)}')

            if len(group.affected_test_cases) <= 5:
                lines.append(
                    f'- **Affected Tests:** {", ".join(group.affected_test_cases)}'
                )
            else:
                lines.append(
                    f'- **Affected Tests:** {", ".join(group.affected_test_cases[:5])} '
                    f'(+{len(group.affected_test_cases) - 5} more)'
                )

            # LLM summary if available
            if group.llm_summary:
                lines.append(f'- **Pattern Summary:** {group.llm_summary}')

            lines.append('')
            lines.append('**Representative Examples:**')

            for j, issue in enumerate(group.representative_issues, 1):
                lines.append(f'\n*Example {j}:*')
                lines.append(f'- Test: {issue.test_case_id}')
                lines.append(f'- Value: {issue.value}')
                if issue.reasoning:
                    lines.append(f'- Reasoning: "{issue.reasoning}"')
                if issue.item_context.get('query'):
                    query = issue.item_context['query']
                    if len(query) > 200:
                        query = query[:200] + '...'
                    lines.append(f'- Query: "{query}"')
                if issue.item_context.get('actual_output'):
                    output = issue.item_context['actual_output']
                    if len(output) > 200:
                        output = output[:200] + '...'
                    lines.append(f'- Output: "{output}"')

            lines.append('')

        # Task section
        lines.append('## Task')
        lines.append('Analyze the grouped quality issues. Provide:')
        lines.append(
            '1. **Critical Patterns:** Which issue groups represent the most significant problems?'
        )
        lines.append(
            '2. **Root Causes:** What underlying issues might be causing these failure patterns?'
        )
        lines.append(
            '3. **Recommended Actions:** Prioritized list of improvements to address these issues'
        )

        return '\n'.join(lines)
