"""
Signal formatting utilities for extracting human-readable feedback from metric results.

This module provides utilities to convert the rich signal data from Axion metrics
into actionable, human-readable feedback that can be used by the optimizer agent
to understand why specific test cases failed.
"""

from typing import Any, Dict, List

from axion._core.logging import get_logger
from axion.schema import MetricScore, TestResult

logger = get_logger(__name__)


class SignalFormatter:
    """
    Extracts and formats human-readable feedback from Axion metric signals.

    Different metrics produce different signal structures:
    - Faithfulness: judged_claims with faithfulness_verdict and reason
    - AnswerRelevancy: aspect coverage information
    - Other metrics: general score with explanation

    This class normalizes these into actionable feedback strings.
    """

    @staticmethod
    def format_metric_failure(metric_score: MetricScore) -> str:
        """
        Format a single failed metric score into human-readable feedback.

        Args:
            metric_score: The MetricScore that failed

        Returns:
            A formatted string describing why this metric failed
        """
        output_lines = [
            f"Metric '{metric_score.name}' (score: {metric_score.score:.2f}):"
        ]

        signals = metric_score.signals
        explanation = metric_score.explanation

        # Try to extract structured feedback from signals
        if signals is not None:
            signal_feedback = SignalFormatter._extract_signal_feedback(
                metric_score.name, signals
            )
            if signal_feedback:
                output_lines.extend(signal_feedback)

        # Add explanation if available and not redundant
        if explanation and explanation != 'Check signals for additional details':
            # Truncate long explanations
            if len(explanation) > 300:
                explanation = explanation[:300] + '...'
            output_lines.append(f'  Explanation: {explanation}')

        return '\n'.join(output_lines)

    @staticmethod
    def _extract_signal_feedback(metric_name: str, signals: Any) -> List[str]:
        """
        Extract feedback from metric-specific signal structures.

        Args:
            metric_name: Name of the metric
            signals: The signals data (varies by metric type)

        Returns:
            List of feedback strings
        """
        feedback = []
        metric_lower = metric_name.lower()

        # Handle Faithfulness signals (judged_claims structure)
        if 'faithfulness' in metric_lower:
            feedback.extend(SignalFormatter._format_faithfulness_signals(signals))

        # Handle AnswerRelevancy signals
        elif 'relevancy' in metric_lower or 'relevance' in metric_lower:
            feedback.extend(SignalFormatter._format_relevancy_signals(signals))

        # Handle Completeness signals
        elif 'completeness' in metric_lower:
            feedback.extend(SignalFormatter._format_completeness_signals(signals))

        # Handle generic signals structure from SignalExtractor
        elif isinstance(signals, dict):
            feedback.extend(SignalFormatter._format_generic_signals(signals))

        # Handle Pydantic model signals
        elif hasattr(signals, 'model_dump'):
            try:
                feedback.extend(
                    SignalFormatter._format_generic_signals(signals.model_dump())
                )
            except Exception:
                pass

        return feedback

    @staticmethod
    def _format_faithfulness_signals(signals: Any) -> List[str]:
        """Format Faithfulness metric signals (FaithfulnessResult structure)."""
        feedback = []

        # Check if it's a FaithfulnessResult model
        judged_claims = None
        if hasattr(signals, 'judged_claims'):
            judged_claims = signals.judged_claims
        elif isinstance(signals, dict) and 'judged_claims' in signals:
            judged_claims = signals['judged_claims']

        if judged_claims:
            # Find claims that are not fully supported
            failed_claims = []
            for claim in judged_claims:
                verdict = None
                if hasattr(claim, 'faithfulness_verdict'):
                    verdict = claim.faithfulness_verdict
                elif isinstance(claim, dict):
                    verdict = claim.get('faithfulness_verdict')

                if verdict:
                    verdict_str = str(verdict).lower()
                    if 'fully supported' not in verdict_str:
                        claim_text = (
                            claim.claim_text
                            if hasattr(claim, 'claim_text')
                            else claim.get('claim_text', '')
                        )
                        reason = (
                            claim.reason
                            if hasattr(claim, 'reason')
                            else claim.get('reason', '')
                        )
                        failed_claims.append((claim_text, verdict_str, reason))

            # Format top failures
            for claim_text, verdict, reason in failed_claims[:3]:
                claim_preview = (
                    claim_text[:80] + '...' if len(claim_text) > 80 else claim_text
                )
                feedback.append(f"  - [CLAIM NOT SUPPORTED] '{claim_preview}'")
                feedback.append(f'    Verdict: {verdict}')
                if reason:
                    feedback.append(f'    Reason: {reason}')

        # Also check verdict_counts if available
        verdict_counts = None
        if hasattr(signals, 'verdict_counts'):
            verdict_counts = signals.verdict_counts
        elif isinstance(signals, dict) and 'verdict_counts' in signals:
            verdict_counts = signals['verdict_counts']

        if verdict_counts:
            issues = []
            for verdict, count in verdict_counts.items():
                if count > 0 and 'fully_supported' not in str(verdict).lower():
                    issues.append(f'{verdict}: {count}')
            if issues:
                feedback.append(f'  Verdict summary: {", ".join(issues)}')

        return feedback

    @staticmethod
    def _format_relevancy_signals(signals: Any) -> List[str]:
        """Format AnswerRelevancy metric signals."""
        feedback = []

        # Handle aspect breakdown if present
        aspect_breakdown = None
        if hasattr(signals, 'aspect_breakdown'):
            aspect_breakdown = signals.aspect_breakdown
        elif isinstance(signals, dict) and 'aspect_breakdown' in signals:
            aspect_breakdown = signals['aspect_breakdown']

        if aspect_breakdown:
            uncovered = []
            for aspect in aspect_breakdown:
                covered = (
                    aspect.covered
                    if hasattr(aspect, 'covered')
                    else aspect.get('covered', True)
                )
                if not covered:
                    aspect_name = (
                        aspect.aspect
                        if hasattr(aspect, 'aspect')
                        else aspect.get('aspect', 'Unknown')
                    )
                    uncovered.append(aspect_name)

            if uncovered:
                feedback.append(f'  - [MISSING ASPECTS] {", ".join(uncovered[:5])}')

        # Handle relevancy verdict
        relevancy_verdict = None
        if hasattr(signals, 'relevancy_verdict'):
            relevancy_verdict = signals.relevancy_verdict
        elif isinstance(signals, dict) and 'relevancy_verdict' in signals:
            relevancy_verdict = signals['relevancy_verdict']

        if relevancy_verdict and str(relevancy_verdict).lower() in ['no', 'idk']:
            feedback.append(f'  - [NOT RELEVANT] Verdict: {relevancy_verdict}')

        return feedback

    @staticmethod
    def _format_completeness_signals(signals: Any) -> List[str]:
        """Format Completeness metric signals."""
        feedback = []

        # Check for missing information
        missing_info = None
        if hasattr(signals, 'missing_information'):
            missing_info = signals.missing_information
        elif isinstance(signals, dict) and 'missing_information' in signals:
            missing_info = signals['missing_information']

        if missing_info and isinstance(missing_info, list):
            for item in missing_info[:3]:
                feedback.append(f'  - [MISSING INFO] {item}')

        return feedback

    @staticmethod
    def _format_generic_signals(signals: Dict) -> List[str]:
        """Format generic signal dictionary structure."""
        feedback = []

        # Handle grouped signals from SignalExtractor
        for group_name, group_signals in signals.items():
            if isinstance(group_signals, list):
                for signal in group_signals:
                    if isinstance(signal, dict):
                        name = signal.get('name', '')
                        value = signal.get('value', '')
                        score = signal.get('score')
                        if score is not None and score < 0.5:
                            feedback.append(
                                f'  - [{name}] {value} (score: {score:.2f})'
                            )
            elif isinstance(group_signals, (int, float)) and group_signals < 0.5:
                feedback.append(f'  - [{group_name}] score: {group_signals:.2f}')

        return feedback

    @staticmethod
    def format_test_result_failures(test_result: TestResult) -> str:
        """
        Format all failure information from a TestResult.

        Args:
            test_result: The TestResult containing failed metrics

        Returns:
            Formatted string with query, output, and failure details
        """
        lines = []

        # Add test case context
        if test_result.test_case:
            item = test_result.test_case
            query = getattr(item, 'query', '') or ''
            actual_output = getattr(item, 'actual_output', '') or ''

            # Truncate long content
            if len(query) > 200:
                query = query[:200] + '...'
            if len(actual_output) > 300:
                actual_output = actual_output[:300] + '...'

            lines.append(f'Query: {query}')
            lines.append(f'Output: {actual_output}')
            lines.append('')

        # Add metric failure details
        for score in test_result.score_results:
            if score.passed is False:
                lines.append(SignalFormatter.format_metric_failure(score))
                lines.append('')

        return '\n'.join(lines)

    @staticmethod
    def format_hard_negatives(
        hard_negatives: List[TestResult], max_failures: int = 3
    ) -> str:
        """
        Format multiple hard negative test results into a single feedback string.

        Args:
            hard_negatives: List of TestResult objects that failed
            max_failures: Maximum number of failures to include

        Returns:
            Formatted string describing all failures
        """
        if not hard_negatives:
            return 'No failures to analyze.'

        sections = []
        for i, test_result in enumerate(hard_negatives[:max_failures], 1):
            sections.append(f'--- Failure {i} ---')
            sections.append(SignalFormatter.format_test_result_failures(test_result))

        return '\n'.join(sections)
