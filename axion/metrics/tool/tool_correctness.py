from difflib import SequenceMatcher
from typing import Any, Dict, List, Literal

from axion._core.logging import get_logger
from axion._core.schema import ToolCall
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)

logger = get_logger(__name__)


@metric(
    name='Tool Correctness',
    key='tool_correctness',
    description='Evaluates whether the expected tools were correctly called by the agent.',
    required_fields=['tools_called', 'expected_tools'],
    optional_fields=[],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['agent', 'tool'],
)
class ToolCorrectness(BaseMetric):
    """
    A metric to evaluate tool calling correctness by comparing expected vs actual tool calls.

    This metric supports different evaluation modes:
    - Name-only matching (default): Only tool names must match
    - Parameter matching: Tool names and input parameters must match with various strategies
    - Strict matching: Tool names, parameters, and order must match exactly
    """

    def __init__(
        self,
        check_parameters: bool = False,
        strict_order: bool = False,
        parameter_matching_strategy: Literal['exact', 'subset', 'fuzzy'] = 'exact',
        fuzzy_threshold: float = 0.8,
        **kwargs,
    ):
        """
        Initialize the Tool Correctness metric.

        Args:
            check_parameters: If True, also validates that tool parameters match
            strict_order: If True, tools must be called in the exact expected order
            parameter_matching_strategy: Strategy for matching parameters:
                - "exact": Parameters must match exactly
                - "subset": Called args must contain all expected args (can have extras)
                - "fuzzy": Similarity-based matching with threshold
            fuzzy_threshold: Threshold for fuzzy matching (0.0 to 1.0)
            **kwargs: Additional arguments passed to BaseMetric
        """
        super().__init__(**kwargs)
        self.check_parameters = check_parameters
        self.strict_order = strict_order
        self.parameter_matching_strategy = parameter_matching_strategy
        self.fuzzy_threshold = fuzzy_threshold

    @trace(name='ToolCorrectness.execute', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem) -> MetricEvaluationResult:
        """
        Evaluate tool correctness for the given dataset item.

        Args:
            item: Dataset item containing tools_called and expected_tools

        Returns:
            MetricEvaluationResult with score and explanation
        """
        tools_called = item.tools_called or []
        expected_tools = item.expected_tools or []

        if not expected_tools and not tools_called:
            return MetricEvaluationResult(
                score=1.0,
                explanation='Correct: No tools were expected and none were called.',
            )

        if not expected_tools and tools_called:
            return MetricEvaluationResult(
                score=0.0,
                explanation=f'Incorrect: No tools were expected, but {len(tools_called)} were called.',
            )

        if self.strict_order:
            score, explanation = self._evaluate_strict_order(
                tools_called, expected_tools
            )
        else:
            score, explanation = self._evaluate_flexible_order(
                tools_called, expected_tools
            )

        return MetricEvaluationResult(score=score, explanation=explanation)

    @trace(name='evaluate_strict_order')
    def _evaluate_strict_order(
        self, tools_called: List[ToolCall], expected_tools: List[ToolCall]
    ) -> tuple[float, str]:
        """Evaluate with strict ordering requirement."""
        if len(tools_called) != len(expected_tools):
            return (
                0.0,
                f'Tool count mismatch: expected {len(expected_tools)}, got {len(tools_called)}',
            )

        for i, (called, expected) in enumerate(zip(tools_called, expected_tools)):
            if not self._tools_match(called, expected):
                # Provide a more detailed explanation for the mismatch
                param_explanation = ''
                if self.check_parameters:
                    param_explanation = f' with args {called.args}'
                return (
                    0.0,
                    f"Tool mismatch at position {i}: expected '{expected.name}', got '{called.name}'{param_explanation}",
                )

        return 1.0, f'All {len(expected_tools)} tools called correctly in order.'

    @trace(name='evaluate_flexible_order')
    def _evaluate_flexible_order(
        self, tools_called: List[ToolCall], expected_tools: List[ToolCall]
    ) -> tuple[float, str]:
        """
        [REWRITTEN] Evaluate without strict ordering, correctly handling duplicate tool calls.

        This implementation iterates through each expected tool and searches for a
        corresponding match in a mutable list of called tools. Once a match is found,
        it's removed to prevent it from being matched again.
        """
        if not expected_tools:
            return 0.0, f'Unexpected tools called: {[t.name for t in tools_called]}'

        remaining_called = list(tools_called)
        matched_pairs = []
        missing_or_incorrect = []

        for expected_tool in expected_tools:
            best_match_index = -1

            # Find the best possible match in the remaining called tools
            for i, called_tool in enumerate(remaining_called):
                if self._tools_match(called_tool, expected_tool):
                    best_match_index = i
                    break  # Found a perfect match, no need to look further

            if best_match_index != -1:
                # Remove the matched tool and add it to our list of successes
                matched_tool = remaining_called.pop(best_match_index)
                matched_pairs.append(
                    {'expected': expected_tool, 'called': matched_tool}
                )
            else:
                missing_or_incorrect.append(expected_tool)

        # The score is recall: number of correctly called tools / number of expected tools.
        score = len(matched_pairs) / len(expected_tools)

        explanation_parts = []
        if matched_pairs:
            explanation_parts.append(
                f"Correctly called: {[p['expected'].name for p in matched_pairs]}"
            )

        if missing_or_incorrect:
            # Show parameter details for missing tools if parameters are being checked
            if self.check_parameters:
                missing_details = [f'{t.name}({t.args})' for t in missing_or_incorrect]
                explanation_parts.append(f'Missing/incorrect calls: {missing_details}')
            else:
                explanation_parts.append(
                    f'Missing tools: {[t.name for t in missing_or_incorrect]}'
                )

        # Any tools left in `remaining_called` are unexpected.
        if remaining_called:
            explanation_parts.append(
                f'Unexpected tools: {[t.name for t in remaining_called]}'
            )

        if not explanation_parts:
            explanation = 'All expected tools were called correctly.'
        else:
            explanation = '; '.join(explanation_parts)

        return score, explanation

    def _tools_match(self, called_tool: ToolCall, expected_tool: ToolCall) -> bool:
        """Check if two tools match based on the configured criteria."""
        if called_tool.name != expected_tool.name:
            return False

        if self.check_parameters:
            return self._parameters_match(called_tool.args, expected_tool.args)

        return True

    def _parameters_match(
        self, called_args: Dict[str, Any], expected_args: Dict[str, Any]
    ) -> bool:
        """
        Check if tool parameters match with flexible matching strategies.
        """
        if self.parameter_matching_strategy == 'exact':
            return called_args == expected_args
        elif self.parameter_matching_strategy == 'subset':
            return self._subset_match(called_args, expected_args)
        elif self.parameter_matching_strategy == 'fuzzy':
            return self._fuzzy_params_match(called_args, expected_args)
        else:
            # Default to exact matching if strategy is unknown
            return called_args == expected_args

    def _subset_match(
        self, called_args: Dict[str, Any], expected_args: Dict[str, Any]
    ) -> bool:
        """Check if all expected parameters are present with correct values."""
        for key, expected_value in expected_args.items():
            if key not in called_args:
                return False

            # Recursively check for subset match in nested dictionaries
            if isinstance(expected_value, dict) and isinstance(
                called_args.get(key), dict
            ):
                if not self._subset_match(called_args[key], expected_value):
                    return False
            elif called_args[key] != expected_value:
                return False
        return True

    def _fuzzy_params_match(
        self, called_args: Dict[str, Any], expected_args: Dict[str, Any]
    ) -> bool:
        """
        Calculates an average similarity score across all expected
        parameters and compares it against a single threshold.
        """
        if not expected_args:
            return True  # No params to check

        total_similarity = 0.0
        for key, expected_val in expected_args.items():
            called_val = called_args.get(key)  # Returns None if key is missing
            total_similarity += self._calculate_similarity(called_val, expected_val)

        average_similarity = total_similarity / len(expected_args)
        return average_similarity >= self.fuzzy_threshold

    def _calculate_similarity(self, val1: Any, val2: Any) -> float:
        """
        Calculate a similarity score (0.0 to 1.0) between two values.

        This is now the single source of truth for comparing any two values,
        consolidating logic from previous helper functions.
        """
        # A value is missing entirely
        if val1 is None or val2 is None:
            return 0.0

        # Type mismatch is a total failure
        if type(val1) != type(val2):
            return 0.0

        if isinstance(val1, str):
            return self._string_similarity(val1, val2)

        if isinstance(val1, (int, float)):
            # Handle case where both numbers are zero
            if val1 == 0 and val2 == 0:
                return 1.0
            # Avoid division by zero if one number is zero
            if val1 == 0 or val2 == 0:
                return 0.0  # Or a small value if near-misses should count
            # Normalized distance for numbers
            return 1.0 - (abs(val1 - val2) / max(abs(val1), abs(val2)))

        if isinstance(val1, dict):
            # For dictionaries, calculate Jaccard similarity of keys and average value similarity
            keys1, keys2 = set(val1.keys()), set(val2.keys())
            intersecting_keys = keys1 & keys2
            union_keys = keys1 | keys2
            if not union_keys:
                return 1.0

            key_similarity = len(intersecting_keys) / len(union_keys)

            value_similarity = 0.0
            if intersecting_keys:
                for key in intersecting_keys:
                    value_similarity += self._calculate_similarity(val1[key], val2[key])
                value_similarity /= len(intersecting_keys)

            return (key_similarity + value_similarity) / 2.0

        # For lists, tuples, and other types
        return 1.0 if val1 == val2 else 0.0

    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity"""
        return SequenceMatcher(None, str1, str2).ratio()
