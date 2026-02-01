"""Tests for prompt optimizer module."""

import pytest

from axion.caliber.prompt_optimizer import (
    OptimizedPrompt,
    OptimizeInput,
    OptimizeOutput,
    PromptOptimizer,
    PromptSuggestion,
    SuggestionOutput,
)
from axion.caliber.analysis import MisalignedCase


class TestPromptSuggestion:
    """Tests for PromptSuggestion dataclass."""

    def test_create_suggestion(self):
        """Test creating a suggestion."""
        suggestion = PromptSuggestion(
            aspect='Criteria Clarity',
            suggestion='Break down accuracy into components',
            rationale='Original criteria is ambiguous',
        )
        assert suggestion.aspect == 'Criteria Clarity'
        assert suggestion.suggestion == 'Break down accuracy into components'
        assert suggestion.rationale == 'Original criteria is ambiguous'


class TestOptimizedPrompt:
    """Tests for OptimizedPrompt dataclass."""

    def test_create_optimized_prompt(self):
        """Test creating optimized prompt result."""
        result = OptimizedPrompt(
            original_criteria='Score 1 if accurate.',
            optimized_criteria='Score 1 if factually correct and addresses the question.',
            suggestions=[
                PromptSuggestion(
                    aspect='Clarity',
                    suggestion='Define accurate',
                    rationale='Ambiguous term',
                )
            ],
            expected_improvement='Should reduce false positives by 30%',
            metadata={'model': 'gpt-4o'},
        )
        assert result.original_criteria == 'Score 1 if accurate.'
        assert 'factually correct' in result.optimized_criteria
        assert len(result.suggestions) == 1
        assert result.metadata['model'] == 'gpt-4o'


class TestPydanticModels:
    """Tests for Pydantic input/output models."""

    def test_optimize_input(self):
        """Test OptimizeInput model."""
        fp_case = MisalignedCase(
            record_id='r1',
            query='Q',
            actual_output='A',
            human_score=0,
            llm_score=1,
        )
        fn_case = MisalignedCase(
            record_id='r2',
            query='Q',
            actual_output='A',
            human_score=1,
            llm_score=0,
        )

        input_data = OptimizeInput(
            system_prompt='You are an evaluator.',
            current_criteria='Score 1 if good.',
            false_positive_examples=[fp_case],
            false_negative_examples=[fn_case],
            total_misaligned=2,
            false_positives=1,
            false_negatives=1,
        )

        assert input_data.system_prompt == 'You are an evaluator.'
        assert len(input_data.false_positive_examples) == 1
        assert len(input_data.false_negative_examples) == 1
        assert input_data.total_misaligned == 2

    def test_suggestion_output(self):
        """Test SuggestionOutput model."""
        suggestion = SuggestionOutput(
            aspect='Edge Cases',
            suggestion='Add guidance for borderline cases',
            rationale='Reduces ambiguity',
        )
        assert suggestion.aspect == 'Edge Cases'

    def test_optimize_output(self):
        """Test OptimizeOutput model."""
        output = OptimizeOutput(
            optimized_criteria='New criteria text',
            suggestions=[
                SuggestionOutput(
                    aspect='Clarity',
                    suggestion='Be specific',
                    rationale='Helps consistency',
                )
            ],
            expected_improvement='Better alignment expected',
        )
        assert output.optimized_criteria == 'New criteria text'
        assert len(output.suggestions) == 1


class TestPromptOptimizer:
    """Tests for PromptOptimizer class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        optimizer = PromptOptimizer()
        assert optimizer._max_examples == 10

    def test_init_custom(self):
        """Test initialization with custom params."""
        optimizer = PromptOptimizer(
            model_name='gpt-4o',
            llm_provider='openai',
            max_examples=20,
        )
        assert optimizer._model_name == 'gpt-4o'
        assert optimizer._llm_provider == 'openai'
        assert optimizer._max_examples == 20

    @pytest.mark.asyncio
    async def test_optimize_perfect_alignment(self):
        """Test optimization with no misalignments."""
        optimizer = PromptOptimizer(model_name='gpt-4o')

        # All aligned results
        results = [
            {'record_id': 'r1', 'human_score': 1, 'llm_score': 1, 'query': 'Q', 'actual_output': 'A'},
            {'record_id': 'r2', 'human_score': 0, 'llm_score': 0, 'query': 'Q', 'actual_output': 'A'},
        ]

        criteria = 'Score 1 if accurate.'
        optimized = await optimizer.optimize(results, criteria)

        assert optimized.original_criteria == criteria
        assert optimized.optimized_criteria == criteria  # No change needed
        assert optimized.suggestions == []
        assert 'No optimization needed' in optimized.expected_improvement
        assert optimized.metadata.get('perfect_alignment') is True

    def test_handler_lazy_init(self):
        """Test that handler is lazily initialized."""
        optimizer = PromptOptimizer(model_name='gpt-4o')
        assert optimizer._handler is None

        # Handler should be created on first access
        # (We don't call _get_handler() here to avoid needing API key)


class TestPromptOptimizerInputValidation:
    """Tests for input validation in PromptOptimizer."""

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Test with empty results."""
        optimizer = PromptOptimizer(model_name='gpt-4o')

        optimized = await optimizer.optimize([], 'Some criteria')

        # No misalignments found
        assert optimized.optimized_criteria == 'Some criteria'
        assert optimized.metadata.get('perfect_alignment') is True

    @pytest.mark.asyncio
    async def test_only_false_positives(self):
        """Test results with only false positives returns early if no misalignment."""
        optimizer = PromptOptimizer(model_name='gpt-4o')

        # All aligned - just testing the extract logic
        results = [
            {'record_id': 'r1', 'human_score': 1, 'llm_score': 1, 'query': 'Q', 'actual_output': 'A'},
        ]

        optimized = await optimizer.optimize(results, 'Criteria')
        assert optimized.metadata.get('perfect_alignment') is True
