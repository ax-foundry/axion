"""Tests for misalignment analysis and prompt optimization module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axion.align.analysis import (
    MisalignedCase,
    MisalignmentAnalysis,
    MisalignmentAnalyzer,
    MisalignmentInput,
    MisalignmentOutput,
    MisalignmentPattern,
    OptimizedPrompt,
    OptimizeInput,
    OptimizeOutput,
    PatternOutput,
    PromptOptimizer,
    PromptSuggestion,
    SuggestionOutput,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_results():
    """Sample evaluation results with misalignments."""
    return [
        {
            'record_id': 'r1',
            'human_score': 1,
            'llm_score': 0,
            'query': 'What is the capital of France?',
            'actual_output': 'Paris is the capital of France.',
            'llm_reasoning': 'Response is too brief.',
        },
        {
            'record_id': 'r2',
            'human_score': 0,
            'llm_score': 1,
            'query': 'Explain quantum computing',
            'actual_output': 'Quantum computing uses qubits.',
            'llm_reasoning': 'Response addresses the question.',
        },
        {
            'record_id': 'r3',
            'human_score': 1,
            'llm_score': 1,  # Aligned
            'query': 'What is 2+2?',
            'actual_output': 'The answer is 4.',
            'llm_reasoning': 'Correct answer.',
        },
        {
            'record_id': 'r4',
            'human_score': 0,
            'llm_score': 0,  # Aligned
            'query': 'Tell me a joke',
            'actual_output': 'Invalid response.',
            'llm_reasoning': 'Not a joke.',
        },
    ]


@pytest.fixture
def sample_results_all_false_positives():
    """Results with only false positives (LLM too lenient)."""
    return [
        {
            'record_id': 'r1',
            'human_score': 0,
            'llm_score': 1,
            'query': 'Query 1',
            'actual_output': 'Output 1',
            'llm_reasoning': 'Reasoning 1',
        },
        {
            'record_id': 'r2',
            'human_score': 0,
            'llm_score': 1,
            'query': 'Query 2',
            'actual_output': 'Output 2',
            'llm_reasoning': 'Reasoning 2',
        },
    ]


@pytest.fixture
def sample_results_aligned():
    """Results with no misalignments."""
    return [
        {
            'record_id': 'r1',
            'human_score': 1,
            'llm_score': 1,
            'query': 'Query 1',
            'actual_output': 'Output 1',
        },
        {
            'record_id': 'r2',
            'human_score': 0,
            'llm_score': 0,
            'query': 'Query 2',
            'actual_output': 'Output 2',
        },
    ]


@pytest.fixture
def mock_misalignment_output():
    """Mock output from misalignment analysis handler."""
    return MisalignmentOutput(
        summary='The LLM is too strict on brief responses and too lenient on incomplete explanations.',
        patterns=[
            PatternOutput(
                pattern_type='false_negative',
                description='LLM rejects concise but correct answers',
                example_ids=['r1'],
            ),
            PatternOutput(
                pattern_type='false_positive',
                description='LLM accepts incomplete technical explanations',
                example_ids=['r2'],
            ),
        ],
        recommendations=[
            'Clarify that brevity is acceptable if accurate',
            'Add criteria for technical completeness',
            'Include examples of borderline cases',
        ],
    )


@pytest.fixture
def mock_optimize_output():
    """Mock output from prompt optimization handler."""
    return OptimizeOutput(
        optimized_criteria='Evaluate responses for accuracy and completeness. Brief answers are acceptable if correct. Technical responses should cover key concepts.',
        suggestions=[
            SuggestionOutput(
                aspect='Brevity guidance',
                suggestion='Add explicit note that concise answers are acceptable',
                rationale='Reduces false negatives on short but correct responses',
            ),
            SuggestionOutput(
                aspect='Technical depth',
                suggestion='Specify minimum depth for technical topics',
                rationale='Reduces false positives on incomplete explanations',
            ),
        ],
        expected_improvement='These changes should reduce misalignment by clarifying ambiguous cases around response length and technical depth.',
    )


# ============================================================================
# Unit Tests for _extract_misaligned_cases
# ============================================================================


class TestExtractMisalignedCases:
    """Tests for extracting misaligned cases."""

    def test_extract_from_dict_results(self, sample_results):
        """Test extracting misaligned cases from dict results."""
        analyzer = MisalignmentAnalyzer()
        fp, fn = analyzer._extract_misaligned_cases(sample_results)

        assert len(fp) == 1  # r2: LLM=1, Human=0
        assert len(fn) == 1  # r1: LLM=0, Human=1
        assert fp[0].record_id == 'r2'
        assert fn[0].record_id == 'r1'

    def test_extract_all_false_positives(self, sample_results_all_false_positives):
        """Test extraction when all misalignments are false positives."""
        analyzer = MisalignmentAnalyzer()
        fp, fn = analyzer._extract_misaligned_cases(sample_results_all_false_positives)

        assert len(fp) == 2
        assert len(fn) == 0

    def test_extract_no_misalignments(self, sample_results_aligned):
        """Test extraction when there are no misalignments."""
        analyzer = MisalignmentAnalyzer()
        fp, fn = analyzer._extract_misaligned_cases(sample_results_aligned)

        assert len(fp) == 0
        assert len(fn) == 0

    def test_extract_handles_missing_fields(self):
        """Test extraction handles missing optional fields."""
        results = [
            {
                'record_id': 'r1',
                'human_score': 1,
                'llm_score': 0,
                # Missing query, actual_output, llm_reasoning
            },
        ]
        analyzer = MisalignmentAnalyzer()
        fp, fn = analyzer._extract_misaligned_cases(results)

        assert len(fn) == 1
        assert fn[0].query == ''
        assert fn[0].actual_output == ''
        assert fn[0].llm_reasoning is None

    def test_extract_accepts_datasetitem_and_metric_aliases(self):
        """Test extraction supports DatasetItem/MetricScore-style aliases."""
        results = [
            {
                # record_id alias
                'id': 'r1',
                # DatasetItem-style (human)
                'judgment': 1,
                # MetricScore-style (LLM judge)
                'score': 0,
                'query': 'Q1',
                'actual_output': 'O1',
                # reasoning alias (MetricScore uses "explanation")
                'explanation': 'Model was too strict.',
                # future-proof: carry structured signals through
                'signals': {'rule': 'brevity', 'confidence': 0.31},
            }
        ]
        analyzer = MisalignmentAnalyzer()
        fp, fn = analyzer._extract_misaligned_cases(results)

        assert len(fp) == 0
        assert len(fn) == 1
        assert fn[0].record_id == 'r1'
        assert fn[0].human_score == 1
        assert fn[0].llm_score == 0
        assert fn[0].llm_reasoning == 'Model was too strict.'
        assert fn[0].signals == {'rule': 'brevity', 'confidence': 0.31}

    def test_extract_does_not_fallback_when_primary_key_present(self):
        """If llm_reasoning is present (even None), don't fall back to explanation."""
        results = [
            {
                'record_id': 'r1',
                'human_score': 1,
                'llm_score': 0,
                'query': 'Q1',
                'actual_output': 'O1',
                'llm_reasoning': None,
                'explanation': 'Should NOT be used.',
            }
        ]
        analyzer = MisalignmentAnalyzer()
        fp, fn = analyzer._extract_misaligned_cases(results)

        assert len(fn) == 1
        assert fn[0].llm_reasoning is None

    def test_extract_object_with_aliases(self):
        """Test extraction from objects using alias attributes."""

        class MetricScoreResult:
            """Simulates a MetricScore-like object with alias attributes."""

            def __init__(self):
                self.id = 'obj_1'  # alias for record_id
                self.judgment = 1  # alias for human_score
                self.score = 0  # alias for llm_score
                self.query = 'Object query'
                self.actual_output = 'Object output'
                self.explanation = 'Object explanation'  # alias for llm_reasoning
                self.signals = {'source': 'metric'}

        results = [MetricScoreResult()]
        analyzer = MisalignmentAnalyzer()
        fp, fn = analyzer._extract_misaligned_cases(results)

        assert len(fp) == 0
        assert len(fn) == 1
        assert fn[0].record_id == 'obj_1'
        assert fn[0].human_score == 1
        assert fn[0].llm_score == 0
        assert fn[0].llm_reasoning == 'Object explanation'
        assert fn[0].signals == {'source': 'metric'}


# ============================================================================
# Integration Tests with Mocked LLM - MisalignmentAnalyzer
# ============================================================================


class TestMisalignmentAnalyzer:
    """Tests for MisalignmentAnalyzer."""

    @pytest.mark.asyncio
    async def test_analyze_returns_analysis(
        self, sample_results, mock_misalignment_output
    ):
        """Test that analyze returns expected analysis result."""
        analyzer = MisalignmentAnalyzer()

        with patch.object(analyzer, '_get_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.execute = AsyncMock(return_value=mock_misalignment_output)
            mock_handler.model_name = 'gpt-4o'
            mock_handler.llm_provider = 'openai'
            mock_get_handler.return_value = mock_handler

            result = await analyzer.analyze(
                sample_results, evaluation_criteria='Be accurate and helpful.'
            )

        assert isinstance(result, MisalignmentAnalysis)
        assert result.total_misaligned == 2
        assert result.false_positives == 1
        assert result.false_negatives == 1
        assert len(result.patterns) == 2
        assert len(result.recommendations) == 3
        assert (
            'too strict' in result.summary.lower() or 'brief' in result.summary.lower()
        )

    @pytest.mark.asyncio
    async def test_analyze_perfect_alignment(self, sample_results_aligned):
        """Test analyze with no misalignments."""
        analyzer = MisalignmentAnalyzer()

        result = await analyzer.analyze(
            sample_results_aligned, evaluation_criteria='Test criteria'
        )

        assert result.total_misaligned == 0
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.patterns == []
        assert 'perfect alignment' in result.summary.lower()
        assert result.metadata.get('perfect_alignment') is True

    @pytest.mark.asyncio
    async def test_analyze_samples_large_datasets(self):
        """Test that large datasets are sampled."""
        # Create 20 false positives
        large_results = [
            {
                'record_id': f'r{i}',
                'human_score': 0,
                'llm_score': 1,
                'query': f'Query {i}',
                'actual_output': f'Output {i}',
            }
            for i in range(20)
        ]

        analyzer = MisalignmentAnalyzer(max_examples=5)

        mock_output = MisalignmentOutput(
            summary='Test summary',
            patterns=[],
            recommendations=['Test rec'],
        )

        with patch.object(analyzer, '_get_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.execute = AsyncMock(return_value=mock_output)
            mock_handler.model_name = 'gpt-4o'
            mock_handler.llm_provider = 'openai'
            mock_get_handler.return_value = mock_handler

            result = await analyzer.analyze(large_results, 'Test criteria')

            # Check that the input was sampled
            call_args = mock_handler.execute.call_args[0][0]
            assert len(call_args.false_positive_examples) == 5  # max_examples

        assert result.false_positives == 20  # Total count preserved


# ============================================================================
# Integration Tests with Mocked LLM - PromptOptimizer
# ============================================================================


class TestPromptOptimizer:
    """Tests for PromptOptimizer."""

    @pytest.mark.asyncio
    async def test_optimize_returns_result(self, sample_results, mock_optimize_output):
        """Test that optimize returns expected result."""
        optimizer = PromptOptimizer()

        with patch.object(optimizer, '_get_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.execute = AsyncMock(return_value=mock_optimize_output)
            mock_handler.model_name = 'gpt-4o'
            mock_handler.llm_provider = 'openai'
            mock_get_handler.return_value = mock_handler

            result = await optimizer.optimize(
                sample_results,
                current_criteria='Be accurate.',
                system_prompt='You are an evaluator.',
            )

        assert isinstance(result, OptimizedPrompt)
        assert result.original_criteria == 'Be accurate.'
        assert 'accuracy' in result.optimized_criteria.lower()
        assert len(result.suggestions) == 2
        assert result.expected_improvement

    @pytest.mark.asyncio
    async def test_optimize_perfect_alignment(self, sample_results_aligned):
        """Test optimize with no misalignments."""
        optimizer = PromptOptimizer()

        result = await optimizer.optimize(
            sample_results_aligned,
            current_criteria='Test criteria',
            system_prompt='Test prompt',
        )

        assert result.original_criteria == 'Test criteria'
        assert result.optimized_criteria == 'Test criteria'
        assert result.suggestions == []
        assert 'no optimization' in result.expected_improvement.lower()
        assert result.metadata.get('perfect_alignment') is True

    @pytest.mark.asyncio
    async def test_optimize_default_system_prompt(
        self, sample_results, mock_optimize_output
    ):
        """Test that default system prompt is used when not provided."""
        optimizer = PromptOptimizer()

        with patch.object(optimizer, '_get_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.execute = AsyncMock(return_value=mock_optimize_output)
            mock_handler.model_name = 'gpt-4o'
            mock_handler.llm_provider = 'openai'
            mock_get_handler.return_value = mock_handler

            await optimizer.optimize(
                sample_results, current_criteria='Test criteria'
            )  # No system_prompt

            call_args = mock_handler.execute.call_args[0][0]
            assert call_args.system_prompt == 'You are an AI evaluator.'


# ============================================================================
# Pydantic Model Tests
# ============================================================================


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_misaligned_case_validation(self):
        """Test MisalignedCase validation."""
        case = MisalignedCase(
            record_id='r1',
            query='Test query',
            actual_output='Test output',
            human_score=1,
            llm_score=0,
            llm_reasoning='Test reasoning',
        )
        assert case.record_id == 'r1'
        assert case.human_score == 1
        assert case.llm_score == 0

    def test_misalignment_input_validation(self):
        """Test MisalignmentInput validation."""
        input_data = MisalignmentInput(
            false_positive_examples=[
                MisalignedCase(
                    record_id='r1',
                    query='Q',
                    actual_output='O',
                    human_score=0,
                    llm_score=1,
                )
            ],
            false_negative_examples=[],
            evaluation_criteria='Test criteria',
        )
        assert len(input_data.false_positive_examples) == 1
        assert len(input_data.false_negative_examples) == 0

    def test_misalignment_output_validation(self):
        """Test MisalignmentOutput validation."""
        output = MisalignmentOutput(
            summary='Test summary',
            patterns=[
                PatternOutput(
                    pattern_type='false_positive',
                    description='Test pattern',
                    example_ids=['r1', 'r2'],
                )
            ],
            recommendations=['Rec 1', 'Rec 2'],
        )
        assert output.summary == 'Test summary'
        assert len(output.patterns) == 1
        assert len(output.recommendations) == 2

    def test_optimize_input_validation(self):
        """Test OptimizeInput validation."""
        input_data = OptimizeInput(
            system_prompt='Test prompt',
            current_criteria='Test criteria',
            false_positive_examples=[],
            false_negative_examples=[],
            total_misaligned=0,
            false_positives=0,
            false_negatives=0,
        )
        assert input_data.system_prompt == 'Test prompt'

    def test_optimize_output_validation(self):
        """Test OptimizeOutput validation."""
        output = OptimizeOutput(
            optimized_criteria='New criteria',
            suggestions=[
                SuggestionOutput(
                    aspect='Test aspect',
                    suggestion='Test suggestion',
                    rationale='Test rationale',
                )
            ],
            expected_improvement='Test improvement',
        )
        assert output.optimized_criteria == 'New criteria'
        assert len(output.suggestions) == 1


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestDataclasses:
    """Tests for dataclass structures."""

    def test_misalignment_pattern_creation(self):
        """Test MisalignmentPattern creation."""
        pattern = MisalignmentPattern(
            pattern_type='false_positive',
            description='Test description',
            count=5,
            example_ids=['r1', 'r2', 'r3', 'r4', 'r5'],
        )
        assert pattern.pattern_type == 'false_positive'
        assert pattern.count == 5

    def test_misalignment_analysis_defaults(self):
        """Test MisalignmentAnalysis with default values."""
        analysis = MisalignmentAnalysis(
            total_misaligned=10,
            false_positives=6,
            false_negatives=4,
            patterns=[],
            summary='Test summary',
            recommendations=['Rec 1'],
        )
        assert analysis.metadata == {}

    def test_prompt_suggestion_creation(self):
        """Test PromptSuggestion creation."""
        suggestion = PromptSuggestion(
            aspect='Test aspect',
            suggestion='Test suggestion',
            rationale='Test rationale',
        )
        assert suggestion.aspect == 'Test aspect'

    def test_optimized_prompt_defaults(self):
        """Test OptimizedPrompt with default values."""
        result = OptimizedPrompt(
            original_criteria='Original',
            optimized_criteria='Optimized',
            suggestions=[],
            expected_improvement='Test improvement',
        )
        assert result.metadata == {}


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_results_list(self):
        """Test with empty results list."""
        analyzer = MisalignmentAnalyzer()
        result = await analyzer.analyze([], 'Test criteria')

        assert result.total_misaligned == 0
        assert result.metadata.get('perfect_alignment') is True

    @pytest.mark.asyncio
    async def test_results_with_object_attributes(self):
        """Test extraction from objects with attributes."""

        class ResultObject:
            def __init__(self, record_id, human_score, llm_score):
                self.record_id = record_id
                self.human_score = human_score
                self.llm_score = llm_score
                self.query = 'Test query'
                self.actual_output = 'Test output'
                self.llm_reasoning = 'Test reasoning'

        results = [
            ResultObject('r1', 1, 0),  # False negative
            ResultObject('r2', 0, 1),  # False positive
        ]

        analyzer = MisalignmentAnalyzer()
        fp, fn = analyzer._extract_misaligned_cases(results)

        assert len(fp) == 1
        assert len(fn) == 1
        assert fp[0].record_id == 'r2'
        assert fn[0].record_id == 'r1'

    @pytest.mark.asyncio
    async def test_all_false_negatives(self):
        """Test when all misalignments are false negatives."""
        results = [
            {
                'record_id': f'r{i}',
                'human_score': 1,
                'llm_score': 0,
                'query': f'Q{i}',
                'actual_output': f'O{i}',
            }
            for i in range(5)
        ]

        analyzer = MisalignmentAnalyzer()
        mock_output = MisalignmentOutput(
            summary='LLM is too strict.',
            patterns=[],
            recommendations=['Be more lenient'],
        )

        with patch.object(analyzer, '_get_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.execute = AsyncMock(return_value=mock_output)
            mock_handler.model_name = 'gpt-4o'
            mock_handler.llm_provider = 'openai'
            mock_get_handler.return_value = mock_handler

            result = await analyzer.analyze(results, 'Test criteria')

        assert result.false_positives == 0
        assert result.false_negatives == 5
        assert result.total_misaligned == 5


# ============================================================================
# Integration Tests for Handler Initialization
# ============================================================================


class TestHandlerInitialization:
    """Tests for handler lazy initialization."""

    def test_analyzer_lazy_handler_creation(self):
        """Test that analyzer creates handler lazily."""
        analyzer = MisalignmentAnalyzer(model_name='gpt-4o', llm_provider='openai')
        assert analyzer._handler is None

        # Handler should still be None until _get_handler is called
        assert analyzer._handler is None

    def test_optimizer_lazy_handler_creation(self):
        """Test that optimizer creates handler lazily."""
        optimizer = PromptOptimizer(model_name='gpt-4o', llm_provider='openai')
        assert optimizer._handler is None

    def test_analyzer_custom_instruction(self):
        """Test analyzer with custom instruction."""
        custom_instruction = 'Custom analysis instruction'
        analyzer = MisalignmentAnalyzer(instruction=custom_instruction)

        assert analyzer._instruction == custom_instruction

    def test_optimizer_custom_max_examples(self):
        """Test optimizer with custom max_examples."""
        optimizer = PromptOptimizer(max_examples=20)
        assert optimizer._max_examples == 20
