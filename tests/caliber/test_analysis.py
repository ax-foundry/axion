"""Tests for misalignment analysis module."""

import pytest

from axion.caliber.analysis import (
    MisalignedCase,
    MisalignmentAnalysis,
    MisalignmentAnalyzer,
    MisalignmentPattern,
    extract_misaligned_cases,
)


class TestMisalignedCase:
    """Tests for MisalignedCase model."""

    def test_create_case(self):
        """Test creating a misaligned case."""
        case = MisalignedCase(
            record_id='r1',
            query='What is Python?',
            actual_output='A snake.',
            human_score=0,
            llm_score=1,
            llm_reasoning='Response mentions Python',
        )
        assert case.record_id == 'r1'
        assert case.human_score == 0
        assert case.llm_score == 1
        assert case.llm_reasoning == 'Response mentions Python'

    def test_create_minimal(self):
        """Test creating case with minimal fields."""
        case = MisalignedCase(
            record_id='r1',
            query='Q',
            actual_output='A',
            human_score=1,
            llm_score=0,
        )
        assert case.llm_reasoning is None
        assert case.signals is None


class TestMisalignmentPattern:
    """Tests for MisalignmentPattern dataclass."""

    def test_create_pattern(self):
        """Test creating a misalignment pattern."""
        pattern = MisalignmentPattern(
            pattern_type='false_positive',
            description='LLM too lenient on vague responses',
            count=5,
            example_ids=['r1', 'r2', 'r3', 'r4', 'r5'],
        )
        assert pattern.pattern_type == 'false_positive'
        assert pattern.count == 5
        assert len(pattern.example_ids) == 5


class TestMisalignmentAnalysis:
    """Tests for MisalignmentAnalysis dataclass."""

    def test_create_analysis(self):
        """Test creating analysis result."""
        analysis = MisalignmentAnalysis(
            total_misaligned=10,
            false_positives=6,
            false_negatives=4,
            patterns=[
                MisalignmentPattern(
                    pattern_type='false_positive',
                    description='Too lenient',
                    count=6,
                    example_ids=['r1', 'r2', 'r3', 'r4', 'r5', 'r6'],
                )
            ],
            summary='LLM tends to be too lenient.',
            recommendations=['Be more strict on vague responses.'],
            metadata={'model': 'gpt-4o'},
        )
        assert analysis.total_misaligned == 10
        assert analysis.false_positives == 6
        assert analysis.false_negatives == 4
        assert len(analysis.patterns) == 1
        assert len(analysis.recommendations) == 1


class TestExtractMisalignedCases:
    """Tests for extract_misaligned_cases function."""

    def test_extract_from_dicts(self):
        """Test extracting misaligned cases from dicts."""
        results = [
            {
                'record_id': 'r1',
                'human_score': 1,
                'llm_score': 1,
                'query': 'Q1',
                'actual_output': 'A1',
            },  # Aligned
            {
                'record_id': 'r2',
                'human_score': 0,
                'llm_score': 1,
                'query': 'Q2',
                'actual_output': 'A2',
            },  # FP
            {
                'record_id': 'r3',
                'human_score': 1,
                'llm_score': 0,
                'query': 'Q3',
                'actual_output': 'A3',
            },  # FN
            {
                'record_id': 'r4',
                'human_score': 0,
                'llm_score': 0,
                'query': 'Q4',
                'actual_output': 'A4',
            },  # Aligned
        ]

        fp, fn = extract_misaligned_cases(results)

        assert len(fp) == 1
        assert len(fn) == 1
        assert fp[0].record_id == 'r2'
        assert fn[0].record_id == 'r3'

    def test_extract_with_aliases(self):
        """Test extracting with alias field names."""
        results = [
            {
                'id': 'r1',  # alias for record_id
                'judgment': 0,  # alias for human_score
                'score': 1,  # alias for llm_score
                'query': 'Q',
                'actual_output': 'A',
                'explanation': 'Reason',  # alias for llm_reasoning
            },
        ]

        fp, fn = extract_misaligned_cases(results)

        assert len(fp) == 1
        assert fp[0].record_id == 'r1'
        assert fp[0].llm_reasoning == 'Reason'

    def test_extract_no_misalignment(self):
        """Test with no misalignments."""
        results = [
            {'record_id': 'r1', 'human_score': 1, 'llm_score': 1, 'query': 'Q', 'actual_output': 'A'},
            {'record_id': 'r2', 'human_score': 0, 'llm_score': 0, 'query': 'Q', 'actual_output': 'A'},
        ]

        fp, fn = extract_misaligned_cases(results)

        assert len(fp) == 0
        assert len(fn) == 0

    def test_extract_empty_input(self):
        """Test with empty input."""
        fp, fn = extract_misaligned_cases([])
        assert fp == []
        assert fn == []

    def test_extract_from_objects(self):
        """Test extracting from objects with attributes."""

        class Result:
            def __init__(self, record_id, human_score, llm_score):
                self.record_id = record_id
                self.human_score = human_score
                self.llm_score = llm_score
                self.query = 'Q'
                self.actual_output = 'A'

        results = [
            Result('r1', 0, 1),  # FP
            Result('r2', 1, 0),  # FN
        ]

        fp, fn = extract_misaligned_cases(results)

        assert len(fp) == 1
        assert len(fn) == 1

    def test_extract_preserves_signals(self):
        """Test that signals field is preserved."""
        results = [
            {
                'record_id': 'r1',
                'human_score': 0,
                'llm_score': 1,
                'query': 'Q',
                'actual_output': 'A',
                'signals': {'confidence': 0.9, 'categories': ['factual']},
            },
        ]

        fp, fn = extract_misaligned_cases(results)

        assert fp[0].signals == {'confidence': 0.9, 'categories': ['factual']}


class TestMisalignmentAnalyzer:
    """Tests for MisalignmentAnalyzer class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        analyzer = MisalignmentAnalyzer()
        assert analyzer._max_examples == 10

    def test_init_custom(self):
        """Test initialization with custom params."""
        analyzer = MisalignmentAnalyzer(
            model_name='gpt-4o',
            llm_provider='openai',
            max_examples=20,
        )
        assert analyzer._model_name == 'gpt-4o'
        assert analyzer._llm_provider == 'openai'
        assert analyzer._max_examples == 20

    @pytest.mark.asyncio
    async def test_analyze_perfect_alignment(self):
        """Test analysis with no misalignments."""
        analyzer = MisalignmentAnalyzer(model_name='gpt-4o')
        results = [
            {'record_id': 'r1', 'human_score': 1, 'llm_score': 1, 'query': 'Q', 'actual_output': 'A'},
            {'record_id': 'r2', 'human_score': 0, 'llm_score': 0, 'query': 'Q', 'actual_output': 'A'},
        ]

        analysis = await analyzer.analyze(results, 'Test criteria')

        assert analysis.total_misaligned == 0
        assert analysis.false_positives == 0
        assert analysis.false_negatives == 0
        assert analysis.patterns == []
        assert 'Perfect alignment' in analysis.summary
        assert analysis.metadata.get('perfect_alignment') is True
