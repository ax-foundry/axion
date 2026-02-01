import pytest

from axion.caliber.example_selector import (
    ExampleSelector,
    SelectionResult,
    SelectionStrategy,
)
from axion.caliber.pattern_discovery import DiscoveredPattern


class TestExampleSelector:
    """Tests for ExampleSelector."""

    @pytest.fixture
    def sample_records(self):
        """Create sample records."""
        return [
            {'id': 'r1', 'query': 'Q1', 'actual_output': 'A1'},
            {'id': 'r2', 'query': 'Q2', 'actual_output': 'A2'},
            {'id': 'r3', 'query': 'Q3', 'actual_output': 'A3'},
            {'id': 'r4', 'query': 'Q4', 'actual_output': 'A4'},
            {'id': 'r5', 'query': 'Q5', 'actual_output': 'A5'},
            {'id': 'r6', 'query': 'Q6', 'actual_output': 'A6'},
        ]

    @pytest.fixture
    def balanced_annotations(self):
        """Create balanced annotations."""
        return {
            'r1': 1,
            'r2': 1,
            'r3': 1,
            'r4': 0,
            'r5': 0,
            'r6': 0,
        }

    @pytest.fixture
    def selector(self):
        """Create selector with fixed seed."""
        return ExampleSelector(seed=42)

    def test_balanced_selection(self, selector, sample_records, balanced_annotations):
        """Test balanced selection strategy."""
        result = selector.select(
            sample_records,
            balanced_annotations,
            count=4,
            strategy=SelectionStrategy.BALANCED,
        )

        assert isinstance(result, SelectionResult)
        assert len(result.examples) == 4
        assert result.strategy_used == SelectionStrategy.BALANCED
        assert 'accepts' in result.metadata
        assert 'rejects' in result.metadata

    def test_balanced_selection_ratio(
        self, selector, sample_records, balanced_annotations
    ):
        """Test that balanced selection maintains 50/50 ratio."""
        result = selector.select(
            sample_records,
            balanced_annotations,
            count=4,
            strategy=SelectionStrategy.BALANCED,
        )

        accepts = sum(
            1 for ex in result.examples if balanced_annotations[ex['id']] == 1
        )
        rejects = sum(
            1 for ex in result.examples if balanced_annotations[ex['id']] == 0
        )

        # Should be roughly balanced (2 each for count=4)
        assert accepts == 2
        assert rejects == 2

    def test_misalignment_guided_selection(
        self, selector, sample_records, balanced_annotations
    ):
        """Test misalignment-guided selection strategy."""
        eval_results = [
            {'id': 'r1', 'llm_score': 0, 'score': 0},  # FN: llm=0, human=1
            {'id': 'r2', 'llm_score': 1, 'score': 1},  # aligned
            {'id': 'r3', 'llm_score': 1, 'score': 1},  # aligned
            {'id': 'r4', 'llm_score': 1, 'score': 1},  # FP: llm=1, human=0
            {'id': 'r5', 'llm_score': 0, 'score': 0},  # aligned
            {'id': 'r6', 'llm_score': 0, 'score': 0},  # aligned
        ]

        result = selector.select(
            sample_records,
            balanced_annotations,
            count=6,
            strategy=SelectionStrategy.MISALIGNMENT_GUIDED,
            eval_results=eval_results,
        )

        assert result.strategy_used == SelectionStrategy.MISALIGNMENT_GUIDED
        assert 'false_positives_selected' in result.metadata
        assert 'false_negatives_selected' in result.metadata

    def test_misalignment_guided_requires_eval_results(
        self, selector, sample_records, balanced_annotations
    ):
        """Test that misalignment-guided strategy requires eval_results."""
        with pytest.raises(ValueError):
            selector.select(
                sample_records,
                balanced_annotations,
                count=4,
                strategy=SelectionStrategy.MISALIGNMENT_GUIDED,
            )

    def test_pattern_aware_selection(
        self, selector, sample_records, balanced_annotations
    ):
        """Test pattern-aware selection strategy."""
        patterns = [
            DiscoveredPattern(
                category='Pattern A',
                description='First pattern',
                count=2,
                record_ids=['r1', 'r2'],
            ),
            DiscoveredPattern(
                category='Pattern B',
                description='Second pattern',
                count=2,
                record_ids=['r3', 'r4'],
            ),
        ]

        result = selector.select(
            sample_records,
            balanced_annotations,
            count=4,
            strategy=SelectionStrategy.PATTERN_AWARE,
            patterns=patterns,
        )

        assert result.strategy_used == SelectionStrategy.PATTERN_AWARE
        assert 'patterns_covered' in result.metadata
        assert 'total_patterns' in result.metadata

    def test_pattern_aware_requires_patterns(
        self, selector, sample_records, balanced_annotations
    ):
        """Test that pattern-aware strategy requires patterns."""
        with pytest.raises(ValueError):
            selector.select(
                sample_records,
                balanced_annotations,
                count=4,
                strategy=SelectionStrategy.PATTERN_AWARE,
            )

    def test_reproducibility_with_seed(self, sample_records, balanced_annotations):
        """Test that same seed produces same results."""
        selector1 = ExampleSelector(seed=42)
        selector2 = ExampleSelector(seed=42)

        result1 = selector1.select(sample_records, balanced_annotations, count=4)
        result2 = selector2.select(sample_records, balanced_annotations, count=4)

        # Same seed should give same examples
        ids1 = {ex['id'] for ex in result1.examples}
        ids2 = {ex['id'] for ex in result2.examples}
        assert ids1 == ids2

    def test_handles_record_id_alias(self, selector, balanced_annotations):
        """Test handling records with record_id instead of id."""
        records = [
            {'record_id': 'r1', 'query': 'Q1'},
            {'record_id': 'r2', 'query': 'Q2'},
            {'record_id': 'r3', 'query': 'Q3'},
            {'record_id': 'r4', 'query': 'Q4'},
        ]

        result = selector.select(
            records,
            balanced_annotations,
            count=2,
            strategy=SelectionStrategy.BALANCED,
        )

        assert len(result.examples) == 2

    def test_handles_count_greater_than_available(
        self, selector, sample_records, balanced_annotations
    ):
        """Test handling when requested count exceeds available records."""
        result = selector.select(
            sample_records,
            balanced_annotations,
            count=100,
            strategy=SelectionStrategy.BALANCED,
        )

        # Should return all available records
        assert len(result.examples) <= len(sample_records)
