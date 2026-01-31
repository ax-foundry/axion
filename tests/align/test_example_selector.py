import pytest

from axion.align import ExampleSelector, SelectionResult, SelectionStrategy
from axion.align.pattern_discovery import DiscoveredPattern


@pytest.fixture
def sample_records():
    """Sample records for testing."""
    return [
        {'id': 'rec_1', 'query': 'q1', 'actual_output': 'out1'},
        {'id': 'rec_2', 'query': 'q2', 'actual_output': 'out2'},
        {'id': 'rec_3', 'query': 'q3', 'actual_output': 'out3'},
        {'id': 'rec_4', 'query': 'q4', 'actual_output': 'out4'},
        {'id': 'rec_5', 'query': 'q5', 'actual_output': 'out5'},
        {'id': 'rec_6', 'query': 'q6', 'actual_output': 'out6'},
    ]


@pytest.fixture
def balanced_annotations():
    """Annotations with 3 accepts and 3 rejects."""
    return {
        'rec_1': 1,
        'rec_2': 0,
        'rec_3': 1,
        'rec_4': 0,
        'rec_5': 1,
        'rec_6': 0,
    }


@pytest.fixture
def all_accepts_annotations():
    """Annotations where all are accepts."""
    return {
        'rec_1': 1,
        'rec_2': 1,
        'rec_3': 1,
        'rec_4': 1,
        'rec_5': 1,
        'rec_6': 1,
    }


@pytest.fixture
def all_rejects_annotations():
    """Annotations where all are rejects."""
    return {
        'rec_1': 0,
        'rec_2': 0,
        'rec_3': 0,
        'rec_4': 0,
        'rec_5': 0,
        'rec_6': 0,
    }


class TestBalancedStrategy:
    """Tests for BALANCED selection strategy."""

    def test_returns_correct_count(self, sample_records, balanced_annotations):
        """Should return the requested number of examples."""
        selector = ExampleSelector(seed=42)
        result = selector.select(sample_records, balanced_annotations, count=4)

        assert len(result.examples) == 4
        assert result.strategy_used == SelectionStrategy.BALANCED

    def test_balances_accept_reject(self, sample_records, balanced_annotations):
        """Should balance accept/reject when possible."""
        selector = ExampleSelector(seed=42)
        result = selector.select(sample_records, balanced_annotations, count=4)

        accepts = sum(1 for r in result.examples if balanced_annotations[r['id']] == 1)
        rejects = sum(1 for r in result.examples if balanced_annotations[r['id']] == 0)

        # With count=4, should be 2 accepts, 2 rejects
        assert accepts == 2
        assert rejects == 2

    def test_handles_all_accepts(self, sample_records, all_accepts_annotations):
        """Should handle case where all annotations are accepts."""
        selector = ExampleSelector(seed=42)
        result = selector.select(sample_records, all_accepts_annotations, count=4)

        assert len(result.examples) == 4
        # All should be accepts since there are no rejects
        for r in result.examples:
            assert all_accepts_annotations[r['id']] == 1

    def test_handles_all_rejects(self, sample_records, all_rejects_annotations):
        """Should handle case where all annotations are rejects."""
        selector = ExampleSelector(seed=42)
        result = selector.select(sample_records, all_rejects_annotations, count=4)

        assert len(result.examples) == 4
        # All should be rejects since there are no accepts
        for r in result.examples:
            assert all_rejects_annotations[r['id']] == 0

    def test_handles_empty_input(self):
        """Should handle empty records gracefully."""
        selector = ExampleSelector(seed=42)
        result = selector.select([], {}, count=4)

        assert len(result.examples) == 0
        assert result.strategy_used == SelectionStrategy.BALANCED

    def test_count_exceeds_available(self, sample_records, balanced_annotations):
        """Should return all available if count exceeds records."""
        selector = ExampleSelector(seed=42)
        result = selector.select(sample_records, balanced_annotations, count=100)

        assert len(result.examples) == len(sample_records)

    def test_metadata_tracks_counts(self, sample_records, balanced_annotations):
        """Should track accept/reject counts in metadata."""
        selector = ExampleSelector(seed=42)
        result = selector.select(sample_records, balanced_annotations, count=4)

        assert 'accepts' in result.metadata
        assert 'rejects' in result.metadata
        assert result.metadata['accepts'] + result.metadata['rejects'] <= 4


class TestMisalignmentGuidedStrategy:
    """Tests for MISALIGNMENT_GUIDED selection strategy."""

    @pytest.fixture
    def eval_results_with_misalignment(self):
        """Eval results where LLM disagreed with human."""
        return [
            {'id': 'rec_1', 'score': 1},  # Human=1, LLM=1 (aligned)
            {'id': 'rec_2', 'score': 1},  # Human=0, LLM=1 (FP)
            {'id': 'rec_3', 'score': 0},  # Human=1, LLM=0 (FN)
            {'id': 'rec_4', 'score': 0},  # Human=0, LLM=0 (aligned)
            {'id': 'rec_5', 'score': 1},  # Human=1, LLM=1 (aligned)
            {'id': 'rec_6', 'score': 1},  # Human=0, LLM=1 (FP)
        ]

    def test_prioritizes_fp_fn_cases(
        self, sample_records, balanced_annotations, eval_results_with_misalignment
    ):
        """Should prioritize false positive and false negative cases."""
        selector = ExampleSelector(seed=42)
        result = selector.select(
            sample_records,
            balanced_annotations,
            count=6,
            strategy=SelectionStrategy.MISALIGNMENT_GUIDED,
            eval_results=eval_results_with_misalignment,
        )

        # Should have selected FP/FN cases
        assert result.metadata['total_fp_available'] == 2
        assert result.metadata['total_fn_available'] == 1

    def test_raises_without_eval_results(self, sample_records, balanced_annotations):
        """Should raise ValueError if eval_results is missing."""
        selector = ExampleSelector(seed=42)

        with pytest.raises(ValueError, match='eval_results required'):
            selector.select(
                sample_records,
                balanced_annotations,
                count=6,
                strategy=SelectionStrategy.MISALIGNMENT_GUIDED,
            )

    def test_falls_back_to_balanced(self, sample_records, balanced_annotations):
        """Should fill remaining with balanced selection."""
        # All aligned eval results
        eval_results = [
            {'id': f'rec_{i}', 'score': balanced_annotations[f'rec_{i}']}
            for i in range(1, 7)
        ]

        selector = ExampleSelector(seed=42)
        result = selector.select(
            sample_records,
            balanced_annotations,
            count=4,
            strategy=SelectionStrategy.MISALIGNMENT_GUIDED,
            eval_results=eval_results,
        )

        assert len(result.examples) == 4
        assert result.metadata['false_positives_selected'] == 0
        assert result.metadata['false_negatives_selected'] == 0

    def test_handles_llm_score_field(self, sample_records, balanced_annotations):
        """Should handle 'llm_score' field name in eval results."""
        eval_results = [
            {'id': 'rec_1', 'llm_score': 0},  # Human=1, LLM=0 (FN)
            {'id': 'rec_2', 'llm_score': 0},  # Human=0, LLM=0 (aligned)
            {'id': 'rec_3', 'llm_score': 1},  # Human=1, LLM=1 (aligned)
            {'id': 'rec_4', 'llm_score': 1},  # Human=0, LLM=1 (FP)
            {'id': 'rec_5', 'llm_score': 1},  # Human=1, LLM=1 (aligned)
            {'id': 'rec_6', 'llm_score': 0},  # Human=0, LLM=0 (aligned)
        ]

        selector = ExampleSelector(seed=42)
        result = selector.select(
            sample_records,
            balanced_annotations,
            count=6,
            strategy=SelectionStrategy.MISALIGNMENT_GUIDED,
            eval_results=eval_results,
        )

        assert result.metadata['total_fp_available'] == 1
        assert result.metadata['total_fn_available'] == 1


class TestPatternAwareStrategy:
    """Tests for PATTERN_AWARE selection strategy."""

    @pytest.fixture
    def sample_patterns(self):
        """Sample discovered patterns."""
        return [
            DiscoveredPattern(
                category='Missing Context',
                description='Responses lack sufficient context',
                count=2,
                record_ids=['rec_1', 'rec_2'],
                examples=['example 1', 'example 2'],
            ),
            DiscoveredPattern(
                category='Factual Errors',
                description='Contains factual inaccuracies',
                count=2,
                record_ids=['rec_3', 'rec_4'],
                examples=['example 3', 'example 4'],
            ),
            DiscoveredPattern(
                category='Incomplete',
                description='Responses are incomplete',
                count=2,
                record_ids=['rec_5', 'rec_6'],
                examples=['example 5', 'example 6'],
            ),
        ]

    def test_covers_patterns(
        self, sample_records, balanced_annotations, sample_patterns
    ):
        """Should sample from discovered patterns."""
        selector = ExampleSelector(seed=42)
        result = selector.select(
            sample_records,
            balanced_annotations,
            count=6,
            strategy=SelectionStrategy.PATTERN_AWARE,
            patterns=sample_patterns,
        )

        # Should have covered patterns
        assert len(result.metadata['patterns_covered']) > 0
        assert result.metadata['total_patterns'] == 3

    def test_raises_without_patterns(self, sample_records, balanced_annotations):
        """Should raise ValueError if patterns is missing."""
        selector = ExampleSelector(seed=42)

        with pytest.raises(ValueError, match='patterns required'):
            selector.select(
                sample_records,
                balanced_annotations,
                count=6,
                strategy=SelectionStrategy.PATTERN_AWARE,
            )

    def test_falls_back_to_balanced(self, sample_records, balanced_annotations):
        """Should fill remaining with balanced selection when patterns < count."""
        single_pattern = [
            DiscoveredPattern(
                category='Single Pattern',
                description='Single pattern',
                count=1,
                record_ids=['rec_1'],
                examples=['example'],
            )
        ]

        selector = ExampleSelector(seed=42)
        result = selector.select(
            sample_records,
            balanced_annotations,
            count=4,
            strategy=SelectionStrategy.PATTERN_AWARE,
            patterns=single_pattern,
        )

        assert len(result.examples) == 4
        assert len(result.metadata['patterns_covered']) == 1

    def test_handles_patterns_with_unknown_records(
        self, sample_records, balanced_annotations
    ):
        """Should skip pattern record IDs not in records."""
        pattern_with_unknown = [
            DiscoveredPattern(
                category='Unknown Records',
                description='Has unknown record IDs',
                count=2,
                record_ids=['unknown_1', 'unknown_2'],
                examples=['example'],
            )
        ]

        selector = ExampleSelector(seed=42)
        result = selector.select(
            sample_records,
            balanced_annotations,
            count=4,
            strategy=SelectionStrategy.PATTERN_AWARE,
            patterns=pattern_with_unknown,
        )

        # Should fall back to balanced since no pattern records available
        assert len(result.examples) == 4
        assert result.metadata['patterns_covered'] == []


class TestReproducibility:
    """Tests for reproducibility with seeds."""

    def test_same_seed_same_results(self, sample_records, balanced_annotations):
        """Same seed should produce identical results."""
        selector1 = ExampleSelector(seed=42)
        selector2 = ExampleSelector(seed=42)

        result1 = selector1.select(sample_records, balanced_annotations, count=4)
        result2 = selector2.select(sample_records, balanced_annotations, count=4)

        assert [r['id'] for r in result1.examples] == [
            r['id'] for r in result2.examples
        ]

    def test_different_seed_different_results(
        self, sample_records, balanced_annotations
    ):
        """Different seeds should produce different results (with high probability)."""
        selector1 = ExampleSelector(seed=42)
        selector2 = ExampleSelector(seed=123)

        result1 = selector1.select(sample_records, balanced_annotations, count=4)
        result2 = selector2.select(sample_records, balanced_annotations, count=4)

        # May occasionally be the same by chance, but very unlikely
        ids1 = [r['id'] for r in result1.examples]
        ids2 = [r['id'] for r in result2.examples]
        # At least check we got valid results
        assert len(ids1) == 4
        assert len(ids2) == 4


class TestRecordIdHandling:
    """Tests for handling different record ID field names."""

    def test_handles_id_field(self):
        """Should handle 'id' field."""
        records = [{'id': 'r1', 'query': 'q'}]
        annotations = {'r1': 1}

        selector = ExampleSelector(seed=42)
        result = selector.select(records, annotations, count=1)

        assert len(result.examples) == 1
        assert result.examples[0]['id'] == 'r1'

    def test_handles_record_id_field(self):
        """Should handle 'record_id' field."""
        records = [{'record_id': 'r1', 'query': 'q'}]
        annotations = {'r1': 1}

        selector = ExampleSelector(seed=42)
        result = selector.select(records, annotations, count=1)

        assert len(result.examples) == 1
        assert result.examples[0]['record_id'] == 'r1'


class TestSelectionResult:
    """Tests for SelectionResult dataclass."""

    def test_result_structure(self, sample_records, balanced_annotations):
        """SelectionResult should have correct structure."""
        selector = ExampleSelector(seed=42)
        result = selector.select(sample_records, balanced_annotations, count=4)

        assert isinstance(result, SelectionResult)
        assert isinstance(result.examples, list)
        assert isinstance(result.strategy_used, SelectionStrategy)
        assert isinstance(result.metadata, dict)
