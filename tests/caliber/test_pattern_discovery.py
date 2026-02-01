"""Tests for pattern discovery module."""

import pytest

from axion.caliber.pattern_discovery import (
    AnnotatedItem,
    AnnotationNote,
    ClusteringInput,
    ClusteringMethod,
    ClusteringOutput,
    DiscoveredPattern,
    PatternCategory,
    PatternDiscovery,
    PatternDiscoveryResult,
)


class TestAnnotatedItem:
    """Tests for AnnotatedItem dataclass."""

    def test_create_minimal(self):
        """Test creating item with minimal fields."""
        item = AnnotatedItem(record_id='r1', score=1)
        assert item.record_id == 'r1'
        assert item.score == 1
        assert item.notes is None
        assert item.query is None
        assert item.actual_output is None

    def test_create_full(self):
        """Test creating item with all fields."""
        item = AnnotatedItem(
            record_id='r1',
            score=0,
            notes='Missing context',
            timestamp='2024-01-01T00:00:00Z',
            query='What is Python?',
            actual_output='A snake.',
        )
        assert item.record_id == 'r1'
        assert item.score == 0
        assert item.notes == 'Missing context'
        assert item.query == 'What is Python?'


class TestDiscoveredPattern:
    """Tests for DiscoveredPattern dataclass."""

    def test_create_pattern(self):
        """Test creating a discovered pattern."""
        pattern = DiscoveredPattern(
            category='Missing Context',
            description='Responses lacking background',
            count=5,
            record_ids=['r1', 'r2', 'r3', 'r4', 'r5'],
            examples=['Example 1', 'Example 2'],
            confidence=0.85,
        )
        assert pattern.category == 'Missing Context'
        assert pattern.count == 5
        assert len(pattern.record_ids) == 5
        assert pattern.confidence == 0.85


class TestPatternDiscoveryResult:
    """Tests for PatternDiscoveryResult dataclass."""

    def test_create_result(self):
        """Test creating a pattern discovery result."""
        patterns = [
            DiscoveredPattern(
                category='Pattern 1',
                description='Desc 1',
                count=3,
                record_ids=['r1', 'r2', 'r3'],
            ),
            DiscoveredPattern(
                category='Pattern 2',
                description='Desc 2',
                count=2,
                record_ids=['r4', 'r5'],
            ),
        ]
        result = PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=['r6'],
            total_analyzed=6,
            method=ClusteringMethod.LLM,
            metadata={'model': 'gpt-4o'},
        )
        assert len(result.patterns) == 2
        assert result.uncategorized == ['r6']
        assert result.total_analyzed == 6
        assert result.method == ClusteringMethod.LLM


class TestClusteringMethod:
    """Tests for ClusteringMethod enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ClusteringMethod.LLM.value == 'llm'
        assert ClusteringMethod.BERTOPIC.value == 'bertopic'
        assert ClusteringMethod.HYBRID.value == 'hybrid'


class TestPydanticModels:
    """Tests for Pydantic input/output models."""

    def test_annotation_note(self):
        """Test AnnotationNote model."""
        note = AnnotationNote(record_id='r1', notes='Good response')
        assert note.record_id == 'r1'
        assert note.notes == 'Good response'

    def test_clustering_input(self):
        """Test ClusteringInput model."""
        input_data = ClusteringInput(
            annotations=[
                AnnotationNote(record_id='r1', notes='Note 1'),
                AnnotationNote(record_id='r2', notes='Note 2'),
            ]
        )
        assert len(input_data.annotations) == 2

    def test_pattern_category(self):
        """Test PatternCategory model."""
        category = PatternCategory(
            category='Missing Info',
            record_ids=['r1', 'r2'],
            description='Responses missing information',
        )
        assert category.category == 'Missing Info'
        assert len(category.record_ids) == 2

    def test_clustering_output(self):
        """Test ClusteringOutput model."""
        output = ClusteringOutput(
            patterns=[
                PatternCategory(
                    category='Pattern 1',
                    record_ids=['r1'],
                    description='Desc',
                )
            ],
            uncategorized=['r2'],
        )
        assert len(output.patterns) == 1
        assert output.uncategorized == ['r2']


class TestPatternDiscovery:
    """Tests for PatternDiscovery class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        discovery = PatternDiscovery()
        assert discovery._max_notes == 50
        assert discovery._min_category_size == 2

    def test_init_custom(self):
        """Test initialization with custom params."""
        discovery = PatternDiscovery(
            model_name='gpt-4o',
            llm_provider='openai',
            max_notes=100,
            min_category_size=3,
        )
        assert discovery._model_name == 'gpt-4o'
        assert discovery._llm_provider == 'openai'
        assert discovery._max_notes == 100
        assert discovery._min_category_size == 3

    def test_normalize_annotations_dict(self):
        """Test normalizing dict annotations."""
        discovery = PatternDiscovery()
        annotations = {
            'r1': {'record_id': 'r1', 'score': 1, 'notes': 'Good'},
            'r2': {'score': 0, 'notes': 'Bad'},
        }
        result = discovery._normalize_annotations(annotations)

        assert 'r1' in result
        assert 'r2' in result
        assert isinstance(result['r1'], AnnotatedItem)
        assert result['r1'].notes == 'Good'
        assert result['r2'].record_id == 'r2'  # Uses key as fallback

    def test_normalize_annotations_items(self):
        """Test normalizing AnnotatedItem annotations."""
        discovery = PatternDiscovery()
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=1, notes='Good'),
        }
        result = discovery._normalize_annotations(annotations)

        assert result['r1'].notes == 'Good'

    def test_normalize_annotations_invalid_type(self):
        """Test normalizing invalid annotation type."""
        discovery = PatternDiscovery()
        annotations = {'r1': 'invalid'}

        with pytest.raises(TypeError, match='Invalid annotation type'):
            discovery._normalize_annotations(annotations)

    def test_format_topic_name(self):
        """Test formatting BERTopic topic name."""
        discovery = PatternDiscovery()

        # Normal case
        topic_words = [('missing', 0.5), ('context', 0.4), ('info', 0.3), ('data', 0.2)]
        name = discovery._format_topic_name(topic_words)
        assert name == 'Missing / Context / Info'

        # Empty case
        assert discovery._format_topic_name([]) == 'Unknown Pattern'

    def test_constants(self):
        """Test class constants."""
        assert PatternDiscovery.BERTOPIC_MIN_DOCUMENTS == 5
        assert PatternDiscovery.MAX_EXAMPLES_PER_PATTERN == 3
        assert PatternDiscovery.EXAMPLE_PREVIEW_CHARS == 100


class TestPatternDiscoveryEmptyInput:
    """Tests for pattern discovery with empty/edge case inputs."""

    @pytest.mark.asyncio
    async def test_discover_no_notes(self):
        """Test discovery with no annotations having notes."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=1, notes=None),
            'r2': AnnotatedItem(record_id='r2', score=0, notes=None),
        }

        # This should return early without calling LLM
        result = await discovery._cluster_with_llm(
            discovery._normalize_annotations(annotations)
        )

        assert result.patterns == []
        assert set(result.uncategorized) == {'r1', 'r2'}
        assert result.metadata.get('error') == 'No items with notes to cluster'
