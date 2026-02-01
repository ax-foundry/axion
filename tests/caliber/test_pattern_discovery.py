"""Tests for pattern discovery module."""

import pytest

from axion.caliber.pattern_discovery import (
    AnnotatedItem,
    AnnotationNote,
    ClusteringInput,
    ClusteringMethod,
    ClusteringOutput,
    DiscoveredPattern,
    LabelInput,
    LabelOutput,
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


class TestLabelModels:
    """Tests for LabelInput and LabelOutput Pydantic models."""

    def test_label_input(self):
        """Test LabelInput model."""
        input_data = LabelInput(
            examples=['Missing context in response', 'Lacks background info'],
            keywords='missing, context, background, info',
        )
        assert len(input_data.examples) == 2
        assert 'missing' in input_data.keywords

    def test_label_output(self):
        """Test LabelOutput model."""
        output = LabelOutput(category_name='Missing Context')
        assert output.category_name == 'Missing Context'

    def test_label_output_alias_category(self):
        """Test LabelOutput accepts 'category' alias."""
        output = LabelOutput(category='Missing Context')
        assert output.category_name == 'Missing Context'

    def test_label_output_alias_name(self):
        """Test LabelOutput accepts 'name' alias."""
        output = LabelOutput(name='Missing Context')
        assert output.category_name == 'Missing Context'

    def test_label_output_alias_label(self):
        """Test LabelOutput accepts 'label' alias."""
        output = LabelOutput(label='Missing Context')
        assert output.category_name == 'Missing Context'


class TestBERTopicClustering:
    """Tests for BERTopic clustering method."""

    @pytest.mark.asyncio
    async def test_bertopic_too_few_documents(self):
        """Test BERTopic with too few documents returns early."""
        discovery = PatternDiscovery()

        # Only 3 items - below BERTOPIC_MIN_DOCUMENTS (5)
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes='Missing context'),
            'r2': AnnotatedItem(record_id='r2', score=0, notes='Lacks detail'),
            'r3': AnnotatedItem(record_id='r3', score=1, notes='Good response'),
        }

        items = discovery._normalize_annotations(annotations)
        result = await discovery._cluster_with_bertopic(items)

        assert result.patterns == []
        assert set(result.uncategorized) == {'r1', 'r2', 'r3'}
        assert result.method == ClusteringMethod.BERTOPIC
        assert 'Too few documents' in result.metadata.get('error', '')

    @pytest.mark.asyncio
    async def test_bertopic_no_notes(self):
        """Test BERTopic with no notes returns early."""
        discovery = PatternDiscovery()

        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes=None),
            'r2': AnnotatedItem(record_id='r2', score=1, notes=None),
        }

        items = discovery._normalize_annotations(annotations)
        result = await discovery._cluster_with_bertopic(items)

        assert result.patterns == []
        assert 'Too few documents' in result.metadata.get('error', '')

    def test_bertopic_embedding_model_config(self):
        """Test BERTopic embedding model can be configured."""
        discovery = PatternDiscovery(bertopic_embedding_model='paraphrase-MiniLM-L6-v2')
        assert discovery._bertopic_embedding_model == 'paraphrase-MiniLM-L6-v2'


class TestHybridClustering:
    """Tests for Hybrid clustering method."""

    @pytest.mark.asyncio
    async def test_hybrid_too_few_documents(self):
        """Test Hybrid with too few documents returns early with correct method."""
        discovery = PatternDiscovery(model_name='gpt-4o')

        # Only 3 items - below BERTOPIC_MIN_DOCUMENTS
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes='Missing context'),
            'r2': AnnotatedItem(record_id='r2', score=0, notes='Lacks detail'),
            'r3': AnnotatedItem(record_id='r3', score=1, notes='Good response'),
        }

        items = discovery._normalize_annotations(annotations)
        result = await discovery._cluster_hybrid(items)

        # Should still return HYBRID method even though it fell back
        assert result.method == ClusteringMethod.HYBRID
        assert result.patterns == []

    def test_label_handler_lazy_init(self):
        """Test that label handler is lazily initialized."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        assert discovery._label_handler is None
        # Handler created on first access (not testing actual call to avoid API)


class TestPatternDiscoveryMethodDispatch:
    """Tests for method dispatch in discover()."""

    @pytest.mark.asyncio
    async def test_discover_dispatches_to_llm(self):
        """Test discover() dispatches to LLM method."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes=None),
        }

        # With no notes, LLM method returns early
        result = await discovery.discover(annotations, method=ClusteringMethod.LLM)
        assert result.method == ClusteringMethod.LLM

    @pytest.mark.asyncio
    async def test_discover_dispatches_to_bertopic(self):
        """Test discover() dispatches to BERTopic method."""
        discovery = PatternDiscovery()
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes='Note 1'),
            'r2': AnnotatedItem(record_id='r2', score=0, notes='Note 2'),
        }

        # Too few docs, returns early
        result = await discovery.discover(annotations, method=ClusteringMethod.BERTOPIC)
        assert result.method == ClusteringMethod.BERTOPIC

    @pytest.mark.asyncio
    async def test_discover_dispatches_to_hybrid(self):
        """Test discover() dispatches to Hybrid method."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes='Note 1'),
            'r2': AnnotatedItem(record_id='r2', score=0, notes='Note 2'),
        }

        # Too few docs, returns early but with HYBRID method
        result = await discovery.discover(annotations, method=ClusteringMethod.HYBRID)
        assert result.method == ClusteringMethod.HYBRID

    @pytest.mark.asyncio
    async def test_discover_invalid_method(self):
        """Test discover() raises on invalid method."""
        discovery = PatternDiscovery()
        annotations = {'r1': AnnotatedItem(record_id='r1', score=0, notes='Note')}

        with pytest.raises(ValueError, match='Unknown clustering method'):
            await discovery.discover(annotations, method='invalid')
