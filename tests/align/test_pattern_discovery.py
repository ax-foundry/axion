"""Tests for pattern discovery module."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axion.align.pattern_discovery import (
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

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_annotations():
    """Sample annotations for testing."""
    return {
        'rec_1': AnnotatedItem(
            record_id='rec_1', score=0, notes='Missing context about user'
        ),
        'rec_2': AnnotatedItem(record_id='rec_2', score=0, notes='No context provided'),
        'rec_3': AnnotatedItem(
            record_id='rec_3', score=0, notes='Incorrect factual claim'
        ),
        'rec_4': AnnotatedItem(record_id='rec_4', score=0, notes='Wrong facts stated'),
        'rec_5': AnnotatedItem(record_id='rec_5', score=0, notes='Needs more detail'),
    }


@pytest.fixture
def sample_dict_annotations():
    """Sample annotations as dictionaries."""
    return {
        'rec_1': {'record_id': 'rec_1', 'score': 0, 'notes': 'Missing context'},
        'rec_2': {'record_id': 'rec_2', 'score': 0, 'notes': 'Lacks detail'},
        'rec_3': {'score': 1, 'notes': 'Good response'},
    }


@pytest.fixture
def mock_clustering_output():
    """Mock output from clustering handler."""
    return ClusteringOutput(
        patterns=[
            PatternCategory(
                category='Missing Context',
                record_ids=['rec_1', 'rec_2'],
                description='Responses lack necessary context',
            ),
            PatternCategory(
                category='Factual Errors',
                record_ids=['rec_3', 'rec_4'],
                description='Responses contain incorrect facts',
            ),
        ],
        uncategorized=['rec_5'],
    )


# ============================================================================
# Unit Tests for _normalize_annotations
# ============================================================================


class TestNormalizeAnnotations:
    """Tests for annotation normalization."""

    def test_normalize_annotated_items(self, sample_annotations):
        """Test normalizing AnnotatedItem instances."""
        discovery = PatternDiscovery()
        result = discovery._normalize_annotations(sample_annotations)

        assert len(result) == 5
        assert all(isinstance(v, AnnotatedItem) for v in result.values())
        assert result['rec_1'].notes == 'Missing context about user'

    def test_normalize_dict_annotations(self, sample_dict_annotations):
        """Test normalizing dict annotations."""
        discovery = PatternDiscovery()
        result = discovery._normalize_annotations(sample_dict_annotations)

        assert len(result) == 3
        assert all(isinstance(v, AnnotatedItem) for v in result.values())
        assert result['rec_1'].notes == 'Missing context'
        assert result['rec_3'].record_id == 'rec_3'

    def test_normalize_mixed_annotations(self):
        """Test normalizing mixed annotation types."""
        annotations = {
            'rec_1': AnnotatedItem(record_id='rec_1', score=0, notes='Note 1'),
            'rec_2': {'score': 1, 'notes': 'Note 2'},
        }
        discovery = PatternDiscovery()
        result = discovery._normalize_annotations(annotations)

        assert len(result) == 2
        assert isinstance(result['rec_1'], AnnotatedItem)
        assert isinstance(result['rec_2'], AnnotatedItem)

    def test_normalize_invalid_type_raises(self):
        """Test that invalid annotation types raise TypeError."""
        annotations = {'rec_1': 'invalid_string'}
        discovery = PatternDiscovery()

        with pytest.raises(TypeError, match='Invalid annotation type'):
            discovery._normalize_annotations(annotations)


# ============================================================================
# Unit Tests for _format_topic_name
# ============================================================================


class TestFormatTopicName:
    """Tests for topic name formatting."""

    def test_format_topic_name_basic(self):
        """Test basic topic name formatting."""
        discovery = PatternDiscovery()
        topic_words = [('context', 0.5), ('missing', 0.4), ('information', 0.3)]
        result = discovery._format_topic_name(topic_words)

        assert result == 'Context / Missing / Information'

    def test_format_topic_name_empty(self):
        """Test formatting empty topic words."""
        discovery = PatternDiscovery()
        result = discovery._format_topic_name([])

        assert result == 'Unknown Pattern'

    def test_format_topic_name_single_word(self):
        """Test formatting single topic word."""
        discovery = PatternDiscovery()
        topic_words = [('error', 0.8)]
        result = discovery._format_topic_name(topic_words)

        assert result == 'Error'


# ============================================================================
# Integration Tests with Mocked LLM
# ============================================================================


class TestLLMClustering:
    """Tests for LLM-based clustering."""

    @pytest.mark.asyncio
    async def test_cluster_with_llm_returns_patterns(
        self, sample_annotations, mock_clustering_output
    ):
        """Test that LLM clustering returns expected patterns."""
        discovery = PatternDiscovery()

        with patch.object(discovery, '_get_clustering_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.execute = AsyncMock(return_value=mock_clustering_output)
            mock_handler.model_name = 'gpt-4o'
            mock_handler.llm_provider = 'openai'
            mock_get_handler.return_value = mock_handler

            result = await discovery.discover(
                sample_annotations, method=ClusteringMethod.LLM
            )

        assert isinstance(result, PatternDiscoveryResult)
        assert result.method == ClusteringMethod.LLM
        assert len(result.patterns) == 2
        assert result.patterns[0].category == 'Missing Context'
        assert result.uncategorized == ['rec_5']
        assert result.total_analyzed == 5

    @pytest.mark.asyncio
    async def test_cluster_filters_small_categories(self, sample_annotations):
        """Test that small categories are filtered based on min_category_size."""
        discovery = PatternDiscovery(min_category_size=3)

        small_output = ClusteringOutput(
            patterns=[
                PatternCategory(
                    category='Small Group',
                    record_ids=['rec_1', 'rec_2'],
                    description='Only 2 items',
                ),
                PatternCategory(
                    category='Large Group',
                    record_ids=['rec_3', 'rec_4', 'rec_5'],
                    description='3 items',
                ),
            ],
            uncategorized=[],
        )

        with patch.object(discovery, '_get_clustering_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.execute = AsyncMock(return_value=small_output)
            mock_handler.model_name = 'gpt-4o'
            mock_handler.llm_provider = 'openai'
            mock_get_handler.return_value = mock_handler

            result = await discovery.discover(sample_annotations)

        # Only the large group should remain
        assert len(result.patterns) == 1
        assert result.patterns[0].category == 'Large Group'


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_annotations(self):
        """Test with empty annotations."""
        discovery = PatternDiscovery()

        with patch.object(discovery, '_get_clustering_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.execute = AsyncMock(
                return_value=ClusteringOutput(patterns=[], uncategorized=[])
            )
            mock_handler.model_name = 'gpt-4o'
            mock_handler.llm_provider = 'openai'
            mock_get_handler.return_value = mock_handler

            result = await discovery.discover({})

        assert result.patterns == []
        assert result.total_analyzed == 0

    @pytest.mark.asyncio
    async def test_no_items_with_notes(self):
        """Test when no items have notes."""
        annotations = {
            'rec_1': AnnotatedItem(record_id='rec_1', score=0, notes=None),
            'rec_2': AnnotatedItem(record_id='rec_2', score=0, notes=''),
        }
        discovery = PatternDiscovery()

        result = await discovery.discover(annotations)

        assert result.patterns == []
        assert 'rec_1' in result.uncategorized
        assert result.metadata.get('error') == 'No items with notes to cluster'

    @pytest.mark.asyncio
    async def test_single_category_result(self, sample_annotations):
        """Test when all items belong to one category."""
        single_output = ClusteringOutput(
            patterns=[
                PatternCategory(
                    category='All Items',
                    record_ids=['rec_1', 'rec_2', 'rec_3', 'rec_4', 'rec_5'],
                    description='All items grouped together',
                ),
            ],
            uncategorized=[],
        )

        discovery = PatternDiscovery()

        with patch.object(discovery, '_get_clustering_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.execute = AsyncMock(return_value=single_output)
            mock_handler.model_name = 'gpt-4o'
            mock_handler.llm_provider = 'openai'
            mock_get_handler.return_value = mock_handler

            result = await discovery.discover(sample_annotations)

        assert len(result.patterns) == 1
        assert result.patterns[0].count == 5


# ============================================================================
# Pydantic Model Tests
# ============================================================================


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_annotation_note_validation(self):
        """Test AnnotationNote validation."""
        note = AnnotationNote(record_id='rec_1', notes='Test note')
        assert note.record_id == 'rec_1'
        assert note.notes == 'Test note'

    def test_clustering_input_validation(self):
        """Test ClusteringInput validation."""
        input_data = ClusteringInput(
            annotations=[
                AnnotationNote(record_id='rec_1', notes='Note 1'),
                AnnotationNote(record_id='rec_2', notes='Note 2'),
            ]
        )
        assert len(input_data.annotations) == 2

    def test_clustering_output_validation(self):
        """Test ClusteringOutput validation."""
        output = ClusteringOutput(
            patterns=[
                PatternCategory(
                    category='Test',
                    record_ids=['rec_1'],
                    description='Test pattern',
                )
            ],
            uncategorized=['rec_2'],
        )
        assert len(output.patterns) == 1
        assert output.uncategorized == ['rec_2']

    def test_label_input_output(self):
        """Test LabelInput and LabelOutput."""
        input_data = LabelInput(examples=['ex1', 'ex2'], keywords='test keywords')
        assert len(input_data.examples) == 2

        output = LabelOutput(category_name='Test Category')
        assert output.category_name == 'Test Category'


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestDataclasses:
    """Tests for dataclass structures."""

    def test_annotated_item_creation(self):
        """Test AnnotatedItem creation."""
        item = AnnotatedItem(
            record_id='rec_1',
            score=0,
            notes='Test notes',
            timestamp='2024-01-01',
            query='test query',
            actual_output='test output',
        )
        assert item.record_id == 'rec_1'
        assert item.score == 0
        assert item.notes == 'Test notes'

    def test_discovered_pattern_defaults(self):
        """Test DiscoveredPattern with default values."""
        pattern = DiscoveredPattern(
            category='Test',
            description='Test desc',
            count=5,
            record_ids=['rec_1', 'rec_2'],
        )
        assert pattern.examples == []
        assert pattern.confidence is None

    def test_pattern_discovery_result(self):
        """Test PatternDiscoveryResult structure."""
        result = PatternDiscoveryResult(
            patterns=[],
            uncategorized=['rec_1'],
            total_analyzed=5,
            method=ClusteringMethod.LLM,
        )
        assert result.total_analyzed == 5
        assert result.method == ClusteringMethod.LLM
        assert result.metadata == {}


# ============================================================================
# BERTopic Tests (Mocked)
# ============================================================================


class TestBERTopicClustering:
    """Tests for BERTopic clustering (mocked)."""

    @pytest.mark.asyncio
    async def test_bertopic_import_error(self, sample_annotations):
        """Test that missing BERTopic raises ImportError."""
        discovery = PatternDiscovery()

        with patch.dict('sys.modules', {'bertopic': None}):
            with pytest.raises(ImportError, match='BERTopic not installed'):
                await discovery.discover(
                    sample_annotations, method=ClusteringMethod.BERTOPIC
                )

    @pytest.mark.asyncio
    async def test_bertopic_too_few_docs(self):
        """Test BERTopic with too few documents."""
        annotations = {
            'rec_1': AnnotatedItem(record_id='rec_1', score=0, notes='Note 1'),
            'rec_2': AnnotatedItem(record_id='rec_2', score=0, notes='Note 2'),
        }
        discovery = PatternDiscovery()

        # Mock BERTopic import to succeed
        mock_bertopic = MagicMock()
        mock_keybert = MagicMock()

        with patch.dict(
            'sys.modules',
            {
                'bertopic': mock_bertopic,
                'bertopic.representation': mock_keybert,
            },
        ):
            # Re-import to get mocked version
            result = await discovery._cluster_with_bertopic(
                discovery._normalize_annotations(annotations)
            )

        assert result.patterns == []
        assert result.metadata.get('error') == 'Too few documents for BERTopic (min 5)'

    @pytest.mark.asyncio
    async def test_bertopic_uses_configured_embedding_model(self):
        """Test that BERTopic uses the configured embedding_model."""
        annotations = {
            'rec_1': AnnotatedItem(record_id='rec_1', score=0, notes='Note 1'),
            'rec_2': AnnotatedItem(record_id='rec_2', score=0, notes='Note 2'),
            'rec_3': AnnotatedItem(record_id='rec_3', score=0, notes='Note 3'),
            'rec_4': AnnotatedItem(record_id='rec_4', score=0, notes='Note 4'),
            'rec_5': AnnotatedItem(record_id='rec_5', score=0, notes='Note 5'),
        }

        custom_embedding_model = 'my-embedding-model'
        discovery = PatternDiscovery(bertopic_embedding_model=custom_embedding_model)

        class FakeTopicInfo:
            def iterrows(self):
                # One real topic (0) and an outlier bucket (-1)
                yield 0, {'Topic': 0}
                yield 1, {'Topic': -1}

        fake_model = MagicMock()
        fake_model.fit_transform.return_value = (
            [0, 0, 0, -1, -1],
            [0.9, 0.8, 0.85, 0.2, 0.1],
        )
        fake_model.get_topic_info.return_value = FakeTopicInfo()
        fake_model.get_topic.return_value = [
            ('context', 0.5),
            ('missing', 0.4),
            ('format', 0.3),
        ]

        bertopic_module = SimpleNamespace(BERTopic=MagicMock(return_value=fake_model))
        representation_module = SimpleNamespace(
            KeyBERTInspired=MagicMock(return_value=MagicMock())
        )

        with patch.dict(
            'sys.modules',
            {
                'bertopic': bertopic_module,
                'bertopic.representation': representation_module,
            },
        ):
            result = await discovery.discover(
                annotations, method=ClusteringMethod.BERTOPIC
            )

        bertopic_module.BERTopic.assert_called_once()
        _, kwargs = bertopic_module.BERTopic.call_args
        assert kwargs['embedding_model'] == custom_embedding_model
        assert result.method == ClusteringMethod.BERTOPIC
