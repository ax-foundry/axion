from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from axion._core.schema import AIMessage, HumanMessage
from axion.dataset import DatasetItem
from axion.dataset_schema import MultiTurnConversation
from axion.metrics.base import MetricEvaluationResult
from axion.metrics.heuristic.citation_presence import CitationPresence


class MockEmbeddingModel:
    """Mock embedding model for testing semantic similarity."""

    async def aget_text_embedding_batch(self, texts):
        """Return mock embeddings based on text content."""
        embeddings = []
        for text in texts:
            if any(
                phrase in text.lower()
                for phrase in [
                    'detailed information',
                    'refer to',
                    'additional',
                    'resources',
                ]
            ):
                # High similarity embedding for resource-like text
                embeddings.append(np.ones(300) * 0.9)
            else:
                # Low similarity embedding for other text
                embeddings.append(np.ones(300) * 0.1)
        return embeddings


@pytest.fixture
def mock_embedding_model():
    """Fixture providing a mock embedding model."""
    return MockEmbeddingModel()


# --- Fixtures for Multi-Turn Conversations ---


@pytest.fixture
def conversation_with_citations() -> MultiTurnConversation:
    """Conversation where only the second assistant turn has a citation.
    Indices: [1, 3, 5] are AIMessages. Index 3 is the passing turn."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Query 1: Tell me about the weather.'),
            AIMessage(
                content='The weather is sunny.'
            ),  # Turn 1 (Index 1) - No citation
            HumanMessage(content='Query 2: Where can I find documentation?'),
            AIMessage(
                content='See this doc: https://example.com/doc. For more details, see these resources:\n- Link 1'
            ),  # Turn 2 (Index 3) - Has citation/section, **Expected Pass Index**
            HumanMessage(content='Query 3: Final question.'),
            AIMessage(content='Final answer.'),  # Turn 3 (Index 5) - No citation
        ]
    )


@pytest.fixture
def conversation_no_citations() -> MultiTurnConversation:
    """Conversation where no assistant turn has any citation."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Query 1.'),
            AIMessage(content='Answer 1.'),
            HumanMessage(content='Query 2.'),
            AIMessage(content='Answer 2.'),
        ]
    )


@pytest.fixture
def conversation_resource_section() -> MultiTurnConversation:
    """Conversation where an assistant turn contains a dedicated resource section.
    Indices: [1, 3] are AIMessages. Index 3 is the passing turn."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Query A.'),
            AIMessage(content='Answer A.'),
            HumanMessage(content='Query B: Get resources.'),
            AIMessage(
                content='Response B.\nFor more detailed information, you can refer to the following resources:\n- https://doc.com/b'
            ),  # Turn 2 (Index 3) - Has section and citation, **Expected Pass Index**
        ]
    )


# --- Test Classes ---

# Define the expected failure explanation once for cleanliness
EXPECTED_FAILURE_EXPLANATION = 'Mode: any_citation. FAILURE: No assistant message satisfied the citation requirement. Ensure citations are present and, if in strict mode, URLs are live.'
EXPECTED_NONE_EXPLANATION = 'No actual output provided'


class TestCitationPresenceInitialization:
    """Test Citation Presence metric initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        metric = CitationPresence()
        assert metric.mode == 'any_citation'
        assert metric.strict is False
        assert metric.embed_model is None
        assert metric.use_semantic_search is False
        assert metric.resource_similarity_threshold == 0.8
        assert (
            metric.custom_resource_phrases == CitationPresence._default_resource_phrases
        )

    def test_custom_initialization(self, mock_embedding_model):
        """Test initialization with custom parameters."""
        custom_phrases = ['Check our docs:', 'See references:']
        metric = CitationPresence(
            mode='resource_section',
            strict=True,
            embed_model=mock_embedding_model,
            use_semantic_search=True,
            resource_similarity_threshold=0.9,
            custom_resource_phrases=custom_phrases,
        )
        assert metric.mode == 'resource_section'
        assert metric.strict is True
        assert metric.embed_model == mock_embedding_model
        assert metric.use_semantic_search is True
        assert metric.resource_similarity_threshold == 0.9
        # Should include both custom and default phrases
        assert all(
            phrase in metric.custom_resource_phrases for phrase in custom_phrases
        )
        assert all(
            phrase in metric.custom_resource_phrases
            for phrase in CitationPresence._default_resource_phrases
        )

    def test_semantic_search_without_model_raises_error(self):
        """Test that using semantic search without embedding model raises error."""
        with pytest.raises(
            ValueError,
            match='An embed_model is required when use_semantic_search is True',
        ):
            CitationPresence(use_semantic_search=True, embed_model=None)


class TestAnyCitationMode:
    """Test any_citation mode functionality (single-turn and multi-turn)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'text, expected_score, expected_message_count',
        [
            # Single-Turn Tests (uses actual_output)
            ('This is just plain text without any links.', 0.0, 0),
            ('Check out https://example.com for more info.', 1.0, 1),
        ],
    )
    async def test_any_citation_single_turn_basic(
        self, text, expected_score, expected_message_count
    ):
        """Test basic citation detection in single-turn mode."""
        item = DatasetItem(actual_output=text, query='test')
        metric = CitationPresence(mode='any_citation')
        result = await metric.execute(item)

        assert isinstance(result, MetricEvaluationResult)
        assert result.score == expected_score
        assert result.signals.passes_presence_check == (expected_score == 1.0)
        assert len(result.signals.messages_with_citations) == expected_message_count
        assert result.signals.total_assistant_messages == 1

    @pytest.mark.asyncio
    async def test_any_citation_multi_turn_success(self, conversation_with_citations):
        """Test multi-turn success (citation found in one turn)."""
        item = DatasetItem(
            conversation=conversation_with_citations, query='Final question.'
        )
        metric = CitationPresence(mode='any_citation')
        result = await metric.execute(item)

        # Total assistant messages (indices 1, 3, 5)
        assert result.signals.total_assistant_messages == 3
        # Passed on the second message (index 3)
        assert result.score == 1.0
        assert result.signals.passes_presence_check is True
        assert result.signals.messages_with_citations == [3]

    @pytest.mark.asyncio
    async def test_any_citation_multi_turn_failure(self, conversation_no_citations):
        """Test multi-turn failure (no citation found anywhere)."""
        item = DatasetItem(conversation=conversation_no_citations, query='Query.')
        metric = CitationPresence(mode='any_citation')
        result = await metric.execute(item)

        assert result.signals.total_assistant_messages == 2
        assert result.score == 0.0
        assert result.signals.passes_presence_check is False
        assert result.signals.messages_with_citations == []

    @pytest.mark.asyncio
    async def test_any_citation_empty_output_handling(self):
        """Test handling of empty or None actual_output."""
        # Case 1: actual_output is empty string (should return 0.0, evaluated as 1 message)
        item = DatasetItem(actual_output='', query='test')
        metric = CitationPresence(mode='any_citation')
        result = await metric.execute(item)

        assert result.score == 0.0
        assert result.explanation == EXPECTED_FAILURE_EXPLANATION

        # Case 2: actual_output is None (should hit initial check)
        item_none = DatasetItem(actual_output=None, query='test')
        metric = CitationPresence(mode='any_citation')
        result_none = await metric.execute(item_none)

        assert result_none.score == 0.0
        assert result_none.explanation == EXPECTED_NONE_EXPLANATION

    @pytest.mark.asyncio
    async def test_any_citation_empty_conversation_handling(self):
        """Test handling of a conversation with no assistant messages."""
        # This relies on the metric falling back to checking actual_output=""
        item = DatasetItem(actual_output='', query='test')
        metric = CitationPresence(mode='any_citation')
        result = await metric.execute(item)

        # Expectation: Metric falls back to check actual_output="", which is 1 assistant message (empty), leading to 0.0 score.
        assert result.score == 0.0
        assert result.signals.total_assistant_messages == 1
        assert result.explanation == EXPECTED_FAILURE_EXPLANATION


class TestResourceSectionMode:
    """Test resource_section mode functionality (single-turn and multi-turn)."""

    @pytest.mark.asyncio
    async def test_resource_section_single_turn_perfect_match(self):
        """Test perfect resource section detection in single-turn mode."""
        text = """Data Cloud is a powerful platform.
        For more detailed information, you can refer to the following resources:
        - Documentation: https://docs.example.com"""
        item = DatasetItem(actual_output=text, query='test')
        metric = CitationPresence(mode='resource_section')
        result = await metric.execute(item)

        assert result.score == 1.0
        assert result.signals.passes_presence_check is True
        assert result.signals.messages_with_citations == [0]
        assert result.signals.total_assistant_messages == 1

    @pytest.mark.asyncio
    async def test_resource_section_multi_turn_success(
        self, conversation_resource_section
    ):
        """Test multi-turn success (resource section found in one turn)."""
        item = DatasetItem(conversation=conversation_resource_section, query='Query B.')
        metric = CitationPresence(mode='resource_section')
        result = await metric.execute(item)

        # Total assistant messages (indices 1, 3)
        assert result.signals.total_assistant_messages == 2
        # Passed on the second message (index 3)
        assert result.score == 1.0
        assert result.signals.passes_presence_check is True
        assert result.signals.messages_with_citations == [3]

    @pytest.mark.asyncio
    async def test_resource_section_multi_turn_failure(
        self, conversation_with_citations
    ):
        """Test multi-turn failure where citations are scattered, not in a dedicated section."""
        item = DatasetItem(conversation=conversation_with_citations, query='Query.')
        metric = CitationPresence(mode='resource_section')
        result = await metric.execute(item)

        assert result.signals.total_assistant_messages == 3
        assert result.score == 0.0
        assert result.signals.passes_presence_check is False
        assert result.signals.messages_with_citations == []

    @pytest.mark.asyncio
    async def test_resource_section_semantic_search_fallback(
        self, mock_embedding_model
    ):
        """Test semantic search fallback in resource section mode."""
        text = """
        Data Cloud provides powerful insights.

        You can find additional documentation at these locations:
        - Technical guide: https://tech.example.com
        - User manual: https://manual.example.com
        """
        item = DatasetItem(actual_output=text, query='test')
        metric = CitationPresence(
            mode='resource_section',
            embed_model=mock_embedding_model,
            use_semantic_search=True,
        )
        result = await metric.execute(item)

        assert result.score == 1.0
        assert result.signals.passes_presence_check is True


class TestStrictMode:
    """Test strict mode functionality for URL validation across turns."""

    @pytest.mark.asyncio
    @patch('axion.metrics.heuristic.citation_presence.CitationPresence._validate_urls')
    async def test_strict_mode_valid_urls_multi_turn(self, mock_validate_urls):
        """Test strict mode success where one turn has valid URLs."""

        # Define the side effect to return validated URLs only for the "good link" turn
        def validate_urls_side_effect(urls: List[str]) -> List[str]:
            if any('example.com/doc' in u for u in urls):
                # This is the "good link" turn (should pass validation)
                return [u for u in urls if 'example.com/doc' in u]
            else:
                # This is the "bad link" turn or a turn with no URLs (should fail validation)
                return []

        mock_validate_urls.side_effect = validate_urls_side_effect

        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Q1'),
                AIMessage(
                    content='Bad link: https://invalid.com'
                ),  # Index 1 (Fails validation)
                HumanMessage(content='Q2'),
                AIMessage(
                    content='Good link: https://example.com/doc. See more: https://docs.com'
                ),
                # Index 3 (Passes validation)
            ]
        )

        item = DatasetItem(conversation=conversation, query='Q2')
        metric = CitationPresence(mode='any_citation', strict=True)
        result = await metric.execute(item)

        assert result.score == 1.0
        assert result.signals.passes_presence_check is True
        # Only the second assistant message (index 3) should be listed as passing
        assert result.signals.messages_with_citations == [3]
        assert result.signals.total_assistant_messages == 2

    @pytest.mark.asyncio
    @patch('axion.metrics.heuristic.citation_presence.CitationPresence._validate_urls')
    async def test_strict_mode_invalid_urls_multi_turn(self, mock_validate_urls):
        """Test strict mode failure where all found URLs are invalid."""
        mock_validate_urls.return_value = []

        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Q1'),
                AIMessage(content='Link 1: https://invalid1.com'),
                HumanMessage(content='Q2'),
                AIMessage(content='Link 2: https://invalid2.com'),
            ]
        )
        item = DatasetItem(conversation=conversation, query='Q2')
        metric = CitationPresence(mode='any_citation', strict=True)
        result = await metric.execute(item)

        assert result.score == 0.0
        assert result.signals.passes_presence_check is False
        assert result.signals.messages_with_citations == []
        assert result.signals.total_assistant_messages == 2


class TestCitationExtraction:
    """Test citation extraction functionality. (Kept simplified as in the original code snippet)."""

    @pytest.mark.parametrize(
        'text, expected_citations',
        [
            ('Plain text without URLs', []),
            ('Visit http://example.com', ['http://example.com']),
            (
                'See https://docs.com and https://help.com',
                ['https://docs.com', 'https://help.com'],
            ),
            (
                'Visit https://example.com and https://example.com',
                ['https://example.com'],
            ),
        ],
    )
    def test_extract_citations(self, text, expected_citations):
        """Test URL extraction from various text formats."""
        citations = CitationPresence()._extract_citations(text)
        assert citations == expected_citations
