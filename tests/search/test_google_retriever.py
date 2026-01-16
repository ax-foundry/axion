from unittest.mock import Mock, patch

import pytest

# Remove tracing for testing
from axion._core.tracing import clear_tracing_config, configure_tracing
from axion.search.google_retriever import GoogleRetriever
from axion.search.schema import SearchNode, SearchResults

clear_tracing_config()
configure_tracing('noop')


@pytest.fixture
def google_retriever():
    """Create a GoogleRetriever instance with default settings."""
    with patch('axion._core.environment.resolve_api_key', return_value='mock-api-key'):
        retriever = GoogleRetriever(api_key='mock-api-key')
        return retriever


@pytest.fixture
def mock_search_results():
    """Create mock search results from the Google Search API."""
    return {
        'search_metadata': {
            'id': 'mock-search-id',
            'status': 'Success',
            'json_endpoint': 'https://serpapi.com/searches/mock-id/json',
            'created_at': '2023-01-01 12:00:00 UTC',
            'processed_at': '2023-01-01 12:00:01 UTC',
            'google_url': 'https://www.google.com/search?q=test+query',
            'raw_html_file': 'https://serpapi.com/searches/mock-id/raw_html',
            'total_time_taken': 1.23,
        },
        'search_parameters': {'engine': 'google', 'q': 'test query', 'num': 5},
        'organic_results': [
            {
                'position': 1,
                'title': 'Test Result 1',
                'link': 'https://example.com/result1',
                'snippet': 'This is the first result snippet.',
                'snippet_highlighted_words': ['test', 'result'],
                'age': '2 days ago',
            },
            {
                'position': 2,
                'title': 'Test Result 2',
                'link': 'https://example.com/result2',
                'snippet': 'This is the second result snippet.',
                'rich_snippet': 'This is a rich snippet.',
                'snippet_highlighted_words': ['test'],
                'age': '1 week ago',
            },
            {
                'position': 3,
                'title': 'Test Result 3',
                'link': 'https://example.com/result3',
                'snippet': 'This is the third result snippet.',
                'rich_snippet_table': 'Column1 | Column2\nValue1 | Value2',
                'age': '2 weeks ago',
            },
        ],
    }


class TestGoogleRetriever:
    def test_initialization_default_params(self, google_retriever):
        """Test initialization with default parameters."""
        assert google_retriever._api_key == 'mock-api-key'
        assert google_retriever.num_web_results == 5

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        with patch(
            'axion._core.environment.resolve_api_key', return_value='mock-api-key'
        ):
            retriever = GoogleRetriever(api_key='custom-api-key', num_web_results=10)
            assert retriever.num_web_results == 10

    def test_generate_params(self, google_retriever):
        """Test generating parameters for SerpAPI query."""
        query = 'test query'
        params = google_retriever._generate_params(query)

        assert params['engine'] == 'google'
        assert params['api_key'] == 'mock-api-key'
        assert params['q'] == query
        assert params['num'] == 5

    @patch('axion.search.google_retriever.GoogleSearch')
    def test_retrieve(self, mock_google_search, google_retriever, mock_search_results):
        """Test retrieving results from Google Search API."""
        # Setup mock GoogleSearch instance
        mock_search_instance = Mock()
        mock_search_instance.get_dict.return_value = mock_search_results
        mock_google_search.return_value = mock_search_instance

        query = 'test query'

        # Call _retrieve
        results = google_retriever.execute_sync(query)

        # Check that GoogleSearch was initialized with correct params
        mock_google_search.assert_called_once()
        args, kwargs = mock_google_search.call_args
        expected_params = {
            'engine': 'google',
            'api_key': 'mock-api-key',
            'q': 'test query',
            'num': 5,
        }
        assert args[0] == expected_params

        # Check results
        assert len(results) == 3
        assert isinstance(results, SearchResults)
        assert all(isinstance(result, SearchNode) for result in results)

        # Check first result
        node = results.nodes[0]
        assert node.text == 'Test Result 1\nThis is the first result snippet.'
        assert node.source == 'https://example.com/result1'
        assert node.date == '2 days ago'
        assert node.highlights == ['test', 'result']

        # Check second result with rich snippet
        assert (
            'Test Result 2\nThis is the second result snippet.\nThis is a rich snippet.'
            in results.nodes[1].text
        )

        # Check third result with rich snippet table
        assert (
            'Test Result 3\nThis is the third result snippet.\nColumn1 | Column2\nValue1 | Value2'
            in results.nodes[2].text
        )

    @patch('axion.search.google_retriever.GoogleSearch')
    def test_retrieve_empty_query(self, mock_google_search, google_retriever):
        """Test retrieving with an empty query string."""
        # Create an empty query bundle
        query = ''

        # Call _retrieve, which should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            google_retriever.execute_sync(query)

        assert 'Query string cannot be empty' in str(excinfo.value)
        mock_google_search.assert_not_called()

    @patch('axion.search.google_retriever.GoogleSearch')
    def test_retrieve_no_results(self, mock_google_search, google_retriever):
        """Test retrieving with no results in the response."""
        # Setup mock with no organic results
        mock_search_instance = Mock()
        mock_search_instance.get_dict.return_value = {'organic_results': []}
        mock_google_search.return_value = mock_search_instance

        # Create a query bundle
        query = 'test query'
        results = google_retriever.execute_sync(query)
        # Check that results is an empty list
        assert len(results) == 0

    @patch('axion.search.google_retriever.GoogleSearch')
    def test_retrieve_missing_organic_results_key(
        self, mock_google_search, google_retriever
    ):
        """Test retrieving with a response missing the organic_results key."""
        # Setup mock with missing organic_results
        mock_search_instance = Mock()
        mock_search_instance.get_dict.return_value = {
            'search_metadata': {}
        }  # No organic_results key
        mock_google_search.return_value = mock_search_instance

        # Create a query bundle
        query = 'test query'

        results = google_retriever.execute_sync(query)
        # Check that results is an empty list
        assert len(results) == 0

    @patch('axion.search.google_retriever.GoogleSearch')
    def test_retrieve_result_missing_fields(self, mock_google_search, google_retriever):
        """Test retrieving with results missing some fields."""
        # Setup mock with minimal fields
        minimal_results = {
            'organic_results': [
                {
                    'title': 'Minimal Result',
                    # Missing link, snippet, etc.
                }
            ]
        }
        mock_search_instance = Mock()
        mock_search_instance.get_dict.return_value = minimal_results
        mock_google_search.return_value = mock_search_instance

        # Create a query bundle
        query = 'test query'

        # Call _retrieve
        results = google_retriever.execute_sync(query)
        node = results.nodes[0]

        # Check results - should handle missing fields gracefully
        assert len(results) == 1
        assert node.text == 'Minimal Result'
        assert not node.source
        assert not node.date
        assert node.highlights == []

    @patch('axion.search.google_retriever.GoogleSearch')
    def test_retrieve_integration_with_retriever_mixin(
        self, mock_google_search, google_retriever, mock_search_results
    ):
        """Test integration with BaseRetriever functions."""
        # Setup mock GoogleSearch instance
        mock_search_instance = Mock()
        mock_search_instance.get_dict.return_value = mock_search_results
        mock_google_search.return_value = mock_search_instance

        query = 'test query'
        # Mock the nodes_to_list method from BaseRetriever
        original_method = google_retriever.nodes_to_list
        with patch.object(
            google_retriever, 'nodes_to_list', wraps=original_method
        ) as mock_nodes_to_list:
            _ = google_retriever.execute_sync(query)

            # Check that nodes_to_list was called
            mock_nodes_to_list.assert_called_once()

            # Check the argument to nodes_to_list
            nodes_arg = mock_nodes_to_list.call_args[0][0]
            assert len(nodes_arg) == 3
            assert all(isinstance(node, SearchNode) for node in nodes_arg)

    @patch('axion.search.google_retriever.GoogleSearch')
    def test_text_generation_with_all_fields(
        self, mock_google_search, google_retriever
    ):
        """Test text generation when all content fields are present."""
        # Setup result with all possible content fields
        full_result = {
            'organic_results': [
                {
                    'title': 'Full Result',
                    'link': 'https://example.com/full',
                    'snippet': 'This is a snippet.',
                    'rich_snippet': 'This is a rich snippet.',
                    'rich_snippet_table': 'Column1 | Column2\nValue1 | Value2',
                }
            ]
        }
        mock_search_instance = Mock()
        mock_search_instance.get_dict.return_value = full_result
        mock_google_search.return_value = mock_search_instance

        query = 'test query'

        # Call _retrieve
        results = google_retriever.execute_sync(query)

        # Check the concatenated text contains all elements
        expected_text = 'Full Result\nThis is a snippet.\nThis is a rich snippet.\nColumn1 | Column2\nValue1 | Value2'
        assert results.nodes[0].text == expected_text

    @patch('axion.search.google_retriever.GoogleSearch')
    def test_text_generation_with_empty_fields(
        self, mock_google_search, google_retriever
    ):
        """Test text generation when some fields are empty."""
        # Setup result with some empty fields
        result_with_empty = {
            'organic_results': [
                {
                    'title': 'Result With Empties',
                    'link': 'https://example.com/empty',
                    'snippet': '',  # Empty snippet
                    'rich_snippet': None,  # None rich_snippet
                    # rich_snippet_table is missing entirely
                }
            ]
        }
        mock_search_instance = Mock()
        mock_search_instance.get_dict.return_value = result_with_empty
        mock_google_search.return_value = mock_search_instance

        query = 'test query'
        results = google_retriever.execute_sync(query)
        node = results.nodes[0]
        # Check that empty/missing fields are properly filtered
        assert node.text == 'Result With Empties'
        # No extra newlines from empty fields
        assert '\n\n' not in node.text
