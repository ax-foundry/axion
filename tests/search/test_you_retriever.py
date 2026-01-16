import warnings
from unittest.mock import Mock, patch

import pytest
import requests

# Remove tracing for testing
from axion._core.tracing import clear_tracing_config, configure_tracing
from axion.search.schema import SearchNode, SearchResults
from axion.search.you_retriever import YouRetriever

clear_tracing_config()
configure_tracing('noop')


@pytest.fixture
def you_retriever():
    """Create a YouRetriever instance with default settings."""
    with patch('axion._core.environment.resolve_api_key', return_value='mock-api-key'):
        retriever = YouRetriever(api_key='mock-api-key')
        return retriever


@pytest.fixture
def you_news_retriever():
    """Create a YouRetriever instance configured for news endpoint."""
    with patch('axion._core.environment.resolve_api_key', return_value='mock-api-key'):
        retriever = YouRetriever(
            api_key='mock-api-key',
            endpoint='news',
            search_lang='en',
            ui_lang='en',
            spellcheck=True,
        )
        return retriever


@pytest.fixture
def mock_search_response():
    """Create a mock search response from the You.com API."""
    return {
        'hits': [
            {
                'url': 'https://example.com/result1',
                'title': 'Example Result 1',
                'snippets': [
                    'This is the first snippet.',
                    'This is additional content.',
                ],
                'age': '2 weeks ago',
            },
            {
                'url': 'https://example.com/result2',
                'title': 'Example Result 2',
                'snippets': ['This is the second snippet.'],
                'age': '1 month ago',
            },
        ]
    }


@pytest.fixture
def mock_news_response():
    """Create a mock news response from the You.com API."""
    return {
        'news': {
            'results': [
                {
                    'url': 'https://example.com/news1',
                    'title': 'Example News 1',
                    'description': 'This is the first news article.',
                    'age': '3 days ago',
                },
                {
                    'url': 'https://example.com/news2',
                    'title': 'Example News 2',
                    'description': 'This is the second news article.',
                    'age': '1 week ago',
                },
            ]
        }
    }


# Tests for YouRetriever
class TestYouRetriever:
    def test_initialization_default_params(self, you_retriever):
        """Test initialization with default parameters."""
        assert you_retriever._api_key == 'mock-api-key'
        assert you_retriever.endpoint == 'search'
        assert you_retriever.num_web_results == 5
        assert you_retriever.safesearch is None
        assert you_retriever.country is None
        assert you_retriever.search_lang is None
        assert you_retriever.ui_lang is None
        assert you_retriever.spellcheck is None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        with patch(
            'axion._core.environment.resolve_api_key', return_value='mock-api-key'
        ):
            retriever = YouRetriever(
                api_key='mock-api-key',
                endpoint='search',
                num_web_results=10,
                safesearch='moderate',
                country='US',
            )

            assert retriever.endpoint == 'search'
            assert retriever.num_web_results == 10
            assert retriever.safesearch == 'moderate'
            assert retriever.country == 'US'

    def test_initialization_invalid_endpoint(self):
        """Test initialization with an invalid endpoint."""
        with patch(
            'axion._core.environment.resolve_api_key', return_value='mock-api-key'
        ):
            with pytest.raises(ValueError) as excinfo:
                YouRetriever(api_key='mock-api-key', endpoint='invalid')

            assert '`endpoint` must be either "search" or "news"' in str(excinfo.value)

    def test_initialization_news_with_search_params_warning(self):
        """Test that a warning is issued when news-specific params are used with search endpoint."""
        with patch(
            'axion._core.environment.resolve_api_key', return_value='mock-api-key'
        ):
            with warnings.catch_warnings(record=True) as w:
                # Enable warnings
                warnings.simplefilter('always')

                # Initialize with news-specific params but search endpoint
                YouRetriever(
                    api_key='mock-api-key',
                    endpoint='search',
                    search_lang='en',
                    ui_lang='en',
                    spellcheck=True,
                )

                # Check if warnings were raised
                assert len(w) == 3
                for warning in w:
                    assert issubclass(warning.category, UserWarning)
                    assert 'News API-specific field' in str(warning.message)

    def test_generate_params_search(self, you_retriever):
        """Test generating parameters for search endpoint."""
        query = 'test query'
        params = you_retriever._generate_params(query)

        assert params['query'] == query
        assert params['num_web_results'] == 5
        assert 'search_lang' not in params
        assert 'ui_lang' not in params
        assert 'spellcheck' not in params

    def test_generate_params_news(self, you_news_retriever):
        """Test generating parameters for news endpoint."""
        query = 'test query'
        params = you_news_retriever._generate_params(query)

        assert params['q'] == query
        assert params['count'] == 5
        assert params['search_lang'] == 'en'
        assert params['ui_lang'] == 'en'
        assert params['spellcheck'] is True

    def test_generate_params_with_none_values(self, you_retriever):
        """Test that None values are filtered out from params."""
        # All default params except setting safesearch explicitly to None
        you_retriever.safesearch = None
        params = you_retriever._generate_params('test')

        assert 'safesearch' not in params
        assert 'country' not in params

    @patch('requests.get')
    def test_retrieve_search_endpoint(
        self, mock_get, you_retriever, mock_search_response
    ):
        """Test retrieving results from search endpoint."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = mock_search_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Create a query bundle
        query = 'test query'

        # Call _retrieve
        results = you_retriever.execute_sync(query)

        # Check that API was called with correct params
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == 'https://api.ydc-index.io/search'
        assert kwargs['headers'] == {'X-API-Key': 'mock-api-key'}
        assert kwargs['params']['query'] == 'test query'

        # Check results
        node = results.nodes[0]
        assert len(results) == 2
        assert isinstance(results, SearchResults)
        assert all(isinstance(result, SearchNode) for result in results)
        assert (
            node.text
            == 'Example Result 1\nThis is the first snippet.\nThis is additional content.'
        )
        assert node.source == 'https://example.com/result1'
        assert node.date == '2 weeks ago'

    @patch('requests.get')
    def test_retrieve_news_endpoint(
        self, mock_get, you_news_retriever, mock_news_response
    ):
        """Test retrieving results from news endpoint."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = mock_news_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Create a query bundle
        query = 'test query'

        # Call _retrieve
        results = you_news_retriever.execute_sync(query)

        # Check that API was called with correct params
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == 'https://api.ydc-index.io/news'
        assert kwargs['headers'] == {'X-API-Key': 'mock-api-key'}
        assert kwargs['params']['q'] == 'test query'

        # Check results
        node = results.nodes[0]
        assert len(results) == 2
        assert isinstance(results, SearchResults)
        assert all(isinstance(result, SearchNode) for result in results)
        assert node.text == 'Example News 1\nThis is the first news article.'
        assert node.source == 'https://example.com/news1'
        assert node.date == '3 days ago'

    @patch('requests.get')
    def test_retrieve_empty_query(self, mock_get, you_retriever):
        """Test retrieving with an empty query string."""
        # Create an empty query bundle
        query = ''

        # Call _retrieve, which should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            you_retriever.execute_sync(query)

        assert 'Query string cannot be empty' in str(excinfo.value)
        mock_get.assert_not_called()

    @patch('requests.get')
    def test_retrieve_http_error(self, mock_get, you_retriever):
        """Test handling of HTTP errors in retrieve."""
        # Setup mock to raise an HTTP error
        mock_response = Mock()
        mock_response.raise_for_status = Mock(
            side_effect=requests.HTTPError('404 Client Error')
        )
        mock_get.return_value = mock_response

        # Create a query bundle
        query = 'test query'

        # Call _retrieve, which should propagate the HTTP error
        with pytest.raises(requests.HTTPError) as excinfo:
            you_retriever.execute_sync(query)

        assert '404 Client Error' in str(excinfo.value)

    @patch('requests.get')
    def test_retrieve_no_results(self, mock_get, you_retriever):
        """Test retrieving with no results in the response."""
        # Setup mock response with no hits
        mock_response = Mock()
        mock_response.json.return_value = {'hits': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Create a query bundle
        query = 'test query'
        results = you_retriever.execute_sync(query)
        # Check that results is an empty list
        assert len(results) == 0

    @patch('requests.get')
    def test_retrieve_integration_with_retriever_mixin(
        self, mock_get, you_retriever, mock_search_response
    ):
        """Test integration with BaseRetriever functions."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = mock_search_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Create a query bundle
        query = 'test query'

        # Mock the nodes_to_list method from BaseRetriever
        original_method = you_retriever.nodes_to_list
        with patch.object(
            you_retriever, 'nodes_to_list', wraps=original_method
        ) as mock_nodes_to_list:
            _ = you_retriever.execute_sync(query)

            # Check that nodes_to_list was called
            mock_nodes_to_list.assert_called_once()

            # Check the argument to nodes_to_list
            nodes_arg = mock_nodes_to_list.call_args[0][0]
            assert len(nodes_arg) == 2
            assert all(isinstance(node, SearchNode) for node in nodes_arg)
