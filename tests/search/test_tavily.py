from unittest.mock import Mock, patch

import pytest

# Remove tracing for testing
from axion._core.tracing import clear_tracing_config, configure_tracing
from axion.search.schema import SearchNode
from axion.search.tavily_retriever import TavilyRetriever

clear_tracing_config()
configure_tracing('noop')


@pytest.fixture
def tavily_retriever():
    with patch('axion._core.environment.resolve_api_key', return_value='mock-api-key'):
        return TavilyRetriever(api_key='mock-api-key')


@pytest.fixture
def mock_post():
    with patch('requests.post') as mock:
        yield mock


def test_generate_search_payload(tavily_retriever):
    query = 'AI advancements'
    payload = tavily_retriever._generate_search_payload(query)
    assert payload['query'] == query
    assert payload['topic'] == 'general'


def test_generate_extract_payload(tavily_retriever):
    urls = ['https://example.com']
    payload = tavily_retriever._generate_extract_payload(urls)
    assert payload['urls'] == urls
    assert payload['extract_depth'] == 'basic'


def test_generate_crawl_payload(tavily_retriever):
    url = 'https://example.com'
    payload = tavily_retriever._generate_crawl_payload(url)
    assert payload['url'] == url
    assert payload['max_depth'] == 1


@pytest.mark.asyncio
async def test_retrieve_search_success(tavily_retriever, mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {
        'query': 'AI',
        'results': [
            {
                'url': 'https://example.com',
                'content': 'Example content',
                'score': 0.9,
                'date': '2023-01-01',
            }
        ],
        'answer': 'AI is evolving.',
        'images': [],
        'response_time': 123,
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    results = await tavily_retriever.retrieve('AI')

    assert isinstance(results.nodes[0], SearchNode)
    assert results.query == 'AI'
    assert results.answer == 'AI is evolving.'


@pytest.mark.asyncio
async def test_extract_success(tavily_retriever, mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {
        'results': [
            {
                'url': 'https://example.com',
                'raw_content': 'Extracted content',
                'images': [],
            }
        ],
        'response_time': 200,
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    results = await tavily_retriever.extract('https://example.com')
    assert isinstance(results.nodes[0], SearchNode)
    assert 'Extracted content' in results.nodes[0].text


@pytest.mark.asyncio
async def test_crawl_success(tavily_retriever, mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {
        'base_url': 'https://example.com',
        'results': [
            {'url': 'https://example.com/page', 'raw_content': 'Crawled page content'}
        ],
        'response_time': 300,
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    results = await tavily_retriever.crawl('https://example.com')
    assert results.query == 'https://example.com'
    assert 'Crawled page content' in results.nodes[0].text
