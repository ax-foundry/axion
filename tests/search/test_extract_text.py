import json
from unittest.mock import Mock, mock_open, patch

from axion.search.extract_text import (
    extract_clean_text_from_url,
    resolve_text_from_result,
    truncate_text,
)


def test_truncate_text_under_limit():
    assert truncate_text('Short text', 20) == 'Short text'


def test_truncate_text_over_limit():
    result = truncate_text('This is a long sentence that needs to be truncated.', 10)
    assert result == 'This is a ...'


def test_resolve_text_fallback_only():
    result_dict = {'title': 'Title', 'snippet': 'Snippet here.', 'extra': 'Ignored'}
    result = resolve_text_from_result(result_dict, crawl=False)
    assert 'Title' in result
    assert 'Snippet here.' in result


@patch('axion.search.extract_text.extract_clean_text_from_url')
def test_resolve_text_with_crawl(mock_extract):
    mock_extract.return_value = 'Crawled page content here'
    result_dict = {
        'url': 'https://example.com',
        'snippet': 'Should not be used if crawl works',
    }

    result = resolve_text_from_result(result_dict, crawl=True, max_crawl_tokens=10)
    assert result == 'Crawled pa'  # Truncated to 10 characters


@patch('axion.search.extract_text.extract_clean_text_from_url')
def test_resolve_text_crawl_fails_fallback(mock_extract):
    mock_extract.side_effect = Exception('Fail crawl')
    result_dict = {'url': 'https://example.com', 'title': 'Fallback title'}

    result = resolve_text_from_result(result_dict, crawl=True)
    assert 'Fallback title' in result


@patch('axion.search.extract_text.requests.get')
@patch(
    'builtins.open',
    new_callable=mock_open,
    read_data=json.dumps({'chrome': ['FakeAgent/1.0']}),
)
def test_extract_clean_text_from_url(mock_file, mock_get):
    # Simulate HTTP response content
    fake_html = """
    <html>
        <body>
            <p>This is a test paragraph with enough content to be captured.</p>
            <script>console.log("should be removed")</script>
            <footer>Footer text</footer>
            <h1>Headline section</h1>
        </body>
    </html>
    """
    mock_response = Mock()
    mock_response.content = fake_html.encode('utf-8')
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = extract_clean_text_from_url('http://example.com')
    assert 'This is a test paragraph' in result
