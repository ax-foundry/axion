from datetime import datetime, timedelta

import pytest

# Remove tracing for testing
from axion._core.tracing import clear_tracing_config, configure_tracing
from axion.search.base import BaseRetriever
from axion.search.schema import SearchNode, SearchResults

clear_tracing_config()
configure_tracing('noop')


class DummyRetriever(BaseRetriever):
    async def retrieve(self, query: str):
        return SearchResults(
            nodes=[
                SearchNode(
                    text='Sample', source='http://example.com', date='1 week ago'
                )
            ],
            query=query,
        )


def test_nodes_to_list():
    nodes = [SearchNode(text='hello', source='src', date='today')]
    result = BaseRetriever.nodes_to_list(nodes, query='q', answer='a', latency=0.1)
    assert isinstance(result, SearchResults)
    assert len(result.nodes) == 1
    assert result.query == 'q'
    assert result.answer == 'a'
    assert result.latency == 0.1


@pytest.mark.parametrize(
    'input_text,expected_range',
    [
        ('2 weeks ago', timedelta(days=14)),
        ('1 month ago', timedelta(days=30)),
        ('3 years ago', timedelta(days=1095)),
        ('invalid date', None),
        ('', None),
    ],
)
def test_extract_date_from_text(input_text, expected_range):
    result = BaseRetriever.extract_date_from_text(input_text)
    if expected_range is None:
        assert result == input_text or result is None
    else:
        result_date = datetime.strptime(result, '%B %d, %Y')
        delta = datetime.now() - result_date
        assert abs(delta - expected_range) < timedelta(days=2)  # allow small time skew


@pytest.mark.asyncio
async def test_execute_async():
    retriever = DummyRetriever()
    result = await retriever.execute('test query')
    assert isinstance(result, SearchResults)
    assert result.query == 'test query'
    assert len(result.nodes) == 1


def test_execute_sync():
    retriever = DummyRetriever()
    result = retriever.execute_sync('sync query')
    assert isinstance(result, SearchResults)
    assert result.query == 'sync query'
    assert len(result.nodes) == 1
