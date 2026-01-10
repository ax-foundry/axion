from axion.search.schema import SearchNode, SearchResults


def test_search_node_defaults():
    node = SearchNode(text='Example node')
    assert node.text == 'Example node'
    assert node.highlights == []
    assert node.source == ''
    assert node.date == ''
    assert node.images == []
    assert node.score is None


def test_search_node_with_values():
    node = SearchNode(
        text='Test content',
        highlights=['match1', 'match2'],
        source='source_url',
        date='2025-05-12',
        score=0.9,
        images=['image1', 'image2'],
    )
    assert node.text == 'Test content'
    assert node.highlights == ['match1', 'match2']
    assert node.source == 'source_url'
    assert node.date == '2025-05-12'
    assert node.score == 0.9
    assert node.images == ['image1', 'image2']


def test_search_results_empty():
    results = SearchResults()
    assert len(results) == 0
    assert list(results) == []
    assert results.query is None
    assert results.answer is None
    assert results.images == []
    assert results.latency is None


def test_search_results_with_nodes():
    node1 = SearchNode(text='Node 1')
    node2 = SearchNode(text='Node 2')
    results = SearchResults(
        nodes=[node1, node2],
        query='example query',
        answer='This is an answer',
        images=['img1'],
        latency=0.25,
    )
    assert len(results) == 2
    assert results.query == 'example query'
    assert results.answer == 'This is an answer'
    assert results.images == ['img1']
    assert results.latency == 0.25

    # Check iterable
    texts = [node.text for node in results]
    assert texts == ['Node 1', 'Node 2']
