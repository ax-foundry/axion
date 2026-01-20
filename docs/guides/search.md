# Search Integrations

Axion provides retriever implementations for popular search APIs to integrate into evaluation pipelines.

## Available Retrievers

| Retriever | API | Best For |
|-----------|-----|----------|
| `GoogleRetriever` | Google Custom Search | Web search, broad coverage |
| `TavilyRetriever` | Tavily AI | AI-optimized search results |
| `YouRetriever` | You.com | Real-time web data |

## Quick Start

```python
from axion.search import GoogleRetriever

retriever = GoogleRetriever(
    api_key="your-serpapi-key"
)

results = await retriever.search("What is RAG in AI?")
for result in results:
    print(f"{result.title}: {result.url}")
```

## Google Search

```python
from axion.search import GoogleRetriever

retriever = GoogleRetriever(
    api_key="your-serpapi-key",
    num_results=10
)

results = await retriever.search(query)
```

## Tavily Search

AI-optimized search with relevance filtering:

```python
from axion.search import TavilyRetriever

retriever = TavilyRetriever(
    api_key="your-tavily-key",
    search_depth="advanced"
)

results = await retriever.search(query)
```

## You.com Search

Real-time web data with snippet extraction:

```python
from axion.search import YouRetriever

retriever = YouRetriever(
    api_key="your-you-key"
)

results = await retriever.search(query)
```

## Using with Evaluation

Combine retrievers with evaluation metrics:

```python
from axion import DatasetItem
from axion.metrics import ContextualRelevancy

# Get retrieval results
results = await retriever.search(query)
content = [r.snippet for r in results]

# Create evaluation item
item = DatasetItem(
    query=query,
    actual_output=agent_response,
    retrieved_content=content
)

# Evaluate retrieval quality
metric = ContextualRelevancy()
score = await metric.evaluate(item)
```

## Next Steps

- [Google Search Deep Dive](../deep-dives/search/google.md) - Full configuration options
- [Tavily Search Deep Dive](../deep-dives/search/tavily.md) - Advanced features
- [API Reference: Search](../reference/search.md) - Full API documentation
