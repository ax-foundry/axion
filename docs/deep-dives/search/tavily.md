# Tavily WebSearch

## Overview
Tavily is an AI-powered search API designed for AI agents and applications that need real-time, accurate, and comprehensive search results. It provides intelligent web search capabilities with content extraction and crawling features to enhance AI applications with up-to-date information.

## API Access
To integrate Tavily into your application, obtain an API key and configure it accordingly.

### Steps to Access Tavily API
1. Sign up and obtain an API key from [Tavily API](https://tavily.com/).
2. Pass the API key in requests or store it securely in your environment variables.
3. Implement API calls within your AI application to fetch real-time search results.

Please note that this file will search for these credentials in an `.env` file at the project's root directory.

## Code Examples
```py title="TavilyRetriever Example"
from axion.search import TavilyRetriever

api_client = TavilyRetriever(num_web_results=3, crawl_pages=False)
results = await api_client.execute('What are Ciena Corp challenges?')
for result in results:
    print(result)

# Crawl specific pages
results = await api_client.crawl('https://www.espn.com/')

# Extract content from URLs
results = await api_client.extract('https://www.espn.com/')
```

## API Reference
::: axion.search.tavily_retriever
