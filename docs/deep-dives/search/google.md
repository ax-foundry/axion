---
icon: custom/globe
---
# SerpAPI Web Search

## Overview
SerpAPI is a real-time API that allows developers and businesses to access search engine results from Google, Bing, Baidu, Yahoo, Yandex, eBay, and YouTube. It is known for its fast speed, variety of Google-related APIs, and affordable pricing plans. SerpAPI is a valuable tool for developers, marketers, and data analysts who need to gather search engine data for various purposes

## API Access
To use SerpAPI, obtain an API key and configure your application accordingly.


### Steps to Access SerpAPI
1. Sign up and obtain an API key from [SerpAPI](https://serpapi.com/).
2. Pass the API key in API requests or store it securely in your environment variables.
3. Implement API calls within your AI application to retrieve structured search results.

Please note that this file will also search for these credentials in an `.env` file at the project's root directory.


## Code Examples
```py title="GoogleRetriever Example"
from axion.search import GoogleRetriever

api_client = GoogleRetriever(num_web_results=3)
results = await api_client.execute('What are Ciena Corp challenges?')
for result in results:
    print(result)
```


---

<div class="ref-nav" markdown="1">

[Search API Reference :octicons-arrow-right-24:](../../reference/search.md){ .md-button .md-button--primary }

</div>
