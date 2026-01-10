#  You.com WebSearch

## Overview
You.com is an AI-powered search engine and conversational AI platform that aims to enhance productivity and provide users with a more personalized and interactive search experience, going beyond traditional search and offering tools like AI agents and chatbots

## API Access
To integrate You.com into your application, obtain an API key and configure it accordingly.


### Steps to Access You.com API
1. Sign up and obtain an API key from [You.com API](https://api.you.com/).
2. Pass the API key in requests or store it securely in your environment variables.
3. Implement API calls within your AI application to fetch real-time search results.

Please note that this file will search for these credentials in an `.env` file at the project's root directory.


## Code Examples
```py title="YouRetriever Example"
from axion.search import YouRetriever

api_client = YouRetriever(num_web_results=5, endpoint='news') # supports both "news" and "search" endpoints
results = await api_client.execute('What are Ciena Corp challenges?')
for result in results:
    print(result)
```



## API Reference
::: axion.search.you_retriever
