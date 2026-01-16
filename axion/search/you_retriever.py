import warnings
from typing import Any, Dict, List, Literal, Optional

import requests

from axion._core.environment import resolve_api_key
from axion._core.logging import get_logger
from axion._core.tracing import trace
from axion._core.utils import Timer
from axion.search.base import BaseRetriever
from axion.search.extract_text import resolve_text_from_result
from axion.search.schema import SearchNode, SearchResults

logger = get_logger(__name__)


class YouRetriever(BaseRetriever):
    """
    Retriever for You.com's Search and News API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Literal['search', 'news'] = 'search',
        num_web_results: Optional[int] = 5,
        crawl_pages: bool = False,
        max_crawl_tokens: Optional[int] = 10000,
        safesearch: Optional[Literal['off', 'moderate', 'strict']] = None,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        spellcheck: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the YouRetriever.

        Args:
            api_key (Optional[str]): You.com API key. If not provided, it will attempt to use the
                `YDC_API_KEY` environment variable.
            callback_manager (Optional[CallbackManager]): Optional manager for handling callback events during retrieval.
            endpoint (Literal['search', 'news']): The You.com API endpoint to query — either "search" for web results
                or "news" for news-specific content. Defaults to "search".
            num_web_results (Optional[int]): Maximum number of search results to return. Must not exceed 20.
            crawl_pages (bool): Whether to crawl and extract the content of the linked pages from the search results.
            max_crawl_tokens (Optional[int]): Maximum number of tokens to retrieve per page when crawling. If None,
                a default internal value is used.
            safesearch (Optional[Literal['off', 'moderate', 'strict']]): Safe search filtering level. Defaults to
                "moderate" if not specified.
            country (Optional[str]): Country code for geo-specific search behavior (e.g., "US" for United States).
            search_lang (Optional[str]): Language code to use for the search query (e.g., "en" for English).
            ui_lang (Optional[str]): Language code for the UI/localized response (e.g., "en").
            spellcheck (Optional[bool]): Whether to enable spell check for the query. Defaults to True if unspecified.
        """

        self._api_key = resolve_api_key(api_key, 'ydc_api_key')

        if endpoint not in ('search', 'news'):
            raise ValueError('`endpoint` must be either "search" or "news"')

        if endpoint != 'news':
            news_api_fields = (search_lang, ui_lang, spellcheck)
            for field in news_api_fields:
                if field:
                    warnings.warn(
                        f"News API-specific field '{field}' is set but `{endpoint=}`. This will have no effect.",
                        UserWarning,
                    )

        self.endpoint = endpoint
        self.num_web_results = num_web_results
        self.crawl_pages = crawl_pages
        self.max_crawl_tokens = max_crawl_tokens
        self.safesearch = safesearch
        self.country = country
        self.search_lang = search_lang
        self.ui_lang = ui_lang
        self.spellcheck = spellcheck

        super().__init__(**kwargs)

    @trace(name='payload', capture_args=True, capture_result=True)
    def _generate_params(self, query: str) -> Dict[str, Any]:
        """Generate API request parameters."""
        params = {'safesearch': self.safesearch, 'country': self.country}

        if self.endpoint == 'search':
            params.update(query=query, num_web_results=self.num_web_results)
        elif self.endpoint == 'news':
            params.update(
                q=query,
                count=self.num_web_results,
                search_lang=self.search_lang,
                ui_lang=self.ui_lang,
                spellcheck=self.spellcheck,
            )
        return {k: v for k, v in params.items() if v is not None}

    @trace(name='build_nodes', capture_args=True)
    def _build_nodes_from_articles(
        self, articles: List[Dict[str, Any]]
    ) -> List[SearchNode]:
        """
        Build Nodes from articles.
        """
        nodes = []
        for article in articles:
            text = resolve_text_from_result(
                article, crawl=self.crawl_pages, max_crawl_tokens=self.max_crawl_tokens
            )
            nodes.append(
                SearchNode(
                    text=text,
                    source=article.get('url'),
                    date=article.get('age'),
                )
            )
        return nodes

    @trace
    async def retrieve(self, query: str) -> SearchResults:
        """
        Perform a search query and return results using You.com API.

        Args:
            query: Query to search.

        Returns:
            A list of nodes with associated scores.
        """
        if not query:
            raise ValueError('Query string cannot be empty.')

        logger.info(
            f"Performing web search for query: '{query}' using endpoint '{self.endpoint}'."
        )

        headers = {'X-API-Key': self._api_key}
        params = self._generate_params(query)
        with Timer() as timer:
            response = requests.get(
                f'https://api.ydc-index.io/{self.endpoint}',
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            results = response.json()
        latency = timer.elapsed_time

        nodes: List[SearchNode] = []

        if self.endpoint == 'search':
            nodes = self._build_nodes_from_articles(results.get('hits', []))
        elif self.endpoint == 'news':
            nodes = self._build_nodes_from_articles(
                results.get('news', {}).get('results', [])
            )
        if not nodes:
            logger.warning(f'No results found for query: {query}')

        return self.nodes_to_list(nodes, latency=latency)
