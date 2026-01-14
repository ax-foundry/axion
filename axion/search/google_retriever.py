from typing import Any, Dict, List, Optional

from axion.search.base import BaseRetriever
from axion.search.extract_text import resolve_text_from_result
from axion.search.schema import SearchNode, SearchResults
from axion._core.environment import resolve_api_key
from axion._core.logging import get_logger
from axion._core.tracing import trace
from axion._core.utils import Timer

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

logger = get_logger(__name__)


def _check_search_available():
    if GoogleSearch is None:
        raise ImportError(
            'Search dependencies not installed. '
            'Install with: pip install axion[search]'
        )


class GoogleRetriever(BaseRetriever):
    """
    Retriever that uses SerpAPI to perform Google searches and format the results into nodes.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        num_web_results: int = 5,
        crawl_pages: bool = False,
        max_crawl_tokens: Optional[int] = 10000,
        **kwargs,
    ) -> None:
        """
        Initialize the GoogleRetriever.

        Args:
            api_key (Optional[str]): SerpAPI key used for authenticating requests.
                Defaults to the value of the 'SERPAPI_KEY' environment variable if not provided.
            num_web_results (int): Number of top search results to return (maximum 20).
            crawl_pages (bool): Whether to fetch and clean full page content from URLs in the search results.
            max_crawl_tokens (Optional[int]): Maximum number of tokens to crawl per page (if crawling is enabled).
        """
        _check_search_available()
        self.num_web_results = num_web_results
        self._api_key = resolve_api_key(api_key, 'serpapi_key')
        self.crawl_pages = crawl_pages
        self.max_crawl_tokens = max_crawl_tokens
        super().__init__(**kwargs)

    @trace(name='generate_params', capture_args=True, capture_result=True)
    def _generate_params(self, query: str) -> Dict[str, Any]:
        """
        Generate parameters for the SerpAPI query.

        Args:
            The search query string.

        Returns:
            Dictionary of parameters to send to SerpAPI.
        """
        return {
            'engine': 'google',
            'api_key': self._api_key,
            'q': query,
            'num': self.num_web_results,
        }

    @trace
    async def retrieve(self, query: str) -> SearchResults:
        """
        Perform a search query and return results.

        Args:
            query: Query to search.

        Returns:
            A list of nodes with associated scores.
        """
        if not query:
            raise ValueError('Query string cannot be empty.')

        logger.info(
            f"Performing web search for query: '{query}' with top {self.num_web_results} results."
        )

        with Timer() as timer:
            response = GoogleSearch(self._generate_params(query)).get_dict()
        latency = timer.elapsed_time
        organic_results = response.get('organic_results', [])

        nodes: List[SearchNode] = [
            SearchNode(
                text=resolve_text_from_result(
                    result,
                    crawl=self.crawl_pages,
                    max_crawl_tokens=self.max_crawl_tokens,
                ),
                source=result.get('link'),
                date=result.get('age'),
                highlights=result.get('snippet_highlighted_words', []),
            )
            for result in organic_results
        ]

        if not nodes:
            logger.warning(f"No organic results found for query: '{query}'")

        return self.nodes_to_list(nodes, latency=latency)
