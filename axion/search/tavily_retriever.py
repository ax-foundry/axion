from typing import Any, Dict, List, Literal, Optional

import requests

from axion._core.environment import resolve_api_key
from axion._core.error import CustomValidationError
from axion._core.logging import get_logger
from axion._core.tracing import trace
from axion.search.base import BaseRetriever
from axion.search.schema import SearchNode, SearchResults

logger = get_logger(__name__)


class TavilyRetriever(BaseRetriever):
    """
    Retriever for Tavily's Search, Extract, and Crawl APIs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Literal['search', 'extract', 'crawl'] = 'search',
        search_depth: Literal['basic', 'advanced'] = 'basic',
        topic: Optional[str] = 'general',
        max_results: Optional[int] = 5,
        crawl_pages: bool = False,
        max_crawl_tokens: Optional[int] = 10000,
        days: Optional[int] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        include_image_descriptions: bool = False,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        extract_depth: Literal['basic', 'advanced'] = 'basic',
        max_depth: Optional[int] = 1,
        max_breadth: Optional[int] = 20,
        limit: Optional[int] = 50,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        allow_external: bool = False,
        categories: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the TavilyRetriever with specified parameters.
        """
        self._api_key = resolve_api_key(api_key, 'tavily_api_key')

        if endpoint not in ('search', 'extract', 'crawl'):
            raise ValueError('`endpoint` must be "search", "extract", or "crawl"')

        self.endpoint = endpoint
        self.search_depth = search_depth
        self.topic = topic
        self.max_results = max_results
        self.crawl_pages = crawl_pages
        self.max_crawl_tokens = max_crawl_tokens
        self.days = days
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.include_image_descriptions = include_image_descriptions
        self.include_domains = include_domains or []
        self.exclude_domains = exclude_domains or []

        self.extract_depth = extract_depth

        self.max_depth = max_depth
        self.max_breadth = max_breadth
        self.limit = limit
        self.instructions = instructions
        self.select_paths = select_paths
        self.select_domains = select_domains
        self.exclude_paths = exclude_paths
        self.allow_external = allow_external
        self.categories = categories

        super().__init__(**kwargs)

    @trace(name='search_payload', capture_args=True)
    def _generate_search_payload(self, query: str) -> Dict[str, Any]:
        """Generate payload for the Search API."""
        payload = {
            'query': query,
            'topic': self.topic,
            'search_depth': self.search_depth,
            'max_results': self.max_results,
            'days': self.days,
            'include_answer': self.include_answer,
            'include_raw_content': self.include_raw_content,
            'include_images': self.include_images,
            'include_image_descriptions': self.include_image_descriptions,
            'include_domains': self.include_domains,
            'exclude_domains': self.exclude_domains,
        }
        return {k: v for k, v in payload.items() if v is not None}

    @trace(name='generate_payload', capture_args=True, capture_result=True)
    def _generate_extract_payload(self, urls: List[str]) -> Dict[str, Any]:
        """Generate payload for the Extract API."""
        payload = {
            'urls': urls,
            'include_images': self.include_images,
            'extract_depth': self.extract_depth,
        }
        return payload

    def _get_headers(self) -> Dict:
        return {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        }

    @trace(name='generate_payload', capture_args=True, capture_result=True)
    def _generate_crawl_payload(self, url: str) -> Dict[str, Any]:
        """Generate payload for the Crawl API."""
        payload = {
            'url': url,
            'max_depth': self.max_depth,
            'max_breadth': self.max_breadth,
            'limit': self.limit,
            'instructions': self.instructions,
            'select_paths': self.select_paths,
            'select_domains': self.select_domains,
            'exclude_paths': self.exclude_paths,
            'exclude_domains': self.exclude_domains,
            'allow_external': self.allow_external,
            'include_images': self.include_images,
            'categories': self.categories,
            'extract_depth': self.extract_depth,
        }
        return {k: v for k, v in payload.items() if v is not None}

    @trace
    async def retrieve(self, query: str) -> SearchResults:
        """
        Retrieve results using the Tavily Search API.
        Only handles 'search' endpoint.
        For 'extract' and 'crawl', call their respective methods directly.
        """
        if self.endpoint != 'search':
            raise ValueError(
                "The _retrieve method only supports the 'search' endpoint."
            )

        if not query:
            raise ValueError('Query string cannot be empty.')

        headers = self._get_headers()
        url = 'https://api.tavily.com/search'
        payload = self._generate_search_payload(query)
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()

        nodes: List[SearchNode] = []

        query = results.get('query', '')
        answer = results.get('answer', '')
        images = results.get('images', [])
        latency = results.get('response_time', None)
        for result in results.get('results', []):
            source = result.get('url', '')
            score = result.get('score', None)
            text = result.get('content', '')
            if self.crawl_pages:
                try:
                    text = self.extract_url_text(source)
                except Exception:
                    pass

            date = result.get('date', None)
            nodes.append(
                SearchNode(text=text or '', source=source, date=date, score=score)
            )

        if not nodes:
            logger.warning(f'No results found for query: {query}')

        return self.nodes_to_list(
            nodes=nodes,
            query=query,
            answer=answer,
            images=images,
            latency=latency,
        )

    @trace(name='extract_url_text', capture_args=True)
    def extract_url_text(self, url: str) -> str:
        """Extract content from a single URL using the Extract API."""
        headers = self._get_headers()
        api_url = 'https://api.tavily.com/extract'
        payload = self._generate_extract_payload([url])
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()

        return results['results'][0].get('raw_content', '')

    @trace(name='extract', capture_args=True, capture_result=True)
    async def extract(self, url: str) -> SearchResults:
        """Extract content from a URL using the Extract API."""
        headers = self._get_headers()
        api_url = 'https://api.tavily.com/extract'
        payload = self._generate_extract_payload([url])
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()

        nodes: List[SearchNode] = []
        latency = results.get('response_time', None)
        for result in results.get('results', []):
            source = result.get('url', None)
            images = result.get('images', [])
            text = result.get('raw_content', '')

            nodes.append(
                SearchNode(
                    text=text,
                    source=source,
                    images=images,
                )
            )
        return self.nodes_to_list(nodes=nodes, latency=latency)

    @trace(name='crawl', capture_args=True, capture_result=True)
    async def crawl(self, url: str) -> SearchResults:
        """Crawl content from a URL using the Crawl API."""
        headers = self._get_headers()
        api_url = 'https://api.tavily.com/crawl'
        payload = self._generate_crawl_payload(url)
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()

        nodes: List[SearchNode] = []

        query = results.get('base_url', '')
        latency = results.get('response_time', None)
        for result in results.get('results', []):
            source = result.get('url', None)
            text = result.get('raw_content', '')
            try:
                nodes.append(
                    SearchNode(
                        text=text,
                        source=source,
                    )
                )
            except (CustomValidationError, ValueError):
                nodes.append(
                    SearchNode(
                        text='',
                        source='',
                    )
                )

        return self.nodes_to_list(nodes=nodes, query=query, latency=latency)
