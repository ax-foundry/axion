import re
from datetime import datetime, timedelta
from typing import List, Optional

from axion._core.asyncio import run_async_function
from axion._core.metadata.schema import ToolMetadata
from axion._core.tracing import init_tracer
from axion._core.tracing.handlers import BaseTraceHandler
from axion.search.schema import SearchNode, SearchResults


class BaseRetriever:
    """
    Mixin with retrieval utilities for converting nodes, processing text-based dates,
    and calling the async retriever engine.
    """

    def __init__(self, tracer: Optional[BaseTraceHandler] = None, **kwargs):
        self.tracer = init_tracer('base', self.get_tool_metadata(), tracer)

    def get_tool_metadata(self):
        return ToolMetadata(
            name=self.__class__.__name__,
            description=f'Search {self.__class__.__name__} API client',
            owner='AXION',
            version='1.0.0',
        )

    @staticmethod
    def nodes_to_list(
        nodes: List[SearchNode],
        query: str = None,
        answer: str = None,
        images: List = None,
        latency: float = None,
    ) -> SearchResults:
        """Convert RetrieverNodes into SearchResults format"""
        return SearchResults(
            nodes=nodes,
            query=query,
            answer=answer,
            images=images or [],
            latency=latency,
        )

    @staticmethod
    def extract_date_from_text(text: str, fmt: str = '%B %d, %Y') -> Optional[str]:
        """
        Parse relative date descriptions like '2 weeks ago' into formatted date strings.

        Returns:
            str or None: Parsed date string or None if parsing fails.
        """
        if not text:
            return None

        match = re.search(r'(\d+)\s*(weeks?|months?|years?)', text, re.IGNORECASE)
        if not match:
            return text

        try:
            num = int(match.group(1))
            unit = match.group(2).lower()

            if 'week' in unit:
                dt = datetime.now() - timedelta(weeks=num)
            elif 'month' in unit:
                dt = datetime.now() - timedelta(days=30 * num)
            elif 'year' in unit:
                dt = datetime.now() - timedelta(days=365 * num)
            else:
                return text

            return dt.strftime(fmt)
        except Exception:
            return None

    async def execute(self, query: str) -> SearchResults:
        """
        Execute async retrieval using retrieve interface.

        Returns:
            SearchResults: Retrieved nodes with information.
        """
        async with self.tracer.acontext():
            return await self.retrieve(query)

    def execute_sync(self, query: str) -> SearchResults:
        """
        Execute sync retrieval using retrieve interface.

        Returns:
            SearchResults: Retrieved nodes with information.
        """
        return run_async_function(self.execute, query)
