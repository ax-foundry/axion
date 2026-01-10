from typing import Any, List, Optional

from axion._core.schema import RichBaseModel
from pydantic import Field


class SearchNode(RichBaseModel):
    """
    Extended node class for retrieval, supporting metadata formatting,
    templated text representation, and additional attributes like highlights, source, and date.
    """

    text: str = Field(default='', description='Text content of the node.')
    highlights: List[str] = Field(
        default_factory=list, description='List of node highlights.'
    )
    source: str = Field(default=None, description='Source of the content.')
    date: str = Field(default=None, description='Date associated with the content.')
    score: Optional[float] = Field(default=None, description='Score for the content.')
    images: Optional[List[Any]] = Field(
        default_factory=list, description='List of images related to the result.'
    )

    def __init__(
        self,
        *args: Any,
        highlights: Optional[List[str]] = None,
        source: Optional[str] = None,
        date: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.highlights = highlights or []
        self.source = source or ''
        self.date = date or ''


class SearchResults(RichBaseModel):
    """
    A structured result from a retrieval operation, containing a node, score,
    original query, generated answer, and optionally any image references.
    """

    nodes: List[SearchNode] = Field(
        default_factory=list,
        description='The retrieved node containing text and metadata.',
    )
    query: Optional[str] = Field(
        default=None, description='The original query that produced this result.'
    )
    answer: Optional[str] = Field(
        default=None, description='Generated answer or summary for the result.'
    )
    images: Optional[List[Any]] = Field(
        default_factory=list, description='List of images related to the result.'
    )
    latency: Optional[float] = Field(
        default=None, description='Response time for the request.'
    )

    def __len__(self):
        """Length of SearchNodes"""
        return len(self.nodes)

    def __iter__(self):
        """Allow iteration over the SearchNodes in the result."""
        return iter(self.nodes)
