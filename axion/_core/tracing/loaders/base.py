"""
Base classes and schemas for trace loaders.

This module provides the foundation for loading traces from observability
platforms (Langfuse, Opik, Logfire) and converting them to Axion Dataset format.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel

if TYPE_CHECKING:
    from axion.dataset import DatasetItem
    from axion.schema import EvaluationResult

logger = get_logger(__name__)

__all__ = [
    'FetchedTraceData',
    'BaseTraceLoader',
]


class FetchedTraceData(RichBaseModel):
    """
    Standardized schema for traces fetched from observability platforms.

    This model aligns with Axion DatasetItem fields to enable seamless
    conversion for evaluation workflows.

    Attributes:
        id: The original trace ID (used as DatasetItem.id)
        query: User query or input (maps to DatasetItem.query)
        actual_output: Model response (maps to DatasetItem.actual_output)
        retrieved_content: Retrieved documents/chunks (maps to DatasetItem.retrieved_content)
        trace_id: Trace ID for scoring back to the platform
        observation_id: Specific span/observation ID for granular scoring
        latency: Response time in seconds
        tags: Platform tags associated with the trace
        timestamp: When the trace was created
        additional_metadata: Platform-specific metadata (e.g., span IDs)
    """

    id: str = Field(..., description='The original trace ID')
    query: str = Field(..., description='User query or input')
    actual_output: str = Field(..., description='Model response or output')
    retrieved_content: List[str] = Field(
        default_factory=list, description='Retrieved documents/chunks'
    )

    # Tracing fields for score upload
    trace_id: str = Field(..., description='Trace ID for scoring')
    observation_id: Optional[str] = Field(
        None, description='Specific span/observation ID for granular scoring'
    )

    # Additional fields
    latency: Optional[float] = Field(None, description='Response time in seconds')
    tags: List[str] = Field(default_factory=list, description='Platform tags')
    timestamp: Optional[datetime] = Field(None, description='Trace creation time')
    additional_metadata: Dict[str, Any] = Field(
        default_factory=dict, description='Platform-specific metadata'
    )

    def to_dataset_item(self) -> 'DatasetItem':
        """
        Convert to Axion DatasetItem for evaluation.

        Returns:
            DatasetItem instance ready for evaluation
        """
        from axion.dataset import DatasetItem

        return DatasetItem(
            id=self.id,
            query=self.query,
            actual_output=self.actual_output,
            retrieved_content=self.retrieved_content,
            trace_id=self.trace_id,
            observation_id=self.observation_id,
            latency=self.latency,
            metadata=(
                json.dumps(self.additional_metadata)
                if self.additional_metadata
                else None
            ),
        )


class BaseTraceLoader(ABC):
    """
    Abstract Base Class for trace loaders.

    Provides common functionality for loading traces from observability
    platforms and converting them to Axion Dataset format.

    Subclasses must implement:
        - fetch_traces(): Platform-specific trace fetching logic
        - push_scores(): Push evaluation scores back to the platform
    """

    @abstractmethod
    def push_scores(
        self,
        evaluation_result: 'EvaluationResult',
        observation_id_field: Optional[str] = 'observation_id',
        flush: bool = True,
    ) -> Dict[str, int]:
        """
        Push evaluation scores back to the observability platform.

        Args:
            evaluation_result: The EvaluationResult from evaluation_runner
            observation_id_field: Field name on DatasetItem containing the
                observation ID for granular scoring. If None, scores attach
                to the trace level.
            flush: Whether to flush the client after uploading

        Returns:
            Dict with counts: {'uploaded': N, 'skipped': M}
        """
        pass

    @abstractmethod
    def fetch_traces(
        self,
        limit: int = 50,
        days_back: int = 7,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Fetch raw traces from the observability platform.

        Args:
            limit: Maximum number of traces to fetch
            days_back: Number of days to look back
            tags: Filter by specific tags
            name: Filter by trace name
            **kwargs: Provider-specific filters or options

        Returns:
            List of raw trace objects from the platform
        """
        pass

    def _normalize_time_window(self, days_back: int) -> datetime:
        """Get standardized start time for filtering."""
        return datetime.now() - timedelta(days=days_back)

    def _safe_json_load(self, data: Any) -> Any:
        """Safely parse stringified JSON."""
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return data
        return data

    def _extract_query(self, input_data: Any) -> str:
        """
        Extract query string from various input formats.

        Handles strings, dicts with common query keys, and fallback to JSON.
        """
        data = self._safe_json_load(input_data)

        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in [
                'query',
                'question',
                'input',
                'message',
                'prompt',
                'user_input',
                'text',
            ]:
                if key in data:
                    return str(data[key])
            return json.dumps(data)
        return str(data)

    def _extract_output(self, output_data: Any) -> str:
        """
        Extract output string from various output formats.

        Handles strings, dicts with common output keys, and fallback to JSON.
        """
        data = self._safe_json_load(output_data)

        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in [
                'output',
                'response',
                'answer',
                'result',
                'content',
                'text',
                'message',
            ]:
                if key in data:
                    return str(data[key])
            return json.dumps(data)
        return str(data)

    def _extract_text_list(self, output: Any) -> List[str]:
        """
        Extract text list from diverse context formats.

        Handles LangChain Documents, JSON strings, dicts, and raw strings.
        """
        data = self._safe_json_load(output)
        texts = []

        if not isinstance(data, list):
            data = [data]

        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                # Common keys for RAG documents
                for key in ['page_content', 'text', 'content', 'document']:
                    if key in item:
                        texts.append(str(item[key]))
                        break
                else:
                    texts.append(json.dumps(item))
            else:
                texts.append(str(item))

        return texts
