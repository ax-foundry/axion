"""
Opik trace loader for Axion.

This module provides functionality to fetch traces from Opik
and convert them to Axion Dataset format.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from axion._core.logging import get_logger
from axion._core.tracing.loaders.base import BaseTraceLoader

if TYPE_CHECKING:
    from axion.schema import EvaluationResult

logger = get_logger(__name__)

__all__ = ['OpikTraceLoader']


class OpikTraceLoader(BaseTraceLoader):
    """
    Trace loader for Opik observability platform.

    Note: This is a placeholder implementation. Full implementation
    will be added when Opik integration is required.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ):
        """
        Initialize Opik client.

        Args:
            api_key: Opik API key (falls back to OPIK_API_KEY env var)
            workspace: Opik workspace (falls back to OPIK_WORKSPACE env var)
        """
        self.api_key = api_key
        self.workspace = workspace
        self._client_initialized = False
        logger.warning('OpikTraceLoader is not yet implemented')

    def push_scores(
        self,
        evaluation_result: 'EvaluationResult',
        observation_id_field: Optional[str] = 'observation_id',
        flush: bool = True,
    ) -> Dict[str, int]:
        """
        Push evaluation scores back to Opik.

        Note: Not yet implemented.
        """
        return self.push_scores_to_opik(evaluation_result, flush=flush)

    def fetch_traces(
        self,
        limit: int = 50,
        days_back: int = 7,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> List[Any]:
        """
        Fetch raw traces from Opik.

        Note: Not yet implemented.
        """
        raise NotImplementedError(
            'OpikTraceLoader.fetch_traces() is not yet implemented. '
            'Please contribute an implementation or use LangfuseTraceLoader.'
        )

    def push_scores_to_opik(
        self,
        evaluation_result: 'EvaluationResult',
        flush: bool = True,
    ) -> Dict[str, int]:
        """
        Push evaluation scores back to Opik traces.

        Note: Not yet implemented.
        """
        raise NotImplementedError(
            'OpikTraceLoader.push_scores_to_opik() is not yet implemented.'
        )
