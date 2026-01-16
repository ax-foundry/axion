from typing import List, Optional

import numpy as np

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.cache import AnalysisCache
from axion.metrics.internals.engine import RAGAnalyzer
from axion.metrics.internals.schema import EvaluationMode
from axion.metrics.schema import SignalDescriptor


class ContextSufficiencyResult(RichBaseModel):
    """Structured result for the ContextSufficiency metric."""

    overall_score: float
    is_sufficient: bool
    reasoning: Optional[str] = None
    context_preview: Optional[str] = None
    query_preview: Optional[str] = None
    analysis_metadata: Optional[dict] = None


@metric(
    name='Contextual Sufficiency',
    key='contextual_sufficiency',
    description='Evaluates whether the retrieved context contains sufficient information to answer the query.',
    required_fields=['query', 'retrieved_content'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['knowledge', 'single_turn'],
)
class ContextualSufficiency(BaseMetric):
    """
    Measures whether the retrieved context is sufficient to answer the user's query.
    This is a diagnostic metric that evaluates the quality of retrieval independent
    of the generated answer.

    Score of 1.0: Context contains all necessary information to answer the query
    Score of 0.0: Context is insufficient to answer the query
    """

    shares_internal_cache = True
    required_components = ['context_sufficiency']

    def __init__(self, mode: EvaluationMode = EvaluationMode.GRANULAR, **kwargs):
        super().__init__(**kwargs)
        self.engine = RAGAnalyzer(mode=mode, **kwargs)

    @trace(name='ContextualSufficiency', capture_args=True, capture_response=True)
    async def execute(
        self, item: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricEvaluationResult:
        self._validate_required_metric_fields(item)
        analysis = await self.engine.execute(item, cache, self.required_components)

        if analysis.has_error():
            return MetricEvaluationResult(
                score=np.nan, explanation=f'Analysis failed: {analysis.error}'
            )

        # Get sufficiency verdict from analysis
        sufficiency_data = getattr(analysis, 'context_sufficiency_verdict', None)

        if sufficiency_data is None:
            return MetricEvaluationResult(
                score=0.0,
                explanation='Context sufficiency could not be evaluated.',
            )

        is_sufficient = getattr(sufficiency_data, 'is_sufficient', False)
        reasoning = getattr(sufficiency_data, 'reasoning', None)

        # Binary score: 1.0 if sufficient, 0.0 if not
        score = 1.0 if is_sufficient else 0.0

        # Create preview of context and query for signals
        context_text = '\n'.join(item.retrieved_content or [])
        context_preview = (
            context_text[:200] + '...' if len(context_text) > 200 else context_text
        )
        query_preview = (
            item.query[:100] + '...' if len(item.query) > 100 else item.query
        )

        result_data = ContextSufficiencyResult(
            overall_score=score,
            is_sufficient=is_sufficient,
            reasoning=reasoning,
            context_preview=context_preview,
            query_preview=query_preview,
        )

        if analysis.metadata:
            self.cost_estimate = getattr(analysis.metadata, 'cost_estimate', 0)

        if is_sufficient:
            summary = 'The retrieved context contains sufficient information to answer the query.'
        else:
            summary = (
                'The retrieved context does NOT contain sufficient information to answer the query. '
                f'Reason: {reasoning or "No specific reason provided."}'
            )

        return MetricEvaluationResult(
            score=score,
            explanation=summary,
            signals=result_data,
            metadata={
                'analysis_metadata': analysis.metadata.model_dump()
                if analysis.metadata
                else {}
            },
        )

    def get_signals(
        self, result: ContextSufficiencyResult
    ) -> List[SignalDescriptor[ContextSufficiencyResult]]:
        """Defines the explainable signals for the ContextSufficiency metric."""
        return [
            SignalDescriptor(
                name='sufficiency_score',
                extractor=lambda r: r.overall_score,
                description='Binary score: 1.0 if context is sufficient, 0.0 if not.',
                headline_display=True,
                score_mapping={'Sufficient': 1.0, 'Insufficient': 0.0},
            ),
            SignalDescriptor(
                name='is_sufficient',
                extractor=lambda r: r.is_sufficient,
                description='Boolean verdict on whether the context is sufficient.',
                headline_display=True,
            ),
            SignalDescriptor(
                name='reasoning',
                extractor=lambda r: r.reasoning or 'N/A',
                description='Explanation of why the context is or is not sufficient.',
            ),
            SignalDescriptor(
                name='query',
                extractor=lambda r: r.query_preview or 'N/A',
                description='Preview of the user query being evaluated.',
            ),
            SignalDescriptor(
                name='context',
                extractor=lambda r: r.context_preview or 'N/A',
                description='Preview of the retrieved context.',
            ),
        ]
