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
from axion.metrics.internals.schema import (
    EvaluationMode,
    JudgedContextChunk,
)
from axion.metrics.schema import SignalDescriptor


class ContextualRelevancyResult(RichBaseModel):
    """Structured result for the ContextualRelevancy metric."""

    overall_score: float
    total_chunks: int
    relevant_chunks: int
    judged_chunks: List[JudgedContextChunk]
    analysis_metadata: Optional[dict] = None


@metric(
    name='Contextual Relevancy',
    key='contextual_relevancy',
    description="Evaluates if the retrieved context is relevant to the user's query.",
    required_fields=['query', 'retrieved_content'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['knowledge', 'single_turn'],
)
class ContextualRelevancy(BaseMetric):
    """Measures the relevancy of retrieval (how much retrieved content is relevant)."""

    shares_internal_cache = True
    required_components = ['chunk_relevancy']

    def __init__(self, mode: EvaluationMode = EvaluationMode.GRANULAR, **kwargs):
        super().__init__(**kwargs)
        self.engine = RAGAnalyzer(mode=mode, **kwargs)

    @trace(name='execute', capture_args=True, capture_response=True)
    async def execute(
        self, item: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricEvaluationResult:
        self._validate_required_metric_fields(item)
        analysis = await self.engine.execute(item, cache, self.required_components)

        if analysis.has_error():
            return MetricEvaluationResult(
                score=np.nan, explanation=f'Analysis failed: {analysis.error}'
            )

        chunks = analysis.judged_context_chunks
        if not chunks:
            return MetricEvaluationResult(
                score=0.0, explanation='No context provided to evaluate relevancy.'
            )

        relevant_count = sum(1 for chunk in chunks if chunk.is_relevant_to_query)
        score = relevant_count / len(chunks)

        result_data = ContextualRelevancyResult(
            overall_score=score,
            total_chunks=len(chunks),
            relevant_chunks=relevant_count,
            judged_chunks=chunks,
        )

        if analysis.metadata:
            self.cost_estimate = getattr(analysis.metadata, 'cost_estimate', 0)

        summary = f'{relevant_count}/{len(chunks)} retrieved chunks were relevant to the query.'

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
        self, result: ContextualRelevancyResult
    ) -> List[SignalDescriptor[ContextualRelevancyResult]]:
        """Defines the explainable signals for the ContextualRelevancy metric."""
        signals = [
            SignalDescriptor(
                name='relevancy_score',
                extractor=lambda r: r.overall_score,
                description='The overall relevancy score (relevant_chunks / total_chunks).',
                headline_display=True,
            ),
            SignalDescriptor(
                name='total_chunks',
                extractor=lambda r: r.total_chunks,
                description='Total number of context chunks retrieved.',
            ),
            SignalDescriptor(
                name='relevant_chunks',
                extractor=lambda r: r.relevant_chunks,
                description='Number of retrieved chunks judged to be relevant to the query.',
            ),
        ]

        for i, chunk in enumerate(result.judged_chunks):
            chunk_preview = (
                chunk.chunk_text[:80] + '...'
                if len(chunk.chunk_text) > 80
                else chunk.chunk_text
            )
            group_name = f'chunk_{i}: "{chunk_preview}"'

            signals.extend(
                [
                    SignalDescriptor(
                        name='is_relevant',
                        group=group_name,
                        extractor=lambda r, idx=i: r.judged_chunks[
                            idx
                        ].is_relevant_to_query,
                        description='The verdict on whether this context chunk is relevant.',
                        headline_display=True,
                        score_mapping={True: 1.0, False: 0.0},
                    ),
                    SignalDescriptor(
                        name='chunk_text',
                        group=group_name,
                        extractor=lambda r, idx=i: r.judged_chunks[idx].chunk_text,
                        description='The full text of the retrieved context chunk.',
                    ),
                ]
            )

        return signals
