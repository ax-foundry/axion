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


class ContextUtilizationResult(RichBaseModel):
    """Structured result for the ContextUtilization metric."""

    overall_score: float
    total_relevant_chunks: int
    utilized_chunks: int
    utilization_rate: float
    judged_chunks: List[JudgedContextChunk]
    analysis_metadata: Optional[dict] = None


@metric(
    name='Contextual Utilization',
    key='contextual_utilization',
    description='Measures how much of the relevant retrieved context was actually used to generate the answer.',
    required_fields=['query', 'actual_output', 'retrieved_content'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['knowledge', 'single_turn'],
)
class ContextualUtilization(BaseMetric):
    """
    Measures efficiency of context usage. Evaluates what proportion of the relevant
    context chunks were actually utilized in generating the final answer.

    High scores indicate efficient context usage (most relevant chunks were used).
    Low scores indicate waste (many relevant chunks were retrieved but not used).
    """

    shares_internal_cache = True
    required_components = ['chunk_relevancy', 'chunk_utilization']

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
                score=0.0,
                explanation='No context provided to evaluate utilization.',
            )

        # Filter to only relevant chunks (the ones that should potentially be used)
        relevant_chunks = [c for c in chunks if c.is_relevant_to_query]

        if not relevant_chunks:
            # If no chunks are relevant, we can't measure utilization
            return MetricEvaluationResult(
                score=0.0,
                explanation='No relevant chunks found to measure utilization against.',
            )

        # Count how many relevant chunks were actually utilized
        utilized_count = sum(
            1 for c in relevant_chunks if getattr(c, 'is_utilized_in_answer', False)
        )

        # Calculate utilization rate
        utilization_rate = utilized_count / len(relevant_chunks)
        score = utilization_rate

        result_data = ContextUtilizationResult(
            overall_score=score,
            total_relevant_chunks=len(relevant_chunks),
            utilized_chunks=utilized_count,
            utilization_rate=utilization_rate,
            judged_chunks=chunks,
        )

        if analysis.metadata:
            self.cost_estimate = getattr(analysis.metadata, 'cost_estimate', 0)

        summary = (
            f'{utilized_count}/{len(relevant_chunks)} relevant chunks '
            f'were utilized in generating the answer ({utilization_rate:.1%} utilization rate).'
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
        self, result: ContextUtilizationResult
    ) -> List[SignalDescriptor[ContextUtilizationResult]]:
        """Defines the explainable signals for the ContextUtilization metric."""
        signals = [
            SignalDescriptor(
                name='utilization_score',
                extractor=lambda r: r.overall_score,
                description='The overall utilization score (utilized_chunks / total_relevant_chunks).',
                headline_display=True,
            ),
            SignalDescriptor(
                name='total_relevant_chunks',
                extractor=lambda r: r.total_relevant_chunks,
                description='Total number of relevant context chunks retrieved.',
            ),
            SignalDescriptor(
                name='utilized_chunks',
                extractor=lambda r: r.utilized_chunks,
                description='Number of relevant chunks that were actually used in the answer.',
            ),
            SignalDescriptor(
                name='utilization_rate',
                extractor=lambda r: r.utilization_rate,
                description='Percentage of relevant chunks that were utilized.',
            ),
        ]

        # Only show details for relevant chunks
        relevant_chunks = [c for c in result.judged_chunks if c.is_relevant_to_query]

        for i, chunk in enumerate(relevant_chunks):
            chunk_preview = (
                chunk.chunk_text[:80] + '...'
                if len(chunk.chunk_text) > 80
                else chunk.chunk_text
            )
            group_name = f'relevant_chunk_{i}: "{chunk_preview}"'

            signals.extend(
                [
                    SignalDescriptor(
                        name='is_utilized',
                        group=group_name,
                        extractor=lambda r, idx=i: getattr(
                            [c for c in r.judged_chunks if c.is_relevant_to_query][idx],
                            'is_utilized_in_answer',
                            False,
                        ),
                        description='Whether this relevant chunk was utilized in the generated answer.',
                        headline_display=True,
                        score_mapping={True: 1.0, False: 0.0},
                    ),
                    SignalDescriptor(
                        name='chunk_text',
                        group=group_name,
                        extractor=lambda r, idx=i: [
                            c for c in r.judged_chunks if c.is_relevant_to_query
                        ][idx].chunk_text,
                        description='The full text of the relevant context chunk.',
                    ),
                ]
            )

        return signals
