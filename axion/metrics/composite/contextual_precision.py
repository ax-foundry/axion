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


class ContextualPrecisionResult(RichBaseModel):
    """Structured result for the ContextualPrecision metric."""

    overall_score: float
    total_chunks: int
    useful_chunks: int
    first_useful_position: Optional[int]
    judged_chunks: List[JudgedContextChunk]
    analysis_metadata: Optional[dict] = None


@metric(
    name='Contextual Precision',
    key='contextual_precision',
    description='Evaluates whether relevant context chunks are ranked higher (Mean Average Precision).',
    required_fields=['query', 'expected_output', 'retrieved_content'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['knowledge', 'single_turn'],
)
class ContextualPrecision(BaseMetric):
    """Measures the quality of retrieval ranking using Mean Average Precision (MAP)."""

    shares_internal_cache = True
    required_components = ['chunk_usefulness']

    def __init__(self, mode: EvaluationMode = EvaluationMode.GRANULAR, **kwargs):
        super().__init__(**kwargs)
        self.engine = RAGAnalyzer(mode=mode, **kwargs)

    @trace(name='ContextualPrecision', capture_args=True, capture_response=True)
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
                score=0.0, explanation='No context provided to evaluate ranking.'
            )

        # Compute Mean Average Precision (MAP)
        useful_indices = [
            i for i, c in enumerate(chunks, start=1) if c.is_useful_for_expected_output
        ]
        useful_count = len(useful_indices)

        if useful_count:
            precisions = [rank / i for rank, i in enumerate(useful_indices, start=1)]
            score = sum(precisions) / useful_count
            first_useful_pos = useful_indices[0]
        else:
            score = 0.0
            first_useful_pos = None

        # Build structured result
        result_data = ContextualPrecisionResult(
            overall_score=score,
            total_chunks=len(chunks),
            useful_chunks=useful_count,
            first_useful_position=first_useful_pos,
            judged_chunks=chunks,
        )

        if analysis.metadata:
            self.cost_estimate = getattr(analysis.metadata, 'cost_estimate', 0)

        # Human-readable summary
        summary = (
            f'MAP score: {score:.2f} ({useful_count}/{len(chunks)} useful chunks found)'
        )

        # Return evaluation result
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
        self, result: ContextualPrecisionResult
    ) -> List[SignalDescriptor[ContextualPrecisionResult]]:
        """Defines the explainable signals for the ContextualRanking metric."""
        signals = [
            SignalDescriptor(
                name='map_score',
                extractor=lambda r: r.overall_score,
                description='The Mean Average Precision (MAP) score, which rewards ranking useful chunks higher.',
                headline_display=True,
            ),
            SignalDescriptor(
                name='total_chunks',
                extractor=lambda r: r.total_chunks,
                description='Total number of context chunks retrieved.',
            ),
            SignalDescriptor(
                name='useful_chunks',
                extractor=lambda r: r.useful_chunks,
                description='Number of chunks judged useful for generating the expected output.',
            ),
            SignalDescriptor(
                name='first_useful_position',
                extractor=lambda r: r.first_useful_position
                if r.first_useful_position is not None
                else 'N/A',
                description='The rank (position) of the first useful chunk in the retrieved list.',
            ),
        ]

        for i, chunk in enumerate(result.judged_chunks):
            chunk_preview = (
                chunk.chunk_text[:80] + '...'
                if len(chunk.chunk_text) > 80
                else chunk.chunk_text
            )
            group_name = f'chunk_{i + 1}: "{chunk_preview}"'

            signals.extend(
                [
                    SignalDescriptor(
                        name='is_useful',
                        group=group_name,
                        extractor=lambda r, idx=i: r.judged_chunks[
                            idx
                        ].is_useful_for_expected_output,
                        description='The verdict on whether this chunk was useful for generating the answer.',
                        headline_display=True,
                        score_mapping={'True': 1.0, 'False': 0.0},
                    ),
                    SignalDescriptor(
                        name='position',
                        group=group_name,
                        extractor=lambda r, idx=i: idx + 1,
                        description='The rank of this chunk in the retrieved list.',
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
