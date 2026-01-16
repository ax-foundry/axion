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


class ContextualRankingResult(RichBaseModel):
    """Structured result for the ContextualPrecision metric."""

    overall_score: float
    total_chunks: int
    relevant_chunks: int
    judged_chunks: List[JudgedContextChunk]
    analysis_metadata: Optional[dict] = None


@metric(
    name='Contextual Ranking',
    key='contextual_ranking',
    description='Evaluates if relevant context chunks are ranked higher (closer to the top).',
    required_fields=['query', 'retrieved_content'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['knowledge', 'single_turn'],
)
class ContextualRanking(BaseMetric):
    """
    Measures the ranking of the retrieved context, with a heavy penalty
    for relevant chunks being ranked low.
    """

    shares_internal_cache = True
    required_components = ['chunk_relevancy']

    def __init__(
        self,
        mode: EvaluationMode = EvaluationMode.GRANULAR,
        **kwargs,
    ):
        """
        Initializes the Contextual Precision metric.
        Args:
            mode: The evaluation mode for the underlying RAG analyzer.
            **kwargs: Additional keyword arguments passed to the RAGAnalyzer.
        """
        super().__init__(**kwargs)
        self.engine = RAGAnalyzer(mode=mode, **kwargs)

    @trace(name='ContextualRanking', capture_args=True, capture_response=True)
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
                score=0.0, explanation='No context chunks were retrieved.'
            )

        score = 0.0
        relevant_chunks_count = 0
        cumulative_relevance = 0

        for i, chunk in enumerate(chunks):
            rank = i + 1  # 1-indexed rank
            is_relevant = chunk.is_relevant_to_query

            if is_relevant:
                relevant_chunks_count += 1
                cumulative_relevance += 1
                # This is the (Num of Relevant Nodes Upto position k) / k part
                precision_at_k = cumulative_relevance / rank
                score += precision_at_k

        # Normalize the score by the total number of relevant chunks
        if relevant_chunks_count > 0:
            final_score = score / relevant_chunks_count
        else:
            # No relevant chunks found, precision is 0 (or 1.0 if you prefer)
            final_score = 0.0

        # Clamp score to [0, 1] range
        final_score = max(0.0, min(1.0, final_score))

        result_data = ContextualRankingResult(
            overall_score=final_score,
            total_chunks=len(chunks),
            relevant_chunks=relevant_chunks_count,
            judged_chunks=chunks,
        )

        if analysis.metadata:
            self.cost_estimate = getattr(analysis.metadata, 'cost_estimate', 0)

        summary = self._generate_summary(
            final_score, relevant_chunks_count, len(chunks)
        )

        return MetricEvaluationResult(
            score=final_score,
            explanation=summary,
            signals=result_data,
            metadata={
                'analysis_metadata': analysis.metadata.model_dump()
                if analysis.metadata
                else {}
            },
        )

    @staticmethod
    def _generate_summary(score: float, relevant_count: int, total_count: int) -> str:
        if relevant_count == 0:
            return f'Poor: No relevant chunks were found out of {total_count}.'

        summary = f'Found {relevant_count} relevant chunk(s) out of {total_count}. '
        if score >= 0.9:
            summary += f'Ranking is excellent (Score: {score:.2f}).'
        elif score >= 0.7:
            summary += f'Ranking is good (Score: {score:.2f}).'
        else:
            summary += (
                f'Ranking is poor (Score: {score:.2f}), '
                'relevant chunks are ranked too low.'
            )
        return summary

    def get_signals(self, result: ContextualRankingResult) -> List[SignalDescriptor]:
        signals = [
            SignalDescriptor(
                name='final_score',
                extractor=lambda r: r.overall_score,
                headline_display=True,
            ),
            SignalDescriptor(
                name='relevant_chunks', extractor=lambda r: r.relevant_chunks
            ),
            SignalDescriptor(name='total_chunks', extractor=lambda r: r.total_chunks),
        ]

        for i, chunk in enumerate(result.judged_chunks):
            group_name = f'chunk_{i+1}: "{chunk.chunk_text[:80]}..."'

            signals.extend(
                [
                    SignalDescriptor(
                        name='is_relevant',
                        group=group_name,
                        extractor=lambda r, idx=i: r.judged_chunks[
                            idx
                        ].is_relevant_to_query,
                        description='Whether this chunk is relevant to the query.',
                        headline_display=True,
                        score_mapping={'True': 1.0, 'False': 0.0},
                    ),
                    SignalDescriptor(
                        name='chunk_text',
                        group=group_name,
                        extractor=lambda r, idx=i: r.judged_chunks[idx].chunk_text,
                    ),
                ]
            )
        return signals
