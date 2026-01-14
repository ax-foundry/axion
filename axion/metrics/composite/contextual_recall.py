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
    JudgedGroundTruthStatement,
)
from axion.metrics.schema import SignalDescriptor


class ContextualRecallResult(RichBaseModel):
    """Structured result for the ContextualRecall metric."""

    overall_score: float
    total_statements: int
    supported_statements: int
    judged_statements: List[JudgedGroundTruthStatement]
    analysis_metadata: Optional[dict] = None


@metric(
    name='ContextualRecall',
    key='contextual_recall',
    description='Evaluates if the retrieved context supports the expected answer.',
    required_fields=['expected_output', 'retrieved_content'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['knowledge', 'single_turn'],
)
class ContextualRecall(BaseMetric):
    """Measures recall of retrieval (how much of the expected answer is in context)."""

    shares_internal_cache = True
    required_components = ['gt']

    def __init__(self, mode: EvaluationMode = EvaluationMode.GRANULAR, **kwargs):
        super().__init__(**kwargs)
        self.engine = RAGAnalyzer(mode=mode, **kwargs)

    @trace(name='ContextualRecall.execute', capture_args=True, capture_response=True)
    async def execute(
        self, item: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricEvaluationResult:
        self._validate_required_metric_fields(item)
        analysis = await self.engine.execute(item, cache, self.required_components)

        if analysis.has_error():
            return MetricEvaluationResult(
                score=np.nan, explanation=f'Analysis failed: {analysis.error}'
            )

        statements = analysis.judged_ground_truth_statements
        if not statements:
            return MetricEvaluationResult(
                score=np.nan,
                explanation='No expected output provided to evaluate recall against.',
            )

        supported_count = sum(1 for s in statements if s.is_supported_by_context)
        score = supported_count / len(statements)

        result_data = ContextualRecallResult(
            overall_score=score,
            total_statements=len(statements),
            supported_statements=supported_count,
            judged_statements=statements,
        )

        if analysis.metadata:
            self.cost_estimate = getattr(analysis.metadata, 'cost_estimate', 0)

        summary = f'{supported_count}/{len(statements)} required statements were supported by the context.'

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
        self, result: ContextualRecallResult
    ) -> List[SignalDescriptor[ContextualRecallResult]]:
        """Defines the explainable signals for the ContextualRecall metric."""
        signals = [
            SignalDescriptor(
                name='recall_score',
                extractor=lambda r: r.overall_score,
                description='The overall recall score (supported_statements / total_statements).',
                headline_display=True,
            ),
            SignalDescriptor(
                name='total_gt_statements',
                extractor=lambda r: r.total_statements,
                description='Total number of factual statements extracted from the expected answer.',
            ),
            SignalDescriptor(
                name='supported_gt_statements',
                extractor=lambda r: r.supported_statements,
                description='Number of ground truth statements supported by the retrieved context.',
            ),
        ]

        for i, stmt in enumerate(result.judged_statements):
            stmt_preview = (
                stmt.statement_text[:80] + '...'
                if len(stmt.statement_text) > 80
                else stmt.statement_text
            )
            group_name = f'gt_statement_{i}: "{stmt_preview}"'

            signals.extend(
                [
                    SignalDescriptor(
                        name='is_supported',
                        group=group_name,
                        extractor=lambda r, idx=i: r.judged_statements[
                            idx
                        ].is_supported_by_context,
                        description='The verdict on whether this ground truth statement is supported by the context.',
                        headline_display=True,
                        score_mapping={True: 1.0, False: 0.0},
                    ),
                    SignalDescriptor(
                        name='statement_text',
                        group=group_name,
                        extractor=lambda r, idx=i: r.judged_statements[
                            idx
                        ].statement_text,
                        description='The full text of the ground truth statement.',
                    ),
                ]
            )

        return signals
