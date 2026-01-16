from typing import List, Literal, Optional

import numpy as np
from axion._core.logging import get_logger
from axion._core.schema import AIMessage, HumanMessage, RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.cache import AnalysisCache
from axion.metrics.internals.engine import RAGAnalyzer
from axion.metrics.internals.judges import RelevancyExplainer
from axion.metrics.internals.schema import EvaluationMode
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

logger = get_logger(__name__)


class StatementBreakdownResult(RichBaseModel):
    statement: str
    verdict: Literal['yes', 'no', 'idk']
    reason: Optional[str] = None
    is_relevant: bool
    turn_index: Optional[int] = Field(
        default=None,
        description='The turn number in a multi-turn evaluation (starts from 0).',
    )


class AnswerRelevancyResult(RichBaseModel):
    """Structured result model for the AnswerRelevancy metric, used for signals."""

    overall_score: float
    explanation: str
    relevant_statements_count: int
    irrelevant_statements_count: int
    ambiguous_statements_count: int
    total_statements_count: int
    statement_breakdown: List[StatementBreakdownResult]
    evaluated_turns_count: Optional[int] = Field(
        default=None,
        description='The number of turns evaluated in a multi-turn conversation.',
    )


@metric(
    name='Answer Relevancy',
    key='answer_relevancy',
    description='Evaluates how relevant an answer is to the input query.',
    required_fields=['query', 'actual_output'],
    optional_fields=['conversation'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['knowledge', 'single_turn', 'multi_turn'],
)
class AnswerRelevancy(BaseMetric):
    """
    Computes answer relevancy scores by analyzing how well the response addresses the input query.
    Supports both single-turn and multi-turn (all turns) evaluation.
    """

    # as_structured_llm = False
    shares_internal_cache = True
    required_components = ['claim_relevancy']

    def __init__(
        self,
        relevancy_mode: Literal['strict', 'task'] = 'task',
        penalize_ambiguity: bool = False,
        mode: EvaluationMode = EvaluationMode.GRANULAR,
        multi_turn_strategy: Literal['last_turn', 'all_turns'] = 'last_turn',
        **kwargs,
    ):
        """
        Initialize the Answer Relevancy metric.

        Args:
            relevancy_mode: The mode for judging relevancy.
                'strict': Only directly answering statements are relevant.
                'task': Closely related, helpful statements are also relevant (default).
            penalize_ambiguity (bool): If True, 'idk' verdicts are scored as 0.0 (irrelevant).
                                       If False (default), 'idk' verdicts are scored as 1.0 (relevant).
            mode (EvaluationMode): The evaluation mode for the internal RAGAnalyzer.

            multi_turn_strategy (Literal): How to handle multi-turn conversations.
                                           'last_turn' (default): Evaluates only the last turn.
                                           'all_turns': Evaluates all Human->AI turns in the conversation.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.penalize_ambiguity = penalize_ambiguity
        self.relevancy_mode = relevancy_mode
        self.verdict_score_mapping = {
            'yes': 1.0,
            'no': 0.0,
            'idk': 0.0 if penalize_ambiguity else 1.0,
        }
        self.multi_turn_strategy = multi_turn_strategy

        self.engine = RAGAnalyzer(
            mode=mode, relevancy_mode=self.relevancy_mode, **kwargs
        )
        self.relevancy_explainer = RelevancyExplainer(**kwargs)

    async def _evaluate_single_turn(
        self,
        item: DatasetItem,
        cache: Optional[AnalysisCache] = None,
        turn_index: int = 0,
    ) -> MetricEvaluationResult:
        """
        Run relevancy evaluation for a single turn (or the last turn of a conversation).
        """
        if not item.actual_output:
            return MetricEvaluationResult(
                score=np.nan,
                explanation='actual_output is missing from the DatasetItem.',
            )
        if not item.query:
            return MetricEvaluationResult(
                score=np.nan, explanation='query is missing from the DatasetItem.'
            )

        analysis = await self.engine.execute(item, cache, self.required_components)

        if analysis.has_error():
            return MetricEvaluationResult(
                score=np.nan, explanation=f'Analysis failed: {analysis.error}'
            )

        claims = analysis.judged_claims
        if not claims:
            return MetricEvaluationResult(
                score=1.0,
                explanation='No statements were extracted from the response.',
                signals=AnswerRelevancyResult(
                    overall_score=1.0,
                    explanation='No statements were extracted from the response.',
                    relevant_statements_count=0,
                    irrelevant_statements_count=0,
                    ambiguous_statements_count=0,
                    total_statements_count=0,
                    statement_breakdown=[],
                    evaluated_turns_count=1,
                ),
            )

        valid_claims = [
            c for c in claims if c.relevancy_verdict in ('yes', 'no', 'idk')
        ]

        # Compute score based on the verdicts
        relevant_count = sum(1 for c in valid_claims if c.relevancy_verdict == 'yes')
        irrelevant_count = sum(1 for c in valid_claims if c.relevancy_verdict == 'no')
        ambiguous_count = sum(1 for c in valid_claims if c.relevancy_verdict == 'idk')
        total_count = len(valid_claims)

        positive_count = relevant_count + (
            0 if self.penalize_ambiguity else ambiguous_count
        )
        score = positive_count / total_count if total_count > 0 else 1.0

        statement_breakdown = [
            StatementBreakdownResult(
                statement=c.claim_text,
                verdict=c.relevancy_verdict,
                reason=c.reason,
                is_relevant=(c.relevancy_verdict != 'no'),
                turn_index=turn_index,
            )
            for c in valid_claims
        ]

        # Generate final human-readable explanation
        irrelevant_reasons = [
            c.reason for c in valid_claims if c.relevancy_verdict == 'no' and c.reason
        ]
        explanation_result = await self.relevancy_explainer.execute(
            irrelevant_statements=irrelevant_reasons,
            input_query=item.query,
            score=score,
        )

        result_data = AnswerRelevancyResult(
            overall_score=score,
            explanation=explanation_result.explanation,
            relevant_statements_count=relevant_count,
            irrelevant_statements_count=irrelevant_count,
            ambiguous_statements_count=ambiguous_count,
            total_statements_count=total_count,
            statement_breakdown=statement_breakdown,
            evaluated_turns_count=1,
        )

        if analysis.metadata:
            self.cost_estimate = getattr(analysis.metadata, 'cost_estimate', 0)

        return MetricEvaluationResult(
            score=score,
            explanation=explanation_result.explanation,
            signals=result_data,
            metadata={
                'analysis_metadata': analysis.metadata.model_dump()
                if analysis.metadata
                else {}
            },
        )

    async def _evaluate_multi_turn(
        self, item: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricEvaluationResult:
        """
        Run relevancy evaluation for all turns in a conversation and aggregate results.
        """
        turns_to_eval = []
        if not item.conversation or not item.conversation.messages:
            return MetricEvaluationResult(
                score=np.nan, explanation='No conversation messages found to evaluate.'
            )

        current_query = None
        for i, message in enumerate(item.conversation.messages):
            if isinstance(message, HumanMessage):
                current_query = message.content
            elif isinstance(message, AIMessage) and message.content:
                if current_query is None:
                    logger.warning(
                        f'Found AIMessage at index {i} with no preceding HumanMessage. Skipping turn.'
                    )
                    continue

                turns_to_eval.append(
                    {
                        'query': current_query,
                        'actual_output': message.content,
                        'turn_index': len(turns_to_eval),
                    }
                )
                # Reset query to None to ensure we only link the *next* AI message
                current_query = None

        if not turns_to_eval:
            return MetricEvaluationResult(
                score=np.nan,
                explanation='No evaluable HumanMessage -> AIMessage turns found in conversation.',
            )

        all_statement_breakdowns: List[StatementBreakdownResult] = []
        all_claims = []
        all_irrelevant_reasons = []
        total_cost = 0.0

        for turn in turns_to_eval:
            turn_item = DatasetItem(
                query=turn['query'], actual_output=turn['actual_output']
            )
            analysis = await self.engine.execute(
                turn_item, cache, self.required_components
            )
            total_cost += (
                getattr(analysis.metadata, 'cost_estimate', 0)
                if analysis.metadata
                else 0
            )

            if analysis.has_error() or not analysis.judged_claims:
                continue

            claims = analysis.judged_claims
            all_claims.extend(claims)

            all_irrelevant_reasons.extend(
                [c.reason for c in claims if c.relevancy_verdict == 'no' and c.reason]
            )

            valid_claims = [
                c for c in claims if c.relevancy_verdict in ('yes', 'no', 'idk')
            ]

            all_statement_breakdowns.extend(
                [
                    StatementBreakdownResult(
                        statement=c.claim_text,
                        verdict=c.relevancy_verdict,
                        reason=c.reason,
                        is_relevant=(c.relevancy_verdict != 'no'),
                        turn_index=turn['turn_index'],
                    )
                    for c in valid_claims
                ]
            )

        if not all_claims:
            return MetricEvaluationResult(
                score=1.0,
                explanation='No statements were extracted from any turn in the conversation.',
            )

        # Micro-Average
        relevant_count = sum(1 for c in all_claims if c.relevancy_verdict == 'yes')
        irrelevant_count = sum(1 for c in all_claims if c.relevancy_verdict == 'no')
        ambiguous_count = sum(1 for c in all_claims if c.relevancy_verdict == 'idk')
        total_count = len(all_claims)

        positive_count = relevant_count + (
            0 if self.penalize_ambiguity else ambiguous_count
        )
        score = positive_count / total_count if total_count > 0 else 1.0

        # Pass all queries for a holistic explanation.
        all_query_contexts = [
            f"Turn {turn['turn_index']}: {turn['query']}" for turn in turns_to_eval
        ]
        holistic_query_context = (
            '\n---\nFull Conversation Query Context:\n' + '\n'.join(all_query_contexts)
        )

        explanation_result = await self.relevancy_explainer.execute(
            irrelevant_statements=all_irrelevant_reasons,
            input_query=holistic_query_context,
            score=score,
        )

        result_data = AnswerRelevancyResult(
            overall_score=score,
            explanation=explanation_result.explanation,
            relevant_statements_count=relevant_count,
            irrelevant_statements_count=irrelevant_count,
            ambiguous_statements_count=ambiguous_count,
            total_statements_count=total_count,
            statement_breakdown=all_statement_breakdowns,
            evaluated_turns_count=len(turns_to_eval),
        )

        self.cost_estimate = total_cost

        return MetricEvaluationResult(
            score=score,
            explanation=explanation_result.explanation,
            signals=result_data,
        )

    @trace(name='AnswerRelevancy', capture_args=True, capture_response=True)
    async def execute(
        self, item: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricEvaluationResult:
        """
        Compute the score based on criteria.
        Automatically handles single-turn or multi-turn evaluation based on
        `self.multi_turn_strategy` and `item.conversation`.
        """
        self._validate_required_metric_fields(item)

        if item.conversation and self.multi_turn_strategy == 'all_turns':
            logger.info('Using multi-turn "all_turns" evaluation approach')
            result = await self._evaluate_multi_turn(item, cache)
        else:
            logger.info('Using single-turn / "last_turn" evaluation approach')
            result = await self._evaluate_single_turn(item, cache, turn_index=0)

        return result

    def get_signals(
        self, result: AnswerRelevancyResult
    ) -> List[SignalDescriptor[AnswerRelevancyResult]]:
        signals = [
            SignalDescriptor(
                name='overall_score',
                extractor=lambda r: r.overall_score,
                headline_display=True,
            ),
            SignalDescriptor(
                name='total_statements', extractor=lambda r: r.total_statements_count
            ),
            SignalDescriptor(
                name='relevant_statements',
                extractor=lambda r: r.relevant_statements_count,
            ),
            SignalDescriptor(
                name='irrelevant_statements',
                extractor=lambda r: r.irrelevant_statements_count,
            ),
            SignalDescriptor(
                name='ambiguous_statements',
                extractor=lambda r: r.ambiguous_statements_count,
            ),
        ]

        if result.evaluated_turns_count:
            signals.append(
                SignalDescriptor(
                    name='evaluated_turns',
                    group='overall',
                    description='Total number of Human->AI turns evaluated.',
                    extractor=lambda r: r.evaluated_turns_count,
                )
            )

        for i, stmt in enumerate(result.statement_breakdown):
            stmt_preview = (
                stmt.statement[:80] + '...'
                if len(stmt.statement) > 80
                else stmt.statement
            )

            # Add turn_index to group name if present
            if stmt.turn_index is not None:
                group_name = f'turn_{stmt.turn_index}_statement_{i}: "{stmt_preview}"'
            else:
                group_name = f'statement_{i}: "{stmt_preview}"'

            signals.extend(
                [
                    SignalDescriptor(
                        name='is_relevant',
                        group=group_name,
                        extractor=lambda r, idx=i: r.statement_breakdown[
                            idx
                        ].is_relevant,
                        headline_display=True,
                    ),
                    SignalDescriptor(
                        name='verdict',
                        group=group_name,
                        extractor=lambda r, idx=i: r.statement_breakdown[idx].verdict,
                        score_mapping=self.verdict_score_mapping,
                    ),
                    SignalDescriptor(
                        name='statement',
                        group=group_name,
                        extractor=lambda r, idx=i: r.statement_breakdown[idx].statement,
                    ),
                    SignalDescriptor(
                        name='reason',
                        group=group_name,
                        extractor=lambda r, idx=i: r.statement_breakdown[idx].reason,
                        description='Reason for the verdict (if any).',
                    ),
                    SignalDescriptor(
                        name='turn_index',
                        group=group_name,
                        description='The conversation turn index, if applicable.',
                        extractor=lambda r, idx=i: r.statement_breakdown[
                            idx
                        ].turn_index,
                    ),
                ]
            )
        return signals
