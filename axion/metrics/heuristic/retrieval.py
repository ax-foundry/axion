from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.schema import SignalDescriptor

logger = get_logger(__name__)


class KDetail(RichBaseModel):
    """Detailed breakdown for a single K value within a multi-K metric."""

    score: float
    hits_in_top_k: Optional[int] = None
    total_retrieved_at_k: Optional[int] = None
    total_relevant: Optional[int] = None
    dcg: Optional[float] = None
    idcg: Optional[float] = None
    is_hit: Optional[bool] = None


class MultiKResult(RichBaseModel):
    """Generic structured result for metrics evaluated across multiple K's."""

    k_values: List[int]
    results_by_k: Dict[int, KDetail]
    overall_score: float  # Score of the largest K, used as the main metric score
    main_k_used: int  # The specific K value used for the primary score


class MeanReciprocalRankResult(RichBaseModel):
    """Structured result for Mean Reciprocal Rank (MRR). (K-independent)"""

    rank_of_first_relevant: Union[int, None]
    score: float


class _RetrievalMetric(BaseMetric):
    """
    Internal base class for retrieval metrics.

    Handles the common logic of parsing the DatasetItem to extract
    the actual ranked list and the expected ground truth.
    """

    def _parse_item(
        self, item: DatasetItem
    ) -> Tuple[
        Optional[List[str]],
        Optional[Dict[str, float]],
        Optional[MetricEvaluationResult],
    ]:
        """
        Parses the DatasetItem to get the data needed for IR metrics.

        Returns:
            Tuple[actual_ids, expected_relevance_map, error_result]:
            - actual_ids: List of retrieved document IDs in order.
            - expected_relevance_map: Dict of {doc_id: relevance_score}
            - error_result: A MetricEvaluationResult if parsing fails, else None.
        """
        if not item.actual_ranking:
            return (
                None,
                None,
                MetricEvaluationResult(
                    score=np.nan,
                    explanation='Metric computation failed: `actual_ranking` field is missing or empty.',
                ),
            )

        if not item.expected_reference:
            return (
                None,
                None,
                MetricEvaluationResult(
                    score=np.nan,
                    explanation='Metric computation failed: `expected_reference` field is missing or empty.',
                ),
            )

        try:
            # Get the ranked list of actual retrieved document IDs
            actual_ids = [str(rank_item['id']) for rank_item in item.actual_ranking]
        except (KeyError, TypeError) as e:
            return (
                None,
                None,
                MetricEvaluationResult(
                    score=np.nan,
                    explanation=f"Metric computation failed: `actual_ranking` items must be dicts with an 'id' key. Error: {e}",
                ),
            )

        try:
            # Get the ground truth relevant documents and their relevance scores
            expected_relevance_map = {
                str(ref_item['id']): float(ref_item.get('relevance', 1.0))
                for ref_item in item.expected_reference
            }
        except (KeyError, TypeError) as e:
            return (
                None,
                None,
                MetricEvaluationResult(
                    score=np.nan,
                    explanation=f"Metric computation failed: `expected_reference` items must be dicts with an 'id' key. Error: {e}",
                ),
            )

        if not expected_relevance_map:
            return (
                None,
                None,
                MetricEvaluationResult(
                    score=np.nan,
                    explanation='Metric computation failed: `expected_reference` parsed to an empty relevance map.',
                ),
            )

        return actual_ids, expected_relevance_map, None


@metric(
    name='Hit Rate @ K',
    key='hit_rate_at_k',
    description='Returns 1 if at least one relevant result appears in the top-K, else 0.',
    required_fields=['actual_ranking', 'expected_reference'],
    score_range=(0, 1),
    tags=['retrieval'],
)
class HitRateAtK(_RetrievalMetric):
    """
    Evaluates if any relevant document was retrieved in the top K results.
    Score is 1 if a hit is found, 0 otherwise. Now supports multiple K values.
    """

    def __init__(
        self, k: Union[int, List[int]] = 10, main_k: Optional[int] = None, **kwargs
    ):
        """
        Initialize the Hit Rate @ K metric.
        Args:
            k: The number of top results to consider, or a list of K values.
            main_k: The K value to use for the main metric score (defaults to max K in k_list).
        """
        super().__init__(**kwargs)
        if isinstance(k, int):
            k = [k]
        self.k_list = sorted([ki for ki in set(k) if ki > 0])
        if not self.k_list:
            raise ValueError('k must contain at least one positive integer.')

        # Determine the K to use for the final score
        self.max_k = self.k_list[-1]
        self.main_k = main_k if main_k is not None else self.max_k

        if self.main_k not in self.k_list:
            raise ValueError(
                f'main_k ({self.main_k}) must be one of the specified k values: {self.k_list}'
            )

    @trace(name='HitRateAtK.execute', capture_args=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        actual_ids, expected_relevance_map, error = self._parse_item(item)
        if error:
            return error

        relevant_docs = {
            doc_id
            for doc_id, relevance in expected_relevance_map.items()
            if relevance > 0
        }

        results_by_k: Dict[int, KDetail] = {}
        main_k_score = 0.0

        for k in self.k_list:
            actual_at_k = actual_ids[:k]
            is_hit = any(doc_id in relevant_docs for doc_id in actual_at_k)
            score = 1.0 if is_hit else 0.0

            results_by_k[k] = KDetail(score=score, is_hit=is_hit)

            if k == self.main_k:
                main_k_score = score

        explanation = f'Scores calculated for K={self.k_list}. Main score (K={self.main_k}) is {main_k_score}.'

        result_data = MultiKResult(
            k_values=self.k_list,
            results_by_k=results_by_k,
            overall_score=main_k_score,
            main_k_used=self.main_k,
        )

        return MetricEvaluationResult(
            score=main_k_score, explanation=explanation, signals=result_data
        )

    def get_signals(self, result: MultiKResult) -> List[SignalDescriptor]:
        """Generates signals detailing the hit rate calculation for all K values."""
        signals = []

        # Overall score (main K)
        signals.append(
            SignalDescriptor(
                name='hit_rate_score',
                extractor=lambda r: r.overall_score,
                description=f'Final Hit Rate @ {result.main_k_used} score (1.0 or 0.0).',
                headline_display=True,
                score_mapping={'True': 1.0, 'False': 0.0},
            )
        )

        # Per-K signals
        for k in result.k_values:
            signals.append(
                SignalDescriptor(
                    name=f'is_hit_at_{k}',
                    group=f'k_{k}_detail',
                    extractor=lambda r, k=k: r.results_by_k[k].is_hit,
                    description=f'Whether a relevant document was found in the top {k} (1.0 for hit, 0.0 for miss).',
                    score_mapping={'True': 1.0, 'False': 0.0},
                )
            )
            signals.append(
                SignalDescriptor(
                    name=f'score_at_{k}',
                    group=f'k_{k}_detail',
                    extractor=lambda r, k=k: r.results_by_k[k].score,
                    description=f'Hit Rate @ {k} score.',
                )
            )

        return signals


@metric(
    name='Mean Reciprocal Rank (MRR)',
    key='mean_reciprocal_rank',
    description='Calculates the reciprocal rank of the first relevant result found.',
    required_fields=['actual_ranking', 'expected_reference'],
    score_range=(0, 1),
    tags=['retrieval'],
)
class MeanReciprocalRank(_RetrievalMetric):
    """
    Calculates the Mean Reciprocal Rank (MRR).
    Score is 1 / (rank of first relevant item). This metric is K-independent.
    """

    @trace(name='MeanReciprocalRank.execute', capture_args=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        actual_ids, expected_relevance_map, error = self._parse_item(item)
        if error:
            return error

        relevant_docs = {
            doc_id
            for doc_id, relevance in expected_relevance_map.items()
            if relevance > 0
        }

        rank = 0
        score = 0.0
        explanation = 'No relevant documents found in the retrieved results.'

        for i, doc_id in enumerate(actual_ids):
            if doc_id in relevant_docs:
                rank = i + 1
                score = 1.0 / rank
                explanation = (
                    f"First relevant document ('{doc_id}') found at rank {rank}."
                )
                break

        result_data = MeanReciprocalRankResult(
            rank_of_first_relevant=rank if rank > 0 else None, score=score
        )

        return MetricEvaluationResult(
            score=score, explanation=explanation, signals=result_data
        )

    def get_signals(self, result: MeanReciprocalRankResult) -> List[SignalDescriptor]:
        """Generates signals detailing the Mean Reciprocal Rank calculation."""
        return [
            SignalDescriptor(
                name='rank_of_first_relevant',
                extractor=lambda r: r.rank_of_first_relevant or -1,
                description='The rank of the first relevant document found (or -1 if none found).',
                headline_display=True,
            ),
            SignalDescriptor(
                name='mrr_score',
                extractor=lambda r: r.score,
                description='Final Mean Reciprocal Rank score (1/rank).',
            ),
        ]


@metric(
    name='NDCG @ K',
    key='ndcg_at_k',
    description='Calculates the Normalized Discounted Cumulative Gain (NDCG) at K.',
    required_fields=['actual_ranking', 'expected_reference'],
    score_range=(0, 1),
    tags=['retrieval'],
)
class NDCGAtK(_RetrievalMetric):
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) at K.
    This metric handles graded relevance scores. Now supports multiple K values.
    """

    def __init__(
        self, k: Union[int, List[int]] = 10, main_k: Optional[int] = None, **kwargs
    ):
        """
        Initialize the NDCG @ K metric.
        Args:
            k: The number of top results to consider, or a list of K values.
            main_k: The K value to use for the main metric score (defaults to max K in k_list).
        """
        super().__init__(**kwargs)
        if isinstance(k, int):
            k = [k]
        self.k_list = sorted([ki for ki in set(k) if ki > 0])
        if not self.k_list:
            raise ValueError('k must contain at least one positive integer.')

        # Determine the K to use for the final score
        self.max_k = self.k_list[-1]
        self.main_k = main_k if main_k is not None else self.max_k

        if self.main_k not in self.k_list:
            raise ValueError(
                f'main_k ({self.main_k}) must be one of the specified k values: {self.k_list}'
            )

    def _dcg_at_k(self, relevances: List[float], k: int) -> float:
        """Helper to compute DCG."""
        relevances = np.array(relevances[:k])
        if relevances.size == 0:
            return 0.0
        # Denominators start at log2(2) = 1.0
        discounts = np.log2(np.arange(2, relevances.size + 2))
        return np.sum(relevances / discounts)

    @trace(name='NDCGAtK.execute', capture_args=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        actual_ids, expected_relevance_map, error = self._parse_item(item)
        if error:
            return error

        # Get relevance scores for the actual retrieved order
        actual_relevances = [
            expected_relevance_map.get(doc_id, 0.0) for doc_id in actual_ids
        ]

        # Calculate Ideal DCG (IDCG) - this is done once
        ideal_relevances = sorted(expected_relevance_map.values(), reverse=True)

        results_by_k: Dict[int, KDetail] = {}
        main_k_score = 0.0

        for k in self.k_list:
            # Calculate DCG for the actual results
            dcg = self._dcg_at_k(actual_relevances, k)

            # Calculate IDCG for this K
            idcg = self._dcg_at_k(ideal_relevances, k)

            if idcg == 0:
                score = 0.0
            else:
                score = dcg / idcg

            results_by_k[k] = KDetail(
                score=score,
                dcg=dcg,
                idcg=idcg,
            )

            if k == self.main_k:
                main_k_score = score

        explanation = f'Scores calculated for K={self.k_list}. Main score (K={self.main_k}) is {main_k_score:.4f}.'

        result_data = MultiKResult(
            k_values=self.k_list,
            results_by_k=results_by_k,
            overall_score=main_k_score,
            main_k_used=self.main_k,
        )

        return MetricEvaluationResult(
            score=main_k_score, explanation=explanation, signals=result_data
        )

    def get_signals(self, result: MultiKResult) -> List[SignalDescriptor]:
        """Generates signals detailing the NDCG @ K calculation for all K values."""
        signals = []

        # Overall score (main K)
        signals.append(
            SignalDescriptor(
                name='ndcg_score',
                extractor=lambda r: r.overall_score,
                description=f'The final Normalized Discounted Cumulative Gain score @ {result.main_k_used}.',
                headline_display=True,
            )
        )

        # Per-K signals
        for k in result.k_values:
            # Create a lambda that uses the captured k value
            signals.extend(
                [
                    SignalDescriptor(
                        name=f'score_at_{k}',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].score,
                        description=f'NDCG @ {k} score.',
                    ),
                    SignalDescriptor(
                        name=f'dcg_at_{k}',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].dcg,
                        description=f'DCG for the actual ranking at K={k}.',
                    ),
                    SignalDescriptor(
                        name=f'idcg_at_{k}',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].idcg,
                        description=f'IDCG for the ideal ranking at K={k}.',
                    ),
                ]
            )

        return signals


@metric(
    name='Precision @ K',
    key='precision_at_k',
    description='Calculates the percentage of top-K results that are relevant.',
    required_fields=['actual_ranking', 'expected_reference'],
    score_range=(0, 1),
    tags=['retrieval'],
)
class PrecisionAtK(_RetrievalMetric):
    """
    Calculates Precision @ K. Now supports multiple K values.
    """

    def __init__(
        self, k: Union[int, List[int]] = 10, main_k: Optional[int] = None, **kwargs
    ):
        """
        Initialize the Precision @ K metric.
        Args:
            k: The number of top results to consider, or a list of K values.
            main_k: The K value to use for the main metric score (defaults to max K in k_list).
        """
        super().__init__(**kwargs)
        if isinstance(k, int):
            k = [k]
        self.k_list = sorted([ki for ki in set(k) if ki > 0])
        if not self.k_list:
            raise ValueError('k must contain at least one positive integer.')

        # Determine the K to use for the final score
        self.max_k = self.k_list[-1]
        self.main_k = main_k if main_k is not None else self.max_k

        if self.main_k not in self.k_list:
            raise ValueError(
                f'main_k ({self.main_k}) must be one of the specified k values: {self.k_list}'
            )

    @trace(name='PrecisionAtK.execute', capture_args=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        actual_ids, expected_relevance_map, error = self._parse_item(item)
        if error:
            return error

        relevant_docs = {
            doc_id
            for doc_id, relevance in expected_relevance_map.items()
            if relevance > 0
        }
        total_relevant = len(relevant_docs)

        results_by_k: Dict[int, KDetail] = {}
        main_k_score = 0.0

        for k in self.k_list:
            actual_at_k = actual_ids[:k]
            total_retrieved_at_k = len(actual_at_k)

            if not actual_at_k:
                score = 0.0
                hits_in_top_k = 0
            else:
                hits_in_top_k = sum(
                    1 for doc_id in actual_at_k if doc_id in relevant_docs
                )
                score = hits_in_top_k / total_retrieved_at_k

            results_by_k[k] = KDetail(
                score=score,
                hits_in_top_k=hits_in_top_k,
                total_retrieved_at_k=total_retrieved_at_k,
                total_relevant=total_relevant,
            )

            if k == self.main_k:
                main_k_score = score

        explanation = f'Scores calculated for K={self.k_list}. Main score (K={self.main_k}) is {main_k_score:.4f}.'

        result_data = MultiKResult(
            k_values=self.k_list,
            results_by_k=results_by_k,
            overall_score=main_k_score,
            main_k_used=self.main_k,
        )
        return MetricEvaluationResult(
            score=main_k_score, explanation=explanation, signals=result_data
        )

    def get_signals(self, result: MultiKResult) -> List[SignalDescriptor]:
        """Generates signals detailing the Precision @ K calculation for all K values."""
        signals = []

        # Overall score (main K)
        signals.append(
            SignalDescriptor(
                name='precision_score',
                extractor=lambda r: r.overall_score,
                description=f'The final Precision @ {result.main_k_used} score.',
                headline_display=True,
            )
        )

        # Per-K signals
        for k in result.k_values:
            signals.extend(
                [
                    SignalDescriptor(
                        name=f'score_at_{k}',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].score,
                        description=f'Precision @ {k} score.',
                    ),
                    SignalDescriptor(
                        name=f'hits_at_{k}',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].hits_in_top_k,
                        description=f'Relevant documents found in the top {k} (True Positives).',
                    ),
                    SignalDescriptor(
                        name='total_relevant',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].total_relevant,
                        description='Total relevant documents for the query.',
                    ),
                    SignalDescriptor(
                        name=f'retrieved_at_{k}',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].total_retrieved_at_k,
                        description=f'Total documents retrieved (max {k}).',
                    ),
                ]
            )

        return signals


@metric(
    name='Recall @ K',
    key='recall_at_k',
    description='Calculates the percentage of all relevant results found in the top-K.',
    required_fields=['actual_ranking', 'expected_reference'],
    score_range=(0, 1),
    tags=['retrieval'],
)
class RecallAtK(_RetrievalMetric):
    """
    Calculates Recall @ K. Now supports multiple K values.
    """

    def __init__(
        self, k: Union[int, List[int]] = 10, main_k: Optional[int] = None, **kwargs
    ):
        """
        Initialize the Recall @ K metric.
        Args:
            k: The number of top results to consider, or a list of K values.
            main_k: The K value to use for the main metric score (defaults to max K in k_list).
        """
        super().__init__(**kwargs)
        if isinstance(k, int):
            k = [k]
        self.k_list = sorted([ki for ki in set(k) if ki > 0])
        if not self.k_list:
            raise ValueError('k must contain at least one positive integer.')

        # Determine the K to use for the final score
        self.max_k = self.k_list[-1]
        self.main_k = main_k if main_k is not None else self.max_k

        if self.main_k not in self.k_list:
            raise ValueError(
                f'main_k ({self.main_k}) must be one of the specified k values: {self.k_list}'
            )

    @trace(name='RecallAtK.execute', capture_args=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        actual_ids, expected_relevance_map, error = self._parse_item(item)
        if error:
            return error

        relevant_docs = {
            doc_id
            for doc_id, relevance in expected_relevance_map.items()
            if relevance > 0
        }
        total_relevant = len(relevant_docs)

        if total_relevant == 0:
            explanation = 'No relevant documents were expected, so recall is trivially 1.0 for all K.'
            # The result must still reflect the chosen main_k
            result_data = MultiKResult(
                k_values=self.k_list,
                results_by_k={k: KDetail(score=1.0) for k in self.k_list},
                overall_score=1.0,
                main_k_used=self.main_k,
            )
            return MetricEvaluationResult(
                score=1.0, explanation=explanation, signals=result_data
            )

        results_by_k: Dict[int, KDetail] = {}
        main_k_score = 0.0

        for k in self.k_list:
            actual_at_k = actual_ids[:k]
            total_retrieved_at_k = len(actual_at_k)
            hits_in_top_k = sum(1 for doc_id in actual_at_k if doc_id in relevant_docs)
            score = hits_in_top_k / total_relevant

            results_by_k[k] = KDetail(
                score=score,
                hits_in_top_k=hits_in_top_k,
                total_retrieved_at_k=total_retrieved_at_k,
                total_relevant=total_relevant,
            )

            if k == self.main_k:
                main_k_score = score

        explanation = f'Scores calculated for K={self.k_list}. Main score (K={self.main_k}) is {main_k_score:.4f}.'

        result_data = MultiKResult(
            k_values=self.k_list,
            results_by_k=results_by_k,
            overall_score=main_k_score,
            main_k_used=self.main_k,
        )

        return MetricEvaluationResult(
            score=main_k_score, explanation=explanation, signals=result_data
        )

    def get_signals(self, result: MultiKResult) -> List[SignalDescriptor]:
        """Generates signals detailing the Recall @ K calculation for all K values."""
        signals = []

        # Overall score (main K)
        signals.append(
            SignalDescriptor(
                name='recall_score',
                extractor=lambda r: r.overall_score,
                description=f'The final Recall @ {result.main_k_used} score.',
                headline_display=True,
            )
        )

        # Per-K signals
        for k in result.k_values:
            signals.extend(
                [
                    SignalDescriptor(
                        name=f'score_at_{k}',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].score,
                        description=f'Recall @ {k} score.',
                    ),
                    SignalDescriptor(
                        name=f'hits_at_{k}',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].hits_in_top_k,
                        description=f'Relevant documents found in the top {k} (True Positives).',
                    ),
                    SignalDescriptor(
                        name='total_relevant',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].total_relevant,
                        description='Total relevant documents in the dataset item (Possible Positives).',
                    ),
                    SignalDescriptor(
                        name=f'retrieved_at_{k}',
                        group=f'k_{k}_detail',
                        extractor=lambda r, k=k: r.results_by_k[k].total_retrieved_at_k,
                        description=f'Total documents retrieved (max {k}).',
                    ),
                ]
            )

        return signals
