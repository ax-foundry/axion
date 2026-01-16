import numpy as np
import pytest

from axion.dataset import DatasetItem
from axion.metrics.heuristic.retrieval import (
    HitRateAtK,
    MeanReciprocalRank,
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
)

ACTUAL_RANKING_A = [
    {'id': 'doc1'},  # Rank 1
    {'id': 'doc2'},  # Rank 2 (Relevant)
    {'id': 'doc3'},  # Rank 3
    {'id': 'doc4'},  # Rank 4 (Relevant)
    {'id': 'doc5'},  # Rank 5
]

EXPECTED_REFERENCE_A = [
    {'id': 'doc2', 'relevance': 1.0},
    {'id': 'doc4', 'relevance': 1.0},
    {'id': 'doc6', 'relevance': 1.0},  # Total 3 relevant
]

EXPECTED_REFERENCE_GRADED = [
    {'id': 'doc1', 'relevance': 3.0},
    {'id': 'doc2', 'relevance': 1.0},
    {'id': 'doc3', 'relevance': 2.0},
    {'id': 'doc4', 'relevance': 0.0},  # Not relevant
]

K_METRICS = [HitRateAtK, NDCGAtK, PrecisionAtK, RecallAtK]
MULTI_K_LIST = [1, 2, 3, 4, 5]


@pytest.mark.parametrize('metric_class', K_METRICS)
def test_k_metrics_initialization(metric_class):
    # Test valid initialization with single int K
    metric_class(k=5)

    # Test valid initialization with list of K
    metric_class(k=[1, 3, 5])

    # Test valid initialization with list and explicit main_k
    metric_class(k=[1, 3, 5], main_k=3)

    # Test invalid initialization (k <= 0)
    with pytest.raises(
        ValueError, match='k must contain at least one positive integer'
    ):
        metric_class(k=0)
    with pytest.raises(
        ValueError, match='k must contain at least one positive integer'
    ):
        metric_class(k=[-1])
    with pytest.raises(
        ValueError, match='k must contain at least one positive integer'
    ):
        metric_class(k=[0, -1])

    # Test invalid initialization (main_k not in k_list)
    with pytest.raises(ValueError, match='main_k'):
        metric_class(k=[1, 3], main_k=5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'actual, expected, error_match',
    [
        (None, EXPECTED_REFERENCE_A, '`actual_ranking` field is missing or empty.'),
        (ACTUAL_RANKING_A, None, '`expected_reference` field is missing or empty.'),
        ([{'not_id': 'doc1'}], EXPECTED_REFERENCE_A, "must be dicts with an 'id' key."),
        (ACTUAL_RANKING_A, [{'not_id': 'doc1'}], "must be dicts with an 'id' key."),
        (
            ACTUAL_RANKING_A,
            [],
            # FIX: Confirmed the exact error message that triggers when expected_reference is falsy/empty list.
            'Metric computation failed: `expected_reference` field is missing or empty.',
        ),
    ],
    ids=[
        'Missing_Actual',
        'Missing_Expected',
        'Malformed_Actual',
        'Malformed_Expected',
        'Empty_Expected_Reference_Map',
    ],
)
async def test_base_parsing_failures(actual, expected, error_match):
    item = DatasetItem(actual_ranking=actual, expected_reference=expected)
    # Use MRR as a representative metric that relies on _parse_item
    metric = MeanReciprocalRank()
    result = await metric.execute(item)

    assert np.isnan(result.score)
    assert error_match in result.explanation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'k_list, main_k, expected_score',
    [
        # Single K tests (main_k = max_k)
        ([1], 0, 0.0),  # main_k=1 (doc2 is at rank 2)
        ([2], 0, 1.0),  # main_k=2 (doc2 is at rank 2)
        ([3, 5], 0, 1.0),  # main_k=5 (Hit)
        # Multi K with explicit main_k
        ([1, 2, 5], 1, 0.0),  # main_k=1 (Miss)
        ([1, 2, 5], 2, 1.0),  # main_k=2 (Hit)
        ([10], 0, 1.0),
    ],
    ids=[
        'k1_miss',
        'k2_hit',
        'k_multi_max',
        'k_multi_main1_miss',
        'k_multi_main2_hit',
        'k10_hit_long',
    ],
)
async def test_hit_rate_at_k_standard(k_list, main_k, expected_score):
    k_list = k_list if k_list[0] != 0 else [1]
    main_k = main_k if main_k != 0 else max(k_list)

    item = DatasetItem(
        actual_ranking=ACTUAL_RANKING_A, expected_reference=EXPECTED_REFERENCE_A
    )
    metric = HitRateAtK(k=k_list, main_k=main_k)
    result = await metric.execute(item)
    assert result.score == expected_score
    # Check for the new generic explanation format
    assert 'Scores calculated for K=' in result.explanation
    assert f'Main score (K={main_k}) is {expected_score}.' in result.explanation


@pytest.mark.asyncio
async def test_hit_rate_at_k_no_relevant_docs():
    # Only non-relevant docs are expected
    item = DatasetItem(
        actual_ranking=ACTUAL_RANKING_A,
        expected_reference=[
            {'id': 'docX', 'relevance': 0.0},
            {'id': 'docY', 'relevance': 0.0},
        ],
    )
    metric = HitRateAtK(k=3)
    result = await metric.execute(item)
    assert result.score == 0.0
    # Check for the new generic explanation format
    assert 'Scores calculated for K=' in result.explanation
    assert 'Main score (K=3) is 0.0.' in result.explanation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'actual, expected_score, expected_rank',
    [
        # Relevant doc at Rank 1 (doc1 is relevant)
        ([{'id': 'doc1'}, {'id': 'doc2'}], 1.0, 1),
        # Relevant doc at Rank 2 (doc2 is relevant)
        ([{'id': 'docX'}, {'id': 'doc2'}, {'id': 'doc1'}], 0.5, 2),
        # Relevant doc at Rank 3 (doc3 is relevant)
        ([{'id': 'docX'}, {'id': 'docY'}, {'id': 'doc3'}], 1 / 3, 3),
        # Relevant doc at end of list (doc3 is relevant)
        ([{'id': 'docX'}, {'id': 'docY'}, {'id': 'docZ'}, {'id': 'doc3'}], 0.25, 4),
    ],
    ids=['rank_1', 'rank_2', 'rank_3', 'rank_4'],
)
async def test_mrr_standard(actual, expected_score, expected_rank):
    item = DatasetItem(
        actual_ranking=actual, expected_reference=EXPECTED_REFERENCE_GRADED
    )
    metric = MeanReciprocalRank()
    result = await metric.execute(item)
    assert np.isclose(result.score, expected_score)
    assert (
        f"First relevant document ('{actual[expected_rank - 1]['id']}') found at rank {expected_rank}."
        in result.explanation
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'k_list, main_k, actual, expected_ref, expected_main_score',
    [
        # P@1: (0/1) = 0.0 -> main_k=1
        ([1], 1, ACTUAL_RANKING_A, EXPECTED_REFERENCE_A, 0.0),
        # P@2: (1/2) = 0.5 -> main_k=2
        ([2, 4], 2, ACTUAL_RANKING_A, EXPECTED_REFERENCE_A, 0.5),
        # P@4: (2/4) = 0.5 -> main_k=4
        ([2, 4], 4, ACTUAL_RANKING_A, EXPECTED_REFERENCE_A, 0.5),
        # P@1: Perfect (doc1 is relevant). -> main_k=1
        ([1], 1, [{'id': 'doc1'}], EXPECTED_REFERENCE_GRADED, 1.0),
    ],
    ids=[
        'P1_miss',
        'P2_partial',
        'P4_partial',
        'P1_perfect',
    ],
)
async def test_precision_at_k_standard(
    k_list, main_k, actual, expected_ref, expected_main_score
):
    item = DatasetItem(actual_ranking=actual, expected_reference=expected_ref)
    metric = PrecisionAtK(k=k_list, main_k=main_k)

    result = await metric.execute(item)
    assert np.isclose(result.score, expected_main_score)
    assert 'Scores calculated for K=' in result.explanation
    assert (
        f'Main score (K={main_k}) is {expected_main_score:.4f}.' in result.explanation
    )


@pytest.mark.asyncio
async def test_precision_at_k_empty_actual():
    # Test the hard failure case handled by the base class (_parse_item)
    item = DatasetItem(actual_ranking=[], expected_reference=EXPECTED_REFERENCE_A)
    metric = PrecisionAtK(k=5)
    result = await metric.execute(item)

    # Assert score is NaN and the base class error message is returned.
    assert np.isnan(result.score)
    assert (
        'Metric computation failed: `actual_ranking` field is missing or empty.'
        in result.explanation
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'k_list, main_k, actual, expected_ref, expected_main_score',
    [
        # Total Relevant = 3 (doc2, doc4, doc6)
        # R@1: (0/3) = 0.0 -> main_k=1
        ([1], 1, ACTUAL_RANKING_A, EXPECTED_REFERENCE_A, 0.0),
        # R@2: (1/3) ≈ 0.3333 -> main_k=2
        ([2, 4], 2, ACTUAL_RANKING_A, EXPECTED_REFERENCE_A, 1 / 3),
        # R@4: (2/3) ≈ 0.6667 -> main_k=4
        ([2, 4], 4, ACTUAL_RANKING_A, EXPECTED_REFERENCE_A, 2 / 3),
        # R@10: (2/3) ≈ 0.6667 -> main_k=10
        ([1, 10], 10, ACTUAL_RANKING_A, EXPECTED_REFERENCE_A, 2 / 3),
    ],
    ids=['R1_miss', 'R2_partial', 'R4_partial', 'R10_max_recall'],
)
async def test_recall_at_k_standard(
    k_list, main_k, actual, expected_ref, expected_main_score
):
    item = DatasetItem(actual_ranking=actual, expected_reference=expected_ref)
    metric = RecallAtK(k=k_list, main_k=main_k)

    result = await metric.execute(item)
    assert np.isclose(result.score, expected_main_score)
    # Check for the new generic explanation format
    assert 'Scores calculated for K=' in result.explanation
    assert (
        f'Main score (K={main_k}) is {expected_main_score:.4f}.' in result.explanation
    )


@pytest.mark.asyncio
async def test_recall_at_k_no_relevant_expected():
    # Total Relevant = 0, so recall is 1.0. The explanation for this case is custom.
    item = DatasetItem(
        actual_ranking=ACTUAL_RANKING_A,
        expected_reference=[{'id': 'docX', 'relevance': 0.0}],
    )
    metric = RecallAtK(k=3)
    result = await metric.execute(item)
    assert result.score == 1.0
    assert (
        'No relevant documents were expected, so recall is trivially 1.0 for all K.'
        in result.explanation
    )


@pytest.mark.asyncio
async def test_recall_at_k_perfect_recall():
    # Total Relevant = 2 (doc2, doc4). Both found in top 2.
    item = DatasetItem(
        actual_ranking=[{'id': 'doc2'}, {'id': 'doc4'}, {'id': 'docX'}],
        expected_reference=[
            {'id': 'doc2', 'relevance': 1},
            {'id': 'doc4', 'relevance': 1},
        ],
    )
    metric = RecallAtK(k=2)
    result = await metric.execute(item)
    assert result.score == 1.0
    # Check for the new generic explanation format
    assert (
        'Scores calculated for K=[2]. Main score (K=2) is 1.0000.' in result.explanation
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'k_list, main_k, expected_main_score',
    [
        # Perfect score at k=3.
        ([3], 3, 1.0),
        # Worst score at k=3.
        ([3], 3, 0.0),
        # Misordered score at k=3.
        ([3], 3, 0.79),
        # Perfect score at k=3, main_k=10
        ([3, 10], 10, 1.0),
    ],
    ids=['perfect', 'worst', 'misordered', 'k_greater_than_ranking'],
)
async def test_ndcg_at_k_various_cases(k_list, main_k, expected_main_score, request):
    # Setup for each case
    case_id = request.node.callspec.id

    if case_id == 'perfect' or case_id == 'k_greater_than_ranking':
        actual = [{'id': 'doc1'}, {'id': 'doc3'}, {'id': 'doc2'}]
        expected_ref = EXPECTED_REFERENCE_GRADED
    elif case_id == 'worst':
        actual = [{'id': 'docX'}, {'id': 'docY'}, {'id': 'docZ'}]
        expected_ref = EXPECTED_REFERENCE_GRADED
    elif case_id == 'misordered':
        actual = [{'id': 'doc2'}, {'id': 'doc3'}, {'id': 'doc1'}]
        expected_ref = EXPECTED_REFERENCE_GRADED

    item = DatasetItem(actual_ranking=actual, expected_reference=expected_ref)
    metric = NDCGAtK(k=k_list, main_k=main_k)
    result = await metric.execute(item)

    # Assert main score
    assert np.isclose(result.score, expected_main_score, atol=0.01)

    # Check new generic wrapper
    assert 'Scores calculated for K=' in result.explanation
    assert f'Main score (K={main_k}) is {result.score:.4f}.' in result.explanation


@pytest.mark.asyncio
async def test_ndcg_at_k_idcg_zero():
    # Expected reference only contains non-relevant documents (IDCG = 0)
    expected_ref = [{'id': 'docA', 'relevance': 0.0}, {'id': 'docB', 'relevance': 0.0}]
    actual = ACTUAL_RANKING_A
    item = DatasetItem(actual_ranking=actual, expected_reference=expected_ref)
    metric = NDCGAtK(k=3)
    result = await metric.execute(item)
    assert result.score == 0.0

    assert 'Scores calculated for K=' in result.explanation
    assert 'Main score (K=3) is 0.0000.' in result.explanation
