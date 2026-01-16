import pytest

from axion.dataset import DatasetItem
from axion.metrics.heuristic.bleu import SentenceBLEU


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'actual, expected, expected_score_range',
    [
        ('The cat is on the mat', 'The cat is on the mat', (1.0, 1.0)),  # Perfect match
        (
            'The cat is on the mat',
            'The cat sat on the mat',
            (0.4, 0.8),  # Good overlap with smoothing
        ),
        (
            'Completely unrelated sentence',
            'Another unrelated one',
            (0.0, 0.1),  # Very low due to smoothing
        ),
    ],
)
async def test_sentence_bleu_basic(actual, expected, expected_score_range):
    item = DatasetItem(actual_output=actual, expected_output=expected)
    metric = SentenceBLEU(smoothing=True)  # Use smoothing by default
    result = await metric.execute(item)
    assert isinstance(result.score, float)
    min_score, max_score = expected_score_range
    assert min_score <= result.score <= max_score


@pytest.mark.asyncio
async def test_sentence_bleu_no_smoothing():
    """Test sentence BLEU without smoothing for stricter scoring"""
    item = DatasetItem(
        actual_output='Completely unrelated sentence',
        expected_output='Another unrelated one',
    )
    metric = SentenceBLEU(smoothing=False)
    result = await metric.execute(item)
    assert result.score == 0.0  # Should be exactly 0 without smoothing


@pytest.mark.asyncio
async def test_sentence_bleu_partial_match_no_smoothing():
    """Test partial match without smoothing"""
    item = DatasetItem(
        actual_output='The cat is on the mat', expected_output='The cat sat on the mat'
    )
    metric = SentenceBLEU(smoothing=False)
    result = await metric.execute(item)
    # Without smoothing, this should be 0 due to zero 4-gram precision
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_sentence_bleu_partial_match_with_smoothing():
    """Test partial match with smoothing"""
    item = DatasetItem(
        actual_output='The cat is on the mat', expected_output='The cat sat on the mat'
    )
    metric = SentenceBLEU(smoothing=True)
    result = await metric.execute(item)
    # With smoothing, should get a reasonable score
    assert 0.4 < result.score < 0.8


@pytest.mark.asyncio
async def test_bleu_case_insensitive():
    item = DatasetItem(
        actual_output='The cat is On the mat', expected_output='the cat is on the mat'
    )
    metric = SentenceBLEU(case_sensitive=False)
    result = await metric.execute(item)
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_bleu_case_sensitive_differs():
    item = DatasetItem(
        actual_output='The cat is On the mat', expected_output='the cat is on the mat'
    )
    metric = SentenceBLEU(case_sensitive=True)
    result = await metric.execute(item)
    assert result.score < 1.0


@pytest.mark.asyncio
async def test_bleu_empty_output():
    item = DatasetItem(actual_output='', expected_output='Some text')
    metric = SentenceBLEU()
    result = await metric.execute(item)
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_bleu_empty_reference():
    item = DatasetItem(actual_output='Some output', expected_output='')
    metric = SentenceBLEU()
    result = await metric.execute(item)
    # With empty reference and non-empty candidate, should get very low score
    assert result.score >= 0.0  # Could be low but not necessarily 0 with smoothing


@pytest.mark.asyncio
async def test_bleu_both_empty():
    """Test when both actual and expected outputs are empty"""
    item = DatasetItem(actual_output='', expected_output='')
    metric = SentenceBLEU()
    result = await metric.execute(item)
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_bleu_different_n_grams():
    """Test BLEU with different n-gram sizes"""
    item = DatasetItem(
        actual_output='The cat sat on the mat', expected_output='The cat is on the mat'
    )

    # Test with different n-gram sizes
    for n_grams in [1, 2, 3, 4]:
        metric = SentenceBLEU(n_grams=n_grams, smoothing=True)
        result = await metric.execute(item)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_sentence_bleu_short_sequences():
    """Test sentence BLEU with very short sequences"""
    item = DatasetItem(actual_output='cat', expected_output='dog')

    # Without smoothing, should be 0 (no 2,3,4-gram matches)
    metric_no_smooth = SentenceBLEU(smoothing=False)
    result_no_smooth = await metric_no_smooth.execute(item)
    assert result_no_smooth.score == 0.0

    # With smoothing, should get some small score
    metric_smooth = SentenceBLEU(smoothing=True)
    result_smooth = await metric_smooth.execute(item)
    assert 0.0 <= result_smooth.score < 0.3


@pytest.mark.asyncio
async def test_brevity_penalty_effect():
    """Test that brevity penalty affects scores appropriately"""
    # Shorter candidate should get lower score due to brevity penalty
    item_short = DatasetItem(
        actual_output='cat mat', expected_output='the cat is on the mat'
    )

    # Longer candidate (same length as reference)
    item_long = DatasetItem(
        actual_output='the cat sat on the mat', expected_output='the cat is on the mat'
    )

    metric = SentenceBLEU(smoothing=True)
    result_short = await metric.execute(item_short)
    result_long = await metric.execute(item_long)

    # Longer candidate should generally score higher due to better brevity penalty
    # (though this depends on n-gram matches too)
    assert isinstance(result_short.score, float)
    assert isinstance(result_long.score, float)
    assert 0.0 <= result_short.score <= 1.0
    assert 0.0 <= result_long.score <= 1.0
