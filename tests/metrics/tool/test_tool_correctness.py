import pytest

from axion._core.schema import ToolCall
from axion.dataset import DatasetItem
from axion.metrics.tool.tool_correctness import ToolCorrectness


def _item(called: list[str], expected: list[str]) -> DatasetItem:
    return DatasetItem(
        tools_called=[ToolCall(name=n, args={}) for n in called],
        expected_tools=[ToolCall(name=n, args={}) for n in expected],
    )


# ---------------------------------------------------------------------------
# Default behaviour — recall, unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_scoring_is_recall():
    """The default must not move: every existing call site inherits it."""
    metric = ToolCorrectness()
    assert metric.scoring == 'recall'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'called, expected, score',
    [
        (['a', 'b'], ['a', 'b'], 1.0),
        (['a'], ['a', 'b'], 0.5),
        (['x'], ['a', 'b'], 0.0),
        # The blind spot recall has by construction: three unasked-for calls
        # alongside the expected one still scores perfect.
        (['a', 'x', 'y', 'z'], ['a'], 1.0),
    ],
)
async def test_recall_ignores_extra_calls(called, expected, score):
    result = await ToolCorrectness().execute(_item(called, expected))
    assert result.score == score


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'called, expected, score',
    [
        (['a'], ['a'], 1.0),
        (['a', 'x'], ['a'], 0.5),
        (['a', 'x', 'y', 'z'], ['a'], 0.25),
        # Precision's own blind spot, the mirror of recall's: one correct call
        # and nothing extra is perfect even though a required tool was skipped.
        (['a'], ['a', 'b'], 1.0),
    ],
)
async def test_precision_penalises_over_calling(called, expected, score):
    result = await ToolCorrectness(scoring='precision').execute(_item(called, expected))
    assert result.score == score


@pytest.mark.asyncio
async def test_precision_with_no_calls_made():
    """No calls means none were wrong, so precision is 1.0 and recall carries it.

    Pinned because the alternative — a ZeroDivisionError — would surface as a
    metric crash on the most ordinary failure an agent has.
    """
    assert (
        await ToolCorrectness(scoring='precision').execute(_item([], ['a']))
    ).score == 1.0
    assert (
        await ToolCorrectness(scoring='recall').execute(_item([], ['a']))
    ).score == 0.0


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'called, expected, score',
    [
        (['a', 'b'], ['a', 'b'], 1.0),
        # precision 1/2, recall 1/1 → 2·.5·1/1.5
        (['a', 'x'], ['a'], pytest.approx(2 / 3)),
        # precision 1/1, recall 1/2 → same by symmetry
        (['a'], ['a', 'b'], pytest.approx(2 / 3)),
        (['x'], ['a'], 0.0),
    ],
)
async def test_f1_penalises_both_directions(called, expected, score):
    result = await ToolCorrectness(scoring='f1').execute(_item(called, expected))
    assert result.score == score


@pytest.mark.asyncio
async def test_f1_is_zero_when_either_term_is_zero():
    """Guards the harmonic mean's 0/0: precision 1.0 with recall 0.0 must not divide by zero."""
    result = await ToolCorrectness(scoring='f1').execute(_item([], ['a']))
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Shared behaviour across modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('scoring', ['recall', 'precision', 'f1'])
async def test_no_tools_expected_and_none_called_is_correct(scoring):
    result = await ToolCorrectness(scoring=scoring).execute(_item([], []))
    assert result.score == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize('scoring', ['recall', 'precision', 'f1'])
async def test_tools_called_when_none_expected_fails(scoring):
    """Answered before any matching runs, so it is mode-independent."""
    result = await ToolCorrectness(scoring=scoring).execute(_item(['a'], []))
    assert result.score == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize('scoring', ['recall', 'precision', 'f1'])
async def test_explanation_names_the_scoring_mode(scoring):
    """A bare score is ambiguous across modes; the report has to say which one ran."""
    result = await ToolCorrectness(scoring=scoring).execute(_item(['a', 'x'], ['a']))
    assert result.explanation.startswith(f'[{scoring}]')
    assert 'Unexpected tools' in result.explanation


@pytest.mark.asyncio
@pytest.mark.parametrize('scoring', ['recall', 'precision', 'f1'])
async def test_strict_order_is_unaffected_by_scoring(scoring):
    """Strict order is already all-or-nothing on both counts, so the mode is inert."""
    metric = ToolCorrectness(strict_order=True, scoring=scoring)
    assert (await metric.execute(_item(['a', 'b'], ['a', 'b']))).score == 1.0
    assert (await metric.execute(_item(['b', 'a'], ['a', 'b']))).score == 0.0
    assert (await metric.execute(_item(['a', 'b', 'x'], ['a', 'b']))).score == 0.0


@pytest.mark.asyncio
async def test_duplicate_calls_count_once_against_precision():
    """The same tool called twice matches one expectation; the second is unexpected."""
    result = await ToolCorrectness(scoring='precision').execute(
        _item(['a', 'a'], ['a'])
    )
    assert result.score == 0.5
