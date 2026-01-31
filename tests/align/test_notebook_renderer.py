import pandas as pd
from pandas.io.formats.style import Styler

from axion.align.notebook import NotebookCaliberHQRenderer


def _sample_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'id': 'item-1',
                'aligned': True,
                'human_score': 1,
                'llm_score': 1,
                'score_difference': 0,
                'query': 'What is the capital of France?',
                'actual_output': 'Paris.',
                'expected_output': 'Paris',
                'llm_explanation': 'The answer is correct.',
            },
            {
                'id': 'item-2',
                'aligned': False,
                'human_score': 0,
                'llm_score': 1,
                'score_difference': 1,
                'query': 'What is 2+2?',
                'actual_output': '5.',
                'expected_output': '4',
                'llm_explanation': 'The answer is incorrect.',
            },
        ]
    )


def test_create_summary_stats_table_returns_styler() -> None:
    renderer = NotebookCaliberHQRenderer()
    results_df = _sample_results_df()

    summary_table = renderer.create_summary_stats_table(results_df, alignment_score=0.5)

    assert isinstance(summary_table, Styler)
    assert list(summary_table.data.columns) == ['Metric', 'Value']
    assert summary_table.data.iloc[0, 0] == '🎯 Overall Alignment'


def test_style_results_returns_styler() -> None:
    renderer = NotebookCaliberHQRenderer()
    results_df = _sample_results_df()

    detailed_table = renderer.style_results(results_df)

    assert isinstance(detailed_table, Styler)
    assert detailed_table.data.equals(results_df)
