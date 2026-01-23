from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from axion.align.ui import AlignEvalRenderer
from axion.dataset import Dataset, DatasetItem

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler


class NotebookAlignEvalRenderer(AlignEvalRenderer):
    """Notebook-specific renderer for AlignEval."""

    def annotate(self, dataset: Dataset) -> None:
        """
        Interactively prompts the user for a score (0 or 1) and saves it
        directly to each item's `judgment` field. This prepares the dataset
        for the execute step.
        """
        print('🚀 Starting Interactive Human Annotation Process...')
        print('For each item, your score will be saved directly to the dataset item.')
        print('-' * 50)

        for i, item in enumerate(dataset.items):
            print(f'\n📝 Item {i + 1}/{len(dataset.items)} (ID: {item.id})')
            print(f'  - Query: {item.query}')
            print(f'  - Expected Output: {item.expected_output}')
            print(f'  - Actual Output: {item.actual_output}')

            while True:
                score_input = input(
                    '  ➡️ Enter judgment (1 for correct/relevant, 0 for incorrect): '
                )
                if score_input in ['0', '1']:
                    item.judgment = score_input
                    break
                else:
                    print("  ❌ Invalid input. Please enter only '0' or '1'.")

        print('\n✅ Human annotation complete! You can now run .execute()')

    def style_results(self, results_df: pd.DataFrame) -> "Styler":
        """
        Applies advanced conditional highlighting and styling to the results DataFrame.
        """
        df = results_df

        def highlight_alignment_scores(row):
            styles = pd.Series('', index=row.index)
            is_aligned = row['aligned']

            if is_aligned:
                score_style = (
                    'background: linear-gradient(135deg, #10b981, #059669); '
                    'color: white; font-weight: bold;'
                )
            else:
                score_style = (
                    'background: linear-gradient(135deg, #ef4444, #dc2626); '
                    'color: white; font-weight: bold;'
                )

            styles['human_score'] = score_style
            styles['llm_score'] = score_style

            diff = row['score_difference']
            if diff > 0:
                styles['score_difference'] = (
                    f'background-color: rgba(239, 68, 68, {0.3 + diff * 0.4}); '
                    'color: #7f1d1d; font-weight: bold; text-align: center;'
                )
            else:
                styles['score_difference'] = (
                    'background-color: rgba(16, 185, 129, 0.2); '
                    'color: #064e3b; font-weight: bold; text-align: center;'
                )
            return styles

        def format_scores(val):
            return (
                f'✅ {int(val)}'
                if val == 1
                else f'❌ {int(val)}'
                if val == 0
                else '❓ N/A'
            )

        def format_difference(val):
            return '🎯 0' if val == 0 else f'⚠️ {val:.1f}'

        def truncate_text(text, max_length=100):
            return (
                str(text)[: max_length - 3] + '...'
                if len(str(text)) > max_length
                else str(text)
            )

        return (
            df.style.apply(highlight_alignment_scores, axis=1)
            .format(
                {
                    'human_score': format_scores,
                    'llm_score': format_scores,
                    'score_difference': format_difference,
                    'query': lambda x: truncate_text(x, 80),
                    'actual_output': lambda x: truncate_text(x, 120),
                    'expected_output': lambda x: truncate_text(x, 120),
                    'llm_explanation': lambda x: truncate_text(x, 150),
                }
            )
            .set_table_styles(
                [
                    {
                        'selector': 'table',
                        'props': [
                            ('border-collapse', 'collapse'),
                            ('width', '100%'),
                            ('font-family', 'sans-serif'),
                        ],
                    },
                    {
                        'selector': 'thead th',
                        'props': [
                            ('background', '#667eea'),
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('padding', '12px'),
                            ('text-align', 'left'),
                        ],
                    },
                    {
                        'selector': 'tbody tr:hover',
                        'props': [('background-color', '#f5f5f5')],
                    },
                    {
                        'selector': 'tbody td',
                        'props': [('padding', '10px'), ('border', '1px solid #ddd')],
                    },
                ]
            )
            .set_properties(**{'text-align': 'left'})
            .hide(subset=['aligned'], axis=1)
        )

    def create_summary_stats_table(
        self, results_df: pd.DataFrame, alignment_score: float
    ) -> "Styler":
        """Creates a beautifully styled summary statistics table."""
        total_items = len(results_df)
        aligned_items = results_df['aligned'].sum()
        misaligned_items = total_items - aligned_items

        summary_data = [
            ['🎯 Overall Alignment', f'{alignment_score:.1%}'],
            ['✅ Aligned Items', f'{aligned_items} / {total_items}'],
            ['⚠️ Misaligned Items', f'{misaligned_items} / {total_items}'],
        ]
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])

        return (
            summary_df.style.hide(axis='index')
            .set_table_styles(
                [
                    {
                        'selector': 'table',
                        'props': [('width', '50%'), ('margin', '20px 0')],
                    },
                    {
                        'selector': 'td',
                        'props': [('padding', '10px'), ('font-size', '14px')],
                    },
                ]
            )
            .set_caption('📈 Alignment Summary')
            .set_properties(**{'text-align': 'left', 'font-weight': 'bold'})
        )

    def display(self, summary_table: "Styler", detailed_table: "Styler") -> None:
        """Display tables in notebook environment."""
        from IPython.display import display

        display(summary_table)
        print('\nDetailed Comparison:')
        display(detailed_table)
