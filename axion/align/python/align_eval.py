from typing import TypeVar

import pandas as pd
from axion._core.asyncio import run_async_function
from axion.align.base import BaseAlignEval
from axion.dataset import Dataset
from axion.metrics.base import BaseMetric

T = TypeVar('T')


class PythonAlignEval(BaseAlignEval):
    """
    A class to calibrate LLM-as-a-judge evaluators by comparing them
    against human-provided scores. It supports a clear, step-by-step workflow:
    1. Initialize: AlignEval(dataset, metric)
    2. Annotate: .annotate() (optional, if data is not pre-annotated)
    3. Execute: .execute()

    This class extends BaseAlignEval with notebook-specific UI functionality.
    """

    def __init__(self, dataset: Dataset, metric: BaseMetric):
        """
        Initializes the NotebookAlignEval class.

        Args:
            dataset (Dataset): The dataset containing items to be evaluated.
            metric (BaseMetric): The LLM-as-a-judge metric to be calibrated.
        """
        super().__init__(dataset, metric)

    def annotate(self) -> None:
        """
        Interactively prompts the user for a score (0 or 1) and saves it
        directly to each item's `judgment` field. This prepares the dataset
        for the execute step.
        """
        print('🚀 Starting Interactive Human Annotation Process...')
        print('For each item, your score will be saved directly to the dataset item.')
        print('-' * 50)

        for i, item in enumerate(self.dataset.items):
            print(f'\n📝 Item {i + 1}/{len(self.dataset.items)} (ID: {item.id})')
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

    def _style_results(self) -> 'pd.io.formats.style.Styler':
        """
        Applies advanced conditional highlighting and styling to the results DataFrame.
        """
        df = self.results_df

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

    def _create_summary_stats_table(self) -> 'pd.io.formats.style.Styler':
        """Creates a beautifully styled summary statistics table."""
        df = self.results_df
        total_items = len(df)
        aligned_items = df['aligned'].sum()
        misaligned_items = total_items - aligned_items

        summary_data = [
            ['🎯 Overall Alignment', f'{self.alignment_score:.1%}'],
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

    def _display_notebook(self, summary_table, detailed_table):
        """Display tables in notebook environment."""
        from IPython.display import display

        display(summary_table)
        print('\nDetailed Comparison:')
        display(detailed_table)

    def execute(self) -> pd.DataFrame:
        """
        Executes the alignment workflow: runs evaluations, prepares results,
        and displays styled summary and detailed tables.

        Returns:
            pd.DataFrame: The raw, unstyled DataFrame containing the detailed results.
        """
        print('🤖 Running LLM-as-a-judge evaluation...')
        run_async_function(self._run_evals_async)
        print('✅ LLM evaluation complete!')

        self._prepare_results_df()

        print('\n' + '=' * 50)
        print('📊 Alignment Evaluation Results')
        print('=' * 50)

        summary_table = self._create_summary_stats_table()
        detailed_table = self._style_results()
        self._display_notebook(summary_table, detailed_table)

        return self.results_df
