from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from axion.align.ui import AlignEvalRenderer
from axion.dataset import Dataset

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler


class ConsoleAlignEvalRenderer(AlignEvalRenderer):
    """Console renderer for AlignEval."""

    def annotate(self, dataset: Dataset) -> None:
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
        return results_df.style

    def create_summary_stats_table(
        self, results_df: pd.DataFrame, alignment_score: float
    ) -> "Styler":
        summary_df = pd.DataFrame(
            [
                ['Overall Alignment', f'{alignment_score:.1%}'],
                ['Aligned Items', f"{results_df['aligned'].sum()} / {len(results_df)}"],
            ],
            columns=['Metric', 'Value'],
        )
        return summary_df.style

    def display(self, summary_table: "Styler", detailed_table: "Styler") -> None:
        print('\nSummary:')
        print(summary_table.data)
        print('\nDetailed Results:')
        print(detailed_table.data)
