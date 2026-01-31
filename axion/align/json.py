from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from axion.align.ui import CaliberHQRenderer
from axion.dataset import Dataset

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler


class JsonCaliberHQRenderer(CaliberHQRenderer):
    """JSON-first renderer for CaliberHQ."""

    def annotate(self, dataset: Dataset) -> None:
        raise RuntimeError(
            'JsonCaliberHQRenderer does not support interactive annotation.'
        )

    def style_results(self, results_df: pd.DataFrame) -> 'Styler':
        return results_df.style

    def create_summary_stats_table(
        self, results_df: pd.DataFrame, alignment_score: float
    ) -> 'Styler':
        summary_df = pd.DataFrame(
            [
                ['Overall Alignment', f'{alignment_score:.1%}'],
                ['Aligned Items', f'{results_df["aligned"].sum()} / {len(results_df)}'],
            ],
            columns=['Metric', 'Value'],
        )
        return summary_df.style

    def display(self, summary_table: 'Styler', detailed_table: 'Styler') -> None:
        return None
