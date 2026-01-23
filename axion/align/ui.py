from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from axion.dataset import Dataset

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler


class AlignEvalRenderer(ABC):
    """UI adapter surface for AlignEval."""

    @abstractmethod
    def annotate(self, dataset: Dataset) -> None:
        """Collect human judgments and write to dataset items."""

    @abstractmethod
    def style_results(self, results_df: pd.DataFrame) -> "Styler":
        """Return a styled table for detailed results."""

    @abstractmethod
    def create_summary_stats_table(
        self, results_df: pd.DataFrame, alignment_score: float
    ) -> "Styler":
        """Return a styled table for summary statistics."""

    @abstractmethod
    def display(self, summary_table: "Styler", detailed_table: "Styler") -> None:
        """Render summary and detailed tables in the UI."""
