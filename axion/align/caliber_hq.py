from typing import Any, Callable, Dict, Optional

import pandas as pd

from axion._core.asyncio import run_async_function
from axion.align.base import BaseCaliberHQ
from axion.align.notebook import NotebookCaliberHQRenderer
from axion.align.ui import CaliberHQRenderer
from axion.dataset import Dataset
from axion.metrics.base import BaseMetric


class CaliberHQ(BaseCaliberHQ):
    """
    A class to calibrate LLM-as-a-judge evaluators by comparing them
    against human-provided scores. It supports a clear, step-by-step workflow:
    1. Initialize: CaliberHQ(dataset, metric)
    2. Annotate: .annotate() (optional, if data is not pre-annotated)
    3. Execute: .execute()

    This class extends BaseCaliberHQ with notebook-specific UI functionality.
    """

    def __init__(
        self,
        dataset: Dataset,
        metric: BaseMetric,
        renderer: CaliberHQRenderer | None = None,
    ):
        """
        Initializes the CaliberHQ class.

        Args:
            dataset (Dataset): The dataset containing items to be evaluated.
            metric (BaseMetric): The LLM-as-a-judge metric to be calibrated.
        """
        super().__init__(dataset, metric)
        self.renderer = renderer or NotebookCaliberHQRenderer()

    def annotate(self) -> None:
        """
        Interactively prompts the user for a score (0 or 1) and saves it
        directly to each item's `judgment` field. This prepares the dataset
        for the execute step.
        """
        self.renderer.annotate(self.dataset)

    def execute(
        self,
        as_dict: bool = False,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame | Dict[str, Any]:
        """
        Executes the alignment workflow: runs evaluations, prepares results,
        and displays styled summary and detailed tables.

        Returns:
            pd.DataFrame | Dict: The detailed results as a DataFrame or JSON dict.
        """
        print('Running LLM-as-a-judge evaluation...')

        async def _run() -> None:
            await self._run_evals_async(on_progress=on_progress)

        run_async_function(_run)
        print('LLM-as-a-judge evaluation complete!')

        self._prepare_results_df()

        print('\n' + '=' * 50)
        print('📊 Calibration Evaluation Results')
        print('=' * 50)

        summary_table = self.renderer.create_summary_stats_table(
            self.results_df, self.alignment_score
        )
        detailed_table = self.renderer.style_results(self.results_df)
        self.renderer.display(summary_table, detailed_table)

        return self.to_dict() if as_dict else self.results_df
