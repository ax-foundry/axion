from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from axion._core.asyncio import run_async_function
from axion.align.base import BaseCaliberHQ
from axion.dataset import Dataset, DatasetItem
from axion.metrics.base import BaseMetric


class WebCaliberHQ(BaseCaliberHQ):
    """API-friendly CaliberHQ for web UIs."""

    def __init__(self, dataset: Dataset, metric: BaseMetric):
        super().__init__(dataset, metric)

    def execute(
        self,
        as_dict: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any] | pd.DataFrame:
        """Run evaluation and return JSON-serializable results."""

        async def _run() -> None:
            await self._run_evals_async(on_progress=on_progress)

        run_async_function(_run)
        self._prepare_results_df()
        return self.to_dict() if as_dict else self.results_df

    @classmethod
    def from_records(cls, records: List[dict], metric: BaseMetric) -> 'WebCaliberHQ':
        """Create from list of dicts (from web upload)."""
        items = [DatasetItem(**record) for record in records]
        dataset = Dataset(items=items)
        return cls(dataset, metric)
