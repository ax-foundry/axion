from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from axion._core.schema import RichEnum
from axion.dataset import Dataset
from axion.runners.metric import MetricRunner
from axion.schema import TestResult


class BaseScoringStrategy(ABC):
    """Abstract base class for a scoring strategy."""

    @abstractmethod
    async def execute(
        self, dataset: Dataset, show_progress: bool = True
    ) -> List[TestResult]:
        """Executes the scoring strategy against a dataset."""
        pass

    @property
    @abstractmethod
    def summary(self) -> Union[Dict[str, Any], None]:
        """Returns the summary of the last run."""
        pass


class FlatScoringStrategy(BaseScoringStrategy):
    """Scoring strategy for a flat list of metrics using MetricRunner."""

    def __init__(self, metrics: List[Any], **kwargs):
        """
        Initializes the strategy with a list of metrics.

        Args:
            metrics (List[Any]): A list of metric objects.
            **kwargs: Additional arguments to pass to the MetricRunner,
                      e.g., max_concurrent, summary_generator, etc.
        """
        self._runner = MetricRunner(metrics=metrics, **kwargs)

    async def execute(
        self, dataset: Dataset, show_progress: bool = True
    ) -> List[TestResult]:
        """Executes the flat metric evaluation."""
        return await self._runner.execute_batch(dataset, show_progress=show_progress)

    @property
    def summary(self) -> Union[Dict[str, Any], None]:
        """Returns the summary from the underlying MetricRunner."""
        return self._runner.summary


class HierarchicalScoringStrategy(BaseScoringStrategy):
    """Scoring strategy for a hierarchical tree configuration using EvalTree."""

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the strategy with a hierarchical model configuration.

        Args:
            config (Dict[str, Any]): The configuration dictionary for EvalTree.
            **kwargs: Additional arguments to pass to the EvalTree,
                      e.g., max_concurrent, summary_generator, trace_granularity.
        """
        from axion.eval_tree.tree import EvalTree

        self._tree = EvalTree(config=config, **kwargs)

    async def execute(
        self, dataset: Dataset, show_progress: bool = True
    ) -> List[TestResult]:
        """Executes the hierarchical evaluation."""
        result_wrapper = await self._tree.batch_execute(
            dataset, show_progress=show_progress
        )
        return result_wrapper.results

    @property
    def summary(self) -> Union[Dict[str, Any], None]:
        """Returns the summary from the underlying EvalTree."""
        return self._tree.summary

    @property
    def tree(self) -> Any:
        """Returns the underlying EvalTree instance for inspection."""
        return self._tree


class ScoringStrategyType(str, RichEnum):
    """Enumeration for the available scoring strategy aliases."""

    FLAT = 'flat'
    TREE = 'tree'

    @classmethod
    def values(cls) -> List[str]:
        """Returns a list of all possible string values for the enum."""
        return [item.value for item in cls]
