from typing import Optional

from axion.eval_tree.aggregation import (
    AggregationStrategy,
    WeightedAverage,
)
from axion.eval_tree.node import Node


class ComponentNode(Node):
    """
    Internal evaluation node that represents a group of child nodes.

    In the optimized two-phase model, this class acts as a structural container,
    holding its children, weight, and the aggregation strategy to be used during
    the in-memory aggregation phase within the EvalTree. It no longer contains
    any execution logic itself.
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Node] = None,
        weight: float = 1.0,
        aggregation_strategy: AggregationStrategy = WeightedAverage(),
        exclude_failed_metrics: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, parent, weight, **kwargs)
        self.aggregation_strategy = aggregation_strategy
        self.exclude_failed_metrics = exclude_failed_metrics

    @property
    def aggregation_strategy_name(self) -> str:
        """Get the class name of the aggregation strategy."""
        return self.aggregation_strategy.__class__.__name__

    def __repr__(self) -> str:
        children_names = [child.name for child in self.children]
        return (
            f'ComponentNode(name={self.name}, strategy={self.aggregation_strategy_name}, '
            f'children={len(children_names)})'
        )
