from typing import Any, Optional

from axion._core.logging import get_logger
from axion.eval_tree.node import Node

logger = get_logger(__name__)


class MetricNode(Node):
    """
    Leaf node that wraps a metric instance.

    This class acts as a container for the metric that will be executed by the
    MetricRunner during the batch calculation phase. It also holds configuration
    like its weight and bias. The execution logic is handled by the MetricRunner,
    not this node.
    """

    def __init__(
        self,
        name: str,
        metric: Any,
        parent: Optional[Node] = None,
        weight: float = 1.0,
        bias: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(name, parent, weight, **kwargs)
        self.metric = metric
        self.bias = bias

    def __repr__(self) -> str:
        return (
            f'MetricNode(name={self.name}, metric={self.metric.__class__.__name__}, '
            f'weight={self.weight}, bias={self.bias:.3f})'
        )
