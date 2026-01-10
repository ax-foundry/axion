from typing import Optional, Union

from axion.eval_tree._node import NodeMixin


class Node(NodeMixin):
    """Base node class with name and weight properties."""

    def __init__(
        self,
        name: str,
        parent: Optional[NodeMixin] = None,
        weight: Union[str, float] = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = name
        self.parent = parent
        self.weight = float(weight) if weight is not None else 1.0
        self.kwargs = kwargs
