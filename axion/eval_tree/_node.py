from __future__ import annotations

from typing import List, Optional


class InvalidNode(Exception):
    pass


class NodeMixin:
    """Base mixin for tree node functionality."""

    def __init__(self) -> None:
        self._parent: Optional[NodeMixin] = None
        self._children: List[NodeMixin] = []

    @property
    def parent(self) -> Optional[NodeMixin]:
        return self._parent

    @parent.setter
    def parent(self, parent: Optional[NodeMixin]) -> None:
        if parent is self:
            raise InvalidNode('Cannot create infinite tree.')
        self._detach()
        self._attach(parent)

    @property
    def children(self) -> List[NodeMixin]:
        return self._children

    def _attach(self, parent: Optional[NodeMixin]) -> None:
        if parent is not None:
            parent.children.append(self)
        self._parent = parent

    def _detach(self) -> None:
        if self.parent is None:
            return
        if self in self.parent._children:
            self.parent._children.remove(self)
        self._parent = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    @property
    def root(self) -> NodeMixin:
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def descendants(self) -> List[NodeMixin]:
        """Get all descendants including self."""
        result = [self]
        for child in self.children:
            result.extend(child.descendants())
        return result
