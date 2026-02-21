from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Generator, Iterator, List, Optional

from axion._core.tracing.collection.smart_access import SmartAccess, _normalize_key


class ObservationNode(SmartAccess):
    """
    Tree node wrapping a single observation with parent/child references.

    Reconstructs the hierarchy visible in the Langfuse timeline from
    ``parent_observation_id`` fields. SmartAccess delegates to the
    underlying :class:`ObservationsView` so dot-notation still works::

        node.name          # observation name
        node.input         # observation input
        node.children[0]   # first child ObservationNode
    """

    def __init__(self, observation: Any) -> None:
        self._observation = observation
        self._parent: Optional[ObservationNode] = None
        self._children: List[ObservationNode] = []

    # -- tree structure ------------------------------------------------------

    @property
    def observation(self) -> Any:
        """The underlying ObservationsView (or dict/object)."""
        return self._observation

    @property
    def parent(self) -> Optional[ObservationNode]:
        return self._parent

    @property
    def children(self) -> List[ObservationNode]:
        return list(self._children)

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def depth(self) -> int:
        d = 0
        node = self._parent
        while node is not None:
            d += 1
            node = node._parent
        return d

    # -- timing --------------------------------------------------------------

    @property
    def start_time(self) -> Optional[datetime]:
        return _parse_datetime(_safe_obs_get(self._observation, 'start_time'))

    @property
    def end_time(self) -> Optional[datetime]:
        return _parse_datetime(_safe_obs_get(self._observation, 'end_time'))

    @property
    def duration(self) -> Optional[timedelta]:
        s, e = self.start_time, self.end_time
        if s is not None and e is not None:
            return e - s
        return None

    # -- traversal -----------------------------------------------------------

    def walk(self) -> Generator[ObservationNode, None, None]:
        """Pre-order depth-first traversal of this subtree."""
        yield self
        for child in self._children:
            yield from child.walk()

    # -- search / navigation -------------------------------------------------

    def find(
        self,
        name: Optional[str] = None,
        type: Optional[str] = None,
    ) -> Optional[ObservationNode]:
        """Return the first descendant matching *name* and/or *type*, or ``None``."""
        for node in self.walk():
            if node is self:
                continue
            if name is not None and _safe_obs_get(node._observation, 'name') != name:
                continue
            if type is not None and _safe_obs_get(node._observation, 'type') != type:
                continue
            return node
        return None

    def __getitem__(self, key: str) -> Any:
        """Search subtree by name first; fall back to observation field lookup.

        ``root['recommendation:ai.generateText']`` finds the descendant node.
        ``root['id']`` returns the observation field when no child has that name.
        Raises ``KeyError`` when neither match.
        """
        # 1. Search descendants by name
        for node in self.walk():
            if node is self:
                continue
            if _safe_obs_get(node._observation, 'name') == key:
                return node

        # 2. Fall back to observation field lookup
        try:
            return self._wrap(self._lookup(key))
        except (KeyError, AttributeError, NotImplementedError):
            pass

        raise KeyError(key)

    def __iter__(self) -> Iterator[ObservationNode]:
        """Iterate over direct children."""
        return iter(self._children)

    def __len__(self) -> int:
        """Number of direct children."""
        return len(self._children)

    def __contains__(self, name: object) -> bool:
        """Check if any descendant in the subtree has the given name."""
        if not isinstance(name, str):
            return False
        for node in self.walk():
            if node is self:
                continue
            if _safe_obs_get(node._observation, 'name') == name:
                return True
        return False

    # -- SmartAccess ---------------------------------------------------------

    def _lookup(self, key: str) -> Any:
        return self._observation._lookup(key)

    def _lookup_insensitive(self, key: str) -> Any:
        if hasattr(self._observation, '_lookup_insensitive'):
            return self._observation._lookup_insensitive(key)
        target = _normalize_key(key)
        for attr in dir(self._observation):
            if _normalize_key(attr) == target:
                return getattr(self._observation, attr)
        return None

    # -- internal ------------------------------------------------------------

    def _add_child(self, child: ObservationNode) -> None:
        child._parent = self
        self._children.append(child)

    def __repr__(self) -> str:
        name = _safe_obs_get(self._observation, 'name') or 'unnamed'
        type_ = _safe_obs_get(self._observation, 'type') or 'unknown'
        n_children = len(self._children)
        return f"<ObservationNode name='{name}' type='{type_}' children={n_children}>"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_obs_get(obs: Any, field: str) -> Any:
    """Safely extract a field from an observation (SmartAccess, dict, or object)."""
    try:
        return obs._lookup(field)
    except (KeyError, AttributeError, TypeError, NotImplementedError):
        return getattr(obs, field, None)


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Coerce a value to datetime. Accepts datetime, ISO-8601 string, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
    return None
