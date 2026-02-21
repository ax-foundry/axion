from typing import Any, Dict

from axion._core.logging import get_logger

logger = get_logger(__name__)


def _normalize_key(key: str) -> str:
    """Normalize keys for fuzzy matching (snake_case -> camelCase support)."""
    return key.lower().replace('_', '')


class SmartAccess:
    """
    Base class that allows dictionary values to be accessed via dot notation.

    Recursively wraps returned dictionaries, lists, and objects so that
    nested structures remain navigable with attribute syntax.

    Subclasses must implement ``_lookup()`` and optionally ``_lookup_insensitive()``.
    """

    def __getattr__(self, key: str) -> Any:
        # 1. Exact match via subclass _lookup
        try:
            val = self._lookup(key)
            return self._wrap(val)
        except (KeyError, AttributeError, NotImplementedError):
            pass

        # 2. Case/separator-insensitive fallback (e.g. .product_type -> productType)
        val = self._lookup_insensitive(key)
        if val is not None:
            logger.debug('Fuzzy match: %r resolved via insensitive lookup', key)
            return self._wrap(val)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __getitem__(self, key: Any) -> Any:
        """Support bracket syntax: obj['key']."""
        return self._wrap(self._lookup(key))

    def _lookup(self, key: str) -> Any:
        """Subclasses must implement how to fetch raw data."""
        raise NotImplementedError

    def _lookup_insensitive(self, key: str) -> Any:
        """Optional hook for fuzzy matching."""
        return None

    def _wrap(self, val: Any) -> Any:
        """Recursively wrap results to ensure dot-notation connectivity."""
        if isinstance(val, dict):
            return SmartDict(val)
        if isinstance(val, list):
            return [self._wrap(x) for x in val]
        # Wrap generic objects (with attributes) that aren't already SmartAccess
        if hasattr(val, '__dict__') and not isinstance(val, SmartAccess):
            return SmartObject(val)
        return val


class SmartDict(SmartAccess):
    """Wraps a standard dictionary to allow dot access with fuzzy matching."""

    def __init__(self, data: Dict):
        self._data = data

    def _lookup(self, key: str) -> Any:
        return self._data[key]

    def _lookup_insensitive(self, key: str) -> Any:
        target = _normalize_key(key)
        for k, v in self._data.items():
            if _normalize_key(k) == target:
                return v
        return None

    def __repr__(self):
        return f'<SmartDict keys={list(self._data.keys())}>'

    def to_dict(self) -> Dict:
        """Return the underlying raw dictionary."""
        return self._data


class SmartObject(SmartAccess):
    """Wraps a generic Python object to ensure its attributes return Smart wrappers."""

    def __init__(self, obj: Any):
        self._obj = obj

    def _lookup(self, key: str) -> Any:
        return getattr(self._obj, key)

    def _lookup_insensitive(self, key: str) -> Any:
        target = _normalize_key(key)
        for k in dir(self._obj):
            if _normalize_key(k) == target:
                return getattr(self._obj, k)
        return None

    def __repr__(self):
        return repr(self._obj)
