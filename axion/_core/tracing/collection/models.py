from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from axion._core.tracing.collection.smart_access import SmartAccess, _normalize_key

class ModelUsageUnit(Enum):
    TOKENS = 'TOKENS'
    CHARACTERS = 'CHARACTERS'
    MILLISECONDS = 'MILLISECONDS'
    SECONDS = 'SECONDS'
    IMAGES = 'IMAGES'
    REQUESTS = 'REQUESTS'


class ObservationLevel(Enum):
    DEFAULT = 'DEFAULT'
    DEBUG = 'DEBUG'
    WARNING = 'WARNING'
    ERROR = 'ERROR'


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    total: int = 0
    unit: Any = ModelUsageUnit.TOKENS


def _obj_to_dict(obj: Any) -> Dict:
    """Convert a raw SDK object or dict into a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, '__dict__'):
        return vars(obj)
    return {}


class TraceView(SmartAccess):
    """
    Wraps the root trace object.

    Holds attributes like id, latency, environment, and the list of observations.
    Accepts either keyword arguments or a single raw object/dict.
    """

    def __init__(self, data: Any = None, **kwargs):
        if data is not None:
            self._data = _obj_to_dict(data)
        else:
            self._data = kwargs
        # Ensure observations is at least an empty list
        if 'observations' not in self._data:
            self._data['observations'] = []

    def _lookup(self, key: str) -> Any:
        return self._data[key]

    def _lookup_insensitive(self, key: str) -> Any:
        target = _normalize_key(key)
        for k, v in self._data.items():
            if _normalize_key(k) == target:
                return v
        return None

    def __repr__(self):
        return f"TraceView(id='{self._data.get('id', 'N/A')}', name='{self._data.get('name', 'N/A')}')"


class ObservationsView(SmartAccess):
    """
    Wraps a single observation (Span/Generation).

    Accepts either keyword arguments or a single raw object/dict.
    """

    def __init__(self, data: Any = None, **kwargs):
        if data is not None:
            self._data = _obj_to_dict(data)
        else:
            self._data = kwargs


    def _lookup(self, key: str) -> Any:
        return self._data[key]

    def _lookup_insensitive(self, key: str) -> Any:
        target = _normalize_key(key)
        for k, v in self._data.items():
            if _normalize_key(k) == target:
                return v
        return None

    def __repr__(self):
        name = self._data.get('name', 'unnamed')
        type_ = self._data.get('type', 'unknown')
        return f"ObservationsView(name='{name}', type='{type_}')"
