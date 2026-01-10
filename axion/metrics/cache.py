import asyncio
from collections import defaultdict
from typing import Any, Dict, Optional

from axion._core.logging import get_logger
from cachetools import LRUCache

logger = get_logger(__name__)


class AnalysisCache:
    """
    A production-grade, item-centric cache for shared metric computations.

    This cache stores analysis results (e.g., 'moments', 'claims') on a per-item
    basis, identified by the DatasetItem's unique ID. It uses an LRU (Least
    Recently Used) policy to manage memory, ensuring it's safe for large-scale
    evaluation runs. It also includes an internal locking mechanism to prevent
    redundant computations in highly concurrent scenarios.
    """

    def __init__(self, max_items: int = 500):
        self._cache: LRUCache[str, Dict[str, Any]] = LRUCache(maxsize=max_items)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def get_lock(self, item_id: str, analysis_name: str) -> asyncio.Lock:
        """Gets a specific lock for a given item and analysis combination."""
        return self._locks[f'{item_id}:{analysis_name}']

    def _get_item_cache(self, item_id: str) -> Dict[str, Any]:
        if item_id not in self._cache:
            self._cache[item_id] = {}
        else:
            logger.debug(f'Cache hit for {item_id}', self._cache[item_id])
        return self._cache[item_id]

    def get(self, item_id: str, analysis_name: str) -> Optional[Any]:
        item_cache = self._get_item_cache(item_id)
        return item_cache.get(analysis_name)

    def set(self, item_id: str, analysis_name: str, value: Any) -> None:
        item_cache = self._get_item_cache(item_id)
        item_cache[analysis_name] = value
        self._cache[item_id] = item_cache

    def has(self, item_id: str, analysis_name: str) -> bool:
        if item_id not in self._cache:
            return False
        return analysis_name in self._cache[item_id]

    def clear(self) -> None:
        self._cache.clear()
        self._locks.clear()
