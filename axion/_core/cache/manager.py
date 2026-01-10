from typing import Any, Optional

from axion._core.cache.schema import CacheConfig


class CacheManager:
    """Manages cache operations for both memory and disk, abstracting the backend."""

    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._initialize_cache()

    def _initialize_cache(self):
        """Configure and return a cache instance based on the provided configuration."""
        if not self.config.use_cache:
            self.cache = None
            return

        if self.config.cache_type == 'disk':
            if not self.config.cache_dir:
                raise ValueError('cache_dir must be provided for disk cache.')
            try:
                import diskcache

                self.cache = diskcache.Cache(self.config.cache_dir)
            except ImportError as e:
                raise ImportError(
                    "diskcache is required for disk caching. Please run 'pip install diskcache'"
                ) from e
        elif self.config.cache_type == 'memory':
            self.cache = {}
        else:
            raise ValueError(f'Unsupported cache type: {self.config.cache_type}')

    def get(self, key: str) -> Optional[Any]:
        """Gets an item from the cache if use_cache is True."""
        if self.cache is not None and self.config.use_cache:
            return self.cache.get(key, None)
        return None

    def set(self, key: str, value: Any):
        """Sets an item in the cache if write_cache is True."""
        if self.cache is not None and self.config.write_cache:
            # For diskcache
            if hasattr(self.cache, 'set'):
                self.cache.set(key, value)
            # For dict
            else:
                self.cache[key] = value

    def close(self):
        """Closes the cache connection if applicable (for diskcache)."""
        if self.cache and hasattr(self.cache, 'close'):
            self.cache.close()
