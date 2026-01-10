from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheConfig:
    """
    Configuration class for controlling caching behavior of metric evaluations.

    Attributes:
        use_cache (bool):
            If True, attempts to read previously computed results from cache to avoid redundant computation.

        write_cache (bool):
            If True, writes newly computed metric results to cache for future use.
            Has no effect if `use_cache` is False.

        cache_type (str):
            Type of caching backend to use.
            - 'memory': Uses in-memory dictionary for caching (fast, but non-persistent).
            - 'disk': Writes cache to disk (persistent across runs).

        cache_dir (Optional[str]):
            Directory path where disk cache files will be stored.
            Only used when `cache_type='disk'`. Defaults to '.cache'.

        cache_task (bool):
            If True, enables caching at the task level (e.g., for full evaluation runs).
            If False, caching applies only at the metric level.
    """

    use_cache: bool = True
    write_cache: bool = True
    cache_type: str = 'memory'  # Options: 'memory' or 'disk'
    cache_dir: Optional[str] = '.cache'
    cache_task: bool = True
