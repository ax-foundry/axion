import asyncio
import functools
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union, cast

F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])


class Timer:
    """
    A timer class to record execution time
    and human-readable timestamps using the time module.

    Usage:
        with Timer() as timer:
            agent.achat(...)
        print(timer.elapsed_time)       # e.g., 1.234
        print(timer.start_timestamp)    # e.g., '2025-05-13 18:42:01'
        print(timer.end_timestamp)      # e.g., '2025-05-13 18:42:02'
    """

    def __init__(self):
        self._elapsed_time: Optional[float] = None
        self._start_time: Optional[float] = None
        self.start_timestamp: Optional[str] = None
        self.end_timestamp: Optional[str] = None
        self.start()

    @property
    def elapsed_time(self) -> Optional[float]:
        """Return the last recorded elapsed time, rounded to milliseconds."""
        return round(self._elapsed_time, 3) if self._elapsed_time is not None else None

    @staticmethod
    def current_timestamp() -> str:
        """Return the current time as a human-readable string."""
        return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())

    def start(self) -> None:
        """Start or restart the timer."""
        self._start_time = time.perf_counter()
        self.start_timestamp = self.current_timestamp()

    def stop(self) -> None:
        """Stop the timer and store the elapsed time and human-readable end timestamp."""
        now = time.perf_counter()
        if self._start_time is not None:
            self._elapsed_time = now - self._start_time
        self.end_timestamp = self.current_timestamp()

    def __enter__(self) -> 'Timer':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        return None


def log_execution_time(func: Union[F, AsyncF]) -> Union[F, AsyncF]:
    """
    Decorator that logs the execution time of a function and stores it as an attribute.
    Works with both synchronous and asynchronous functions.
    Uses the Timer class for consistent UTC-based tracking.

    Args:
        func: The function to be wrapped, can be sync or async

    Returns:
        Wrapped function with execution time logging
    """

    from axion._core.logging import get_logger

    logger = get_logger(__name__)

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs) -> Any:
            with Timer() as timer:
                result = await func(self, *args, **kwargs)
            self._execution_time = timer.elapsed_time
            # Log the execution time
            logger.info(f'{func.__name__} executed in {timer.elapsed_time:.4f} seconds')
            return result

        return cast(AsyncF, async_wrapper)
    else:

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs) -> Any:
            with Timer() as timer:
                result = func(self, *args, **kwargs)
            self._execution_time = timer.elapsed_time
            # Log the execution time
            logger.info(f'{func.__name__} executed in {timer.elapsed_time:.4f} seconds')
            return result

        return cast(F, sync_wrapper)


def base_model_dump_json(model, indent: int = 2) -> str:
    """
    Serializes a model to a JSON string with optional indentation.

    If the model defines a `clean_model_dump_json` method, it will be used
    to produce a cleaned JSON output (e.g., excluding empty or null fields).
    Otherwise, it falls back to the standard `model_dump_json` method.

    Args:
        model: The model instance to serialize. Expected to be a Pydantic model.
        indent (int): Number of spaces to use for indentation in the output JSON. Defaults to 2.

    Returns:
        str: The serialized JSON string.
    """
    dump_method = getattr(model, 'clean_model_dump_json', None)
    if callable(dump_method):
        return dump_method(indent=indent)
    return model.model_dump_json(indent=indent)
