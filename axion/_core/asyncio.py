import asyncio
import functools
import threading
from concurrent.futures import Executor
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)

T = TypeVar('T')


class SemaphoreExecutor:
    """Runs functions asynchronously with a limit on concurrent executions.

    This class provides a simple way to manage concurrency for a mix of
    synchronous (blocking) and asynchronous functions. It uses an
    `asyncio.Semaphore` to ensure that no more than `max_concurrent`
    operations are running at the same time.

    For synchronous functions, it uses `loop.run_in_executor` to prevent
    them from blocking the asyncio event loop. By default, it uses asyncio's
    internal `ThreadPoolExecutor`, but it allows for dependency injection of a
    custom executor (e.g., `ThreadPoolExecutor` or `ProcessPoolExecutor`)
    for more advanced use cases.

    Attributes:
        semaphore (asyncio.Semaphore): The semaphore instance controlling concurrency.
        executor (Optional[Executor]): The executor to run sync functions in.
            If None, the event loop's default executor is used.

    Examples:
        **1. Basic Usage (Default Executor)**

        In this example, we limit the execution of a blocking I/O-like
        function to 2 concurrent tasks. The class uses asyncio's default
        thread pool.

        >>> import asyncio
        >>> import time
        >>>
        >>> def blocking_io_task(duration: float, task_id: int) -> int:
        ...     print(f"Task {task_id}: Starting, will take {duration}s.")
        ...     time.sleep(duration)  # Simulate blocking I/O
        ...     print(f"Task {task_id}: Finished.")
        ...     return task_id * 10
        >>>
        >>> async def main():
        ...     runner = SemaphoreExecutor(max_concurrent=2)
        ...     tasks = [
        ...         runner.run(blocking_io_task, 1, task_id=1),
        ...         runner.run(blocking_io_task, 1, task_id=2),
        ...         runner.run(blocking_io_task, 1, task_id=3),
        ...         runner.run(blocking_io_task, 1, task_id=4),
        ...     ]
        ...     results = await asyncio.gather(*tasks)
        ...     print(f"Results: {results}")
        >>>
        >>> asyncio.run(main())

        **2. Advanced Usage (Injecting a Custom Executor)**

        Here, we create and manage our own `ThreadPoolExecutor`. This gives us
        explicit control over its lifecycle and configuration. The user is
        responsible for shutting down the executor. Using a `with` statement
        is the recommended approach.

        >>> import asyncio
        >>> from concurrent.futures import ThreadPoolExecutor
        >>>
        >>> # (blocking_io_task function is the same as above)
        >>>
        >>> async def main_with_custom_executor():
        ...     # The user creates and is responsible for the executor
        ...     with ThreadPoolExecutor(max_workers=3, thread_name_prefix="CustomPool") as pool:
        ...         runner = SemaphoreExecutor(max_concurrent=3, executor=pool)
        ...         tasks = [runner.run(blocking_io_task, 0.5, i) for i in range(5)]
        ...         await asyncio.gather(*tasks)
        >>>
        >>> asyncio.run(main_with_custom_executor())
    """

    def __init__(self, max_concurrent: int = 5, executor: Optional[Executor] = None):
        """Initializes the SemaphoreExecutor.

        Args:
            max_concurrent (int): The maximum number of functions that can be
                run concurrently.
            executor (Optional[Executor]): An optional custom executor
                (e.g., ThreadPoolExecutor or ProcessPoolExecutor) for running
                sync functions. If None, the event loop's default executor
                is used. The lifecycle of a provided executor is the
                responsibility of the user.
        """
        if max_concurrent <= 0:
            raise ValueError('max_concurrent must be a positive integer')
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.executor = executor

    async def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Runs a function under the control of the semaphore.

        Args:
            func (Callable): The function to execute. Can be a regular
                synchronous function or an async coroutine function.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value of the executed function.
        """
        async with self.semaphore:
            if asyncio.iscoroutinefunction(func):
                # If it's a coroutine function, await it directly.
                return await func(*args, **kwargs)
            else:
                # For a regular sync function, run it in the specified executor
                # to avoid blocking the event loop.
                loop = asyncio.get_running_loop()
                # functools.partial is used to pass args/kwargs to the function
                # when it's called by the executor.
                return await loop.run_in_executor(
                    self.executor, functools.partial(func, *args, **kwargs)
                )


class ResourcePool(Generic[T]):
    """A generic, async-safe pool for managing a limited number of resources."""

    def __init__(self, max_size: int, resource_factory: Callable[[], Any]):
        self.max_size = max_size
        self._factory = resource_factory
        self._resources: asyncio.Queue[T] = asyncio.Queue(max_size)
        self._created_resources = 0
        self._lock = asyncio.Lock()

    async def _create_resource(self) -> T:
        """Creates a new resource using the provided factory."""
        if asyncio.iscoroutinefunction(self._factory):
            resource = await self._factory()
        else:
            resource = self._factory()
        self._created_resources += 1
        return resource

    @asynccontextmanager
    async def get_resource(self) -> AsyncGenerator:
        """Acquires a resource from the pool, creating one if necessary."""
        if self._resources.empty() and self._created_resources < self.max_size:
            async with self._lock:
                if self._resources.empty() and self._created_resources < self.max_size:
                    resource = await self._create_resource()
                    await self._resources.put(resource)

        resource = await self._resources.get()
        try:
            yield resource
        finally:
            await self._resources.put(resource)


class QueueExecutor:
    """Runs functions asynchronously using a worker pool pattern with asyncio.Queue.

    This class provides task processing via a fixed number of worker coroutines
    that consume from a queue. Unlike SemaphoreExecutor which creates tasks
    upfront, QueueExecutor adds tasks to a queue and processes them through
    a persistent worker pool. This provides better backpressure handling and
    memory efficiency for large workloads.

    The executor must be started with `async with` or explicit `start()`/`stop()` calls.

    Attributes:
        num_workers (int): Number of concurrent worker coroutines.
        queue (asyncio.Queue): The task queue with optional max size for backpressure.
        executor (Optional[Executor]): Optional executor for sync functions.

    Examples:
        **1. Basic Usage with Context Manager**

        >>> import asyncio
        >>> import time
        >>>
        >>> def blocking_task(duration: float, task_id: int) -> int:
        ...     print(f"Task {task_id}: Starting, will take {duration}s.")
        ...     time.sleep(duration)
        ...     print(f"Task {task_id}: Finished.")
        ...     return task_id * 10
        >>>
        >>> async def main():
        ...     async with QueueExecutor(num_workers=2) as executor:
        ...         # Submit tasks - they go into queue
        ...         futures = [
        ...             executor.submit(blocking_task, 1, task_id=i)
        ...             for i in range(5)
        ...         ]
        ...         # Wait for all results
        ...         results = await asyncio.gather(*futures)
        ...         print(f"Results: {results}")
        >>>
        >>> asyncio.run(main())

        **2. With Backpressure (maxsize)**

        >>> async def main_with_backpressure():
        ...     # Queue size of 10 means submit() will block if queue is full
        ...     async with QueueExecutor(num_workers=2, maxsize=10) as executor:
        ...         futures = []
        ...         for i in range(100):
        ...             # This will block if queue is full (backpressure)
        ...             future = executor.submit(blocking_task, 0.1, task_id=i)
        ...             futures.append(future)
        ...
        ...         results = await asyncio.gather(*futures)
        ...         print(f"Processed {len(results)} tasks")
        >>>
        >>> asyncio.run(main_with_backpressure())

        **3. Manual Start/Stop**

        >>> async def main_manual():
        ...     executor = QueueExecutor(num_workers=3)
        ...     await executor.start()
        ...
        ...     try:
        ...         futures = [executor.submit(blocking_task, 0.5, i) for i in range(5)]
        ...         await asyncio.gather(*futures)
        ...     finally:
        ...         await executor.stop()
        >>>
        >>> asyncio.run(main_manual())

        **4. Async Functions**

        >>> async def async_task(duration: float, task_id: int) -> int:
        ...     print(f"Async task {task_id}: Starting")
        ...     await asyncio.sleep(duration)
        ...     print(f"Async task {task_id}: Done")
        ...     return task_id * 100
        >>>
        >>> async def main_async():
        ...     async with QueueExecutor(num_workers=3) as executor:
        ...         futures = [executor.submit(async_task, 0.5, i) for i in range(10)]
        ...         results = await asyncio.gather(*futures)
        ...         print(f"Results: {results}")
        >>>
        >>> asyncio.run(main_async())
    """

    def __init__(
        self,
        num_workers: int = 5,
        maxsize: int = 0,
        executor: Optional[Executor] = None,
    ):
        """Initializes the QueueExecutor.

        Args:
            num_workers (int): Number of worker coroutines to process tasks.
            maxsize (int): Maximum queue size. 0 means unlimited. When the queue
                is full, submit() will block until space is available (backpressure).
            executor (Optional[Executor]): Optional custom executor for sync functions.
                If None, uses the event loop's default executor.
        """
        if num_workers <= 0:
            raise ValueError('num_workers must be a positive integer')

        self.num_workers = num_workers
        self.maxsize = maxsize
        self.executor = executor
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._workers: List[asyncio.Task] = []
        self._running = False

    async def start(self):
        """Starts the worker pool. Must be called before submitting tasks."""
        if self._running:
            return

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(worker_id=i))
            for i in range(self.num_workers)
        ]

    async def stop(self, timeout: Optional[float] = None):
        """Stops the worker pool gracefully.

        Args:
            timeout (Optional[float]): Maximum time to wait for workers to finish.
                If None, waits indefinitely.
        """
        if not self._running:
            return

        self._running = False

        # Send sentinel values to stop workers
        for _ in range(self.num_workers):
            await self.queue.put(None)

        # Wait for workers to finish
        if timeout:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Cancel workers if timeout exceeded
                for worker in self._workers:
                    worker.cancel()
                await asyncio.gather(*self._workers, return_exceptions=True)
        else:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()

    async def _worker(self, worker_id: int):
        """Worker coroutine that processes tasks from the queue."""
        while self._running:
            try:
                item = await self.queue.get()

                # Sentinel value to stop worker
                if item is None:
                    self.queue.task_done()
                    break

                func, args, kwargs, future = item

                try:
                    # Execute the function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            self.executor, functools.partial(func, *args, **kwargs)
                        )

                    # Set the result on the future
                    if not future.cancelled():
                        future.set_result(result)

                except Exception as e:
                    # Set exception on the future
                    if not future.cancelled():
                        future.set_exception(e)

                finally:
                    self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log unexpected errors but keep worker running
                print(f'Worker {worker_id} encountered unexpected error: {e}')

    def submit(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> asyncio.Future:
        """Submits a function to be executed by the worker pool.

        Args:
            func (Callable): The function to execute (sync or async).
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            asyncio.Future: A future that will contain the result.

        Raises:
            RuntimeError: If the executor hasn't been started.
        """
        if not self._running:
            raise RuntimeError(
                'QueueExecutor must be started before submitting tasks. '
                "Use 'async with QueueExecutor(...)' or call 'await executor.start()'"
            )

        future = asyncio.Future()

        async def _submit():
            await self.queue.put((func, args, kwargs, future))
            return future

        return asyncio.create_task(
            _submit_and_return_future(self.queue, func, args, kwargs, future)
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False


async def _submit_and_return_future(
    queue: asyncio.Queue,
    func: Callable,
    args: Tuple,
    kwargs: dict,
    future: asyncio.Future,
) -> Any:
    """Helper to submit to queue and await the future."""
    await queue.put((func, args, kwargs, future))
    return await future


def run_async_function(
    async_func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
) -> T:
    """
    Runs an async function in both sync and async-safe contexts.
    If already inside an event loop, runs the coroutine in a separate thread.

    Args:
        async_func: Async function to execute
        *args: Positional args to pass
        **kwargs: Keyword args to pass

    Returns:
        The result of the async function.
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside a running event loop (e.g., Jupyter), run in a thread
        result_container = {}

        def thread_runner():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = new_loop.run_until_complete(async_func(*args, **kwargs))
            result_container['result'] = result
            new_loop.close()

        thread = threading.Thread(target=thread_runner)
        thread.start()
        thread.join()
        return result_container['result']
    else:
        return asyncio.run(async_func(*args, **kwargs))


async def gather_with_progress(
    tasks: List[asyncio.Task],
    description: str,
    show_progress: bool,
) -> List[Any]:
    """
    Executes a list of asyncio tasks with an optional TQDM progress bar.

    Args:
        tasks: A list of asyncio.Task objects to be executed.
        description: A string description to display on the progress bar.
        show_progress: A boolean indicating whether to show the progress bar.

    Returns:
        A list containing the results of the completed tasks.
    """
    if not tasks:
        return []

    if show_progress:
        try:
            from tqdm.asyncio import tqdm

            return await tqdm.gather(*tasks, desc=description)
        except ImportError:
            return await asyncio.gather(*tasks)
    else:
        return await asyncio.gather(*tasks)
