"""Tests for the sync/async bridge in ``axion._core.asyncio``."""

from __future__ import annotations

import asyncio

import pytest

from axion._core.asyncio import run_async_function


@pytest.mark.asyncio
async def test_pending_background_task_is_cancelled_before_the_loop_closes():
    """A task the callee left running must be cancelled, not orphaned.

    Libraries start a background worker on first use and keep it for the life of
    the loop -- litellm's LoggingWorker is the one that surfaced this. When the
    bridge closed its loop with that task still pending, the task kept a
    reference to the dead loop and raised 'Event loop is closed' from inside an
    unrelated later step, on a run that otherwise succeeded.
    """
    observed: dict[str, bool] = {}

    async def worker(queue: asyncio.Queue) -> None:
        try:
            while True:
                await queue.get()
        except asyncio.CancelledError:
            observed['cancelled'] = True
            raise

    async def payload() -> str:
        queue: asyncio.Queue = asyncio.Queue()
        asyncio.create_task(worker(queue))
        # Let the worker reach its await, so it is genuinely pending.
        await asyncio.sleep(0)
        return 'done'

    assert run_async_function(payload) == 'done'
    assert observed.get('cancelled') is True


@pytest.mark.asyncio
async def test_failure_inside_the_thread_reaches_the_caller():
    """The callee's exception is re-raised here, not swallowed into a KeyError."""

    async def payload() -> None:
        raise ValueError('boom')

    with pytest.raises(ValueError, match='boom'):
        run_async_function(payload)


def test_runs_from_a_synchronous_caller():
    """With no running loop the coroutine runs directly on a fresh one."""

    async def payload(value: int) -> int:
        await asyncio.sleep(0)
        return value * 2

    assert run_async_function(payload, 21) == 42
