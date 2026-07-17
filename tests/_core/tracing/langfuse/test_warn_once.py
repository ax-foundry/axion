"""The missing-setup warnings fire once per process, then drop to DEBUG.

A tracer is constructed per metric instantiation; without the once-guard the
same 'Langfuse credentials not configured' warning repeats for every metric
in a run (pure noise for deliberately-untraced deterministic scoring).
"""

import logging
from unittest.mock import patch

import pytest

from axion._core.tracing.langfuse.tracer import LangfuseTracer


@pytest.fixture(autouse=True)
def _reset_warn_flags():
    LangfuseTracer._warned_not_installed = False
    LangfuseTracer._warned_no_credentials = False
    yield
    LangfuseTracer._warned_not_installed = False
    LangfuseTracer._warned_no_credentials = False


def _make_tracer_without_credentials() -> None:
    with patch.dict(
        'os.environ',
        {'LANGFUSE_PUBLIC_KEY': '', 'LANGFUSE_SECRET_KEY': ''},
        clear=False,
    ):
        LangfuseTracer(public_key=None, secret_key=None)


def test_missing_credentials_warns_once(caplog):
    with caplog.at_level(logging.DEBUG, logger='axion._core.tracing.langfuse.tracer'):
        _make_tracer_without_credentials()
        _make_tracer_without_credentials()
        _make_tracer_without_credentials()

    matching = [r for r in caplog.records if 'credentials not configured' in r.message]
    warnings = [r for r in matching if r.levelno == logging.WARNING]
    debugs = [r for r in matching if r.levelno == logging.DEBUG]
    assert len(warnings) == 1
    assert len(debugs) == 2


def test_not_installed_warns_once(caplog):
    with (
        patch('axion._core.tracing.langfuse.tracer.LANGFUSE_AVAILABLE', False),
        caplog.at_level(logging.DEBUG, logger='axion._core.tracing.langfuse.tracer'),
    ):
        _make_tracer_without_credentials()
        _make_tracer_without_credentials()

    matching = [r for r in caplog.records if 'not installed' in r.message]
    warnings = [r for r in matching if r.levelno == logging.WARNING]
    debugs = [r for r in matching if r.levelno == logging.DEBUG]
    assert len(warnings) == 1
    assert len(debugs) == 1
