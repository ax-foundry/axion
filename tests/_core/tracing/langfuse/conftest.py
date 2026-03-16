import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock

_langfuse_stub = types.ModuleType('langfuse')


class _FakeLangfuse:
    def __init__(self, **kwargs):
        self._update_calls: List[Dict[str, Any]] = []

    def start_as_current_observation(self, **kwargs):
        obs = MagicMock()
        obs.__enter__ = MagicMock(return_value=obs)
        obs.__exit__ = MagicMock(return_value=False)
        obs.id = 'obs-id'
        obs.trace_id = 'trace-id'
        return obs

    def update_current_trace(self, **kwargs):
        self._update_calls.append(kwargs)

    def flush(self):
        pass

    def shutdown(self):
        pass


_langfuse_stub.Langfuse = _FakeLangfuse
sys.modules.setdefault('langfuse', _langfuse_stub)

# Import after stub is in place
from axion._core.tracing.langfuse.tracer import LangfuseTracer  # noqa: E402


def make_tracer(session_id=None, user_id=None, tags=None, **kwargs) -> LangfuseTracer:
    """Build a LangfuseTracer with a fake client."""
    tracer = LangfuseTracer.__new__(LangfuseTracer)
    tracer.metadata_type = 'default'
    tracer.tool_metadata = LangfuseTracer._create_default_tool_meta()
    tracer.auto_flush = False
    tracer.kwargs = {}
    tracer.logger = MagicMock()
    from axion._core.utils import Timer

    tracer.timer = Timer()
    tracer._current_span = None
    tracer._span_stack = []
    from axion._core.uuid import uuid7

    tracer._trace_id = str(uuid7())
    tracer._metadata = tracer._create_metadata()
    tracer.tags = tags or []
    tracer.environment = None
    tracer.session_id = LangfuseTracer._validate_session_id(session_id)
    tracer.user_id = LangfuseTracer._validate_user_id(user_id)
    tracer._client = _FakeLangfuse()
    return tracer
