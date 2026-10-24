"""The DeepEval and RAGAS adapters call LiteLLM directly, bypassing
LiteLLMWrapper, so they must apply the reasoning-model temperature rule
themselves."""

import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from axion.integrations.models import litellm_deepeval, litellm_ragas
from axion.llm_registry import resolve_temperature


def _stub_base_classes():
    """Supply the optional base classes when the extras are not installed.

    Without ``deepeval`` and ``ragas`` the adapters fall back to ``object`` as
    their base and cannot be constructed, so the request-building code under
    test would be unreachable in a default dev install.
    """
    added = []
    if importlib.util.find_spec('deepeval') is None:

        class DeepEvalBaseLLM:
            def __init__(self, model_name=None, *args, **kwargs):
                self.model_name = model_name

        for name in ('deepeval', 'deepeval.models'):
            sys.modules.setdefault(name, types.ModuleType(name))
            added.append(name)
        base = types.ModuleType('deepeval.models.base_model')
        base.DeepEvalBaseLLM = DeepEvalBaseLLM
        sys.modules['deepeval.models.base_model'] = base
        added.append('deepeval.models.base_model')
    if importlib.util.find_spec('ragas') is None:

        class BaseRagasLLM:
            def __init__(self, run_config=None, **kwargs):
                self.run_config = run_config

        sys.modules.setdefault('ragas', types.ModuleType('ragas'))
        llms = types.ModuleType('ragas.llms')
        llms.BaseRagasLLM = BaseRagasLLM
        sys.modules['ragas.llms'] = llms
        added += ['ragas', 'ragas.llms']
    return added


@pytest.fixture(scope='module', autouse=True)
def adapters():
    added = _stub_base_classes()
    importlib.reload(litellm_deepeval)
    importlib.reload(litellm_ragas)
    yield SimpleNamespace(
        deepeval=litellm_deepeval.LiteLLMDeepEval, ragas=litellm_ragas.LiteLLMRagas
    )
    for name in added:
        sys.modules.pop(name, None)
    importlib.reload(litellm_deepeval)
    importlib.reload(litellm_ragas)


def _response(text='ok'):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


@pytest.mark.parametrize(
    'model, requested, expected',
    [
        ('gpt-6-sol', None, 1.0),
        ('gpt-6-sol', 0.0, 1.0),
        ('openai/gpt-6-luna', 0.3, 1.0),
        ('o3-mini', None, 1.0),
        ('anthropic/claude-sonnet-4-20250514', None, 0.0),
        ('anthropic/claude-sonnet-4-20250514', 0.7, 0.7),
    ],
)
def test_resolve_temperature(model, requested, expected):
    assert resolve_temperature(model, requested) == expected


class TestDeepEvalAdapter:
    @pytest.fixture(autouse=True)
    def _cls(self, adapters):
        self.LLM = adapters.deepeval

    def test_default_model_sends_fixed_temperature(self):
        llm = self.LLM()
        with patch('litellm.completion', return_value=_response()) as call:
            llm.generate('hi')
        assert call.call_args.kwargs['temperature'] == 1.0

    @pytest.mark.asyncio
    async def test_async_path_matches(self):
        llm = self.LLM(model='gpt-6-luna', temperature=0.0)
        with patch(
            'litellm.acompletion', new=AsyncMock(return_value=_response())
        ) as call:
            await llm.a_generate('hi')
        assert call.call_args.kwargs['temperature'] == 1.0

    def test_other_models_keep_requested_temperature(self):
        llm = self.LLM(model='anthropic/claude-sonnet-4-20250514')
        with patch('litellm.completion', return_value=_response()) as call:
            llm.generate('hi')
        assert call.call_args.kwargs['temperature'] == 0.0


class TestRagasAdapter:
    @pytest.fixture(autouse=True)
    def _cls(self, adapters):
        self.LLM = adapters.ragas

    def test_per_call_default_is_overridden_for_reasoning_models(self):
        llm = self.LLM()
        with patch('litellm.completion', return_value=_response()) as call:
            llm.generate_text('hi')  # RAGAS's own default is 1e-8
        assert call.call_args.kwargs['temperature'] == 1.0

    @pytest.mark.asyncio
    async def test_async_per_call_default_is_overridden(self):
        llm = self.LLM(model='gpt-6-sol')
        with patch(
            'litellm.acompletion', new=AsyncMock(return_value=_response())
        ) as call:
            await llm.agenerate_text('hi', temperature=0.0)
        assert call.call_args.kwargs['temperature'] == 1.0

    def test_other_models_honor_per_call_temperature(self):
        llm = self.LLM(model='anthropic/claude-sonnet-4-20250514')
        with patch('litellm.completion', return_value=_response()) as call:
            llm.generate_text('hi', temperature=0.5)
        assert call.call_args.kwargs['temperature'] == 0.5
        with patch('litellm.completion', return_value=_response()) as call:
            llm.generate_text('hi', temperature=1e-8)
        assert call.call_args.kwargs['temperature'] == 1e-8
