import os
from typing import Any

import pytest
from axion._core.environment import settings
from axion.llm_registry import (
    BaseProvider,
    LLMCostEstimator,
    LLMRegistry,
)


class MockOpenAIClient:
    def __init__(self, **kwargs):
        pass


class MockLlamaIndexLLM:
    def __init__(self, **kwargs):
        pass

    def complete(self, **kwargs):
        pass

    def acomplete(self, **kwargs):
        pass


class MockLlamaIndexEmbedding:
    def __init__(self, **kwargs):
        pass

    def get_text_embedding(self, **kwargs):
        pass

    def aget_text_embedding(self, **kwargs):
        pass


@LLMRegistry.register('llm_gateway')
class MockLLMGatewayProvider(BaseProvider):
    def __init__(self, **kwargs):
        super().__init__(api_key=kwargs.get('api_key', 'mock-api-key'), **kwargs)

    def create_llm(self, model_name: str, **kwargs) -> Any:
        return MockLlamaIndexLLM(model_name=model_name, **kwargs)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        return MockLlamaIndexEmbedding(model_name=model_name, **kwargs)


@LLMRegistry.register('llama_index')
class MockLlamaIndexProvider(BaseProvider):
    def __init__(self, **kwargs):
        super().__init__(api_key=kwargs.get('api_key', 'mock-api-key'), **kwargs)

    def create_llm(self, model_name: str, **kwargs) -> Any:
        return MockLlamaIndexLLM(model_name=model_name, **kwargs)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        return MockLlamaIndexEmbedding(model_name=model_name, **kwargs)


@pytest.fixture(autouse=True)
def reset_settings_and_env(monkeypatch):
    """
    A fixture that automatically runs for every test. It resets
    Settings and environment variables to a clean state before each test
    and restores them afterward.
    """
    # Store original values
    original_llm_provider = settings.llm_provider
    original_model_name = settings.llm_model_name
    original_api_key = settings.openai_api_key
    original_base_url = settings.api_base_url

    # Set to a known default state for testing
    monkeypatch.setattr(settings, 'llm_model_name', 'gpt-4o')
    monkeypatch.setattr(settings, 'openai_api_key', 'test-key')
    monkeypatch.setattr(settings, 'api_base_url', None)

    # Clean up any test-specific environment variables
    monkeypatch.delenv('LLM_PROVIDER', raising=False)
    monkeypatch.delenv('LLM_MODEL_NAME', raising=False)
    monkeypatch.delenv('API_BASE_URL', raising=False)

    yield

    # Teardown: Restore original values
    settings.llm_provider = original_llm_provider
    settings.llm_model_name = original_model_name
    settings.openai_api_key = original_api_key
    settings.api_base_url = original_base_url


def test_llm_cost_estimator_known_model():
    """Tests cost estimation for a model with a defined price."""
    cost = LLMCostEstimator.estimate(
        model_name='gpt-4o', prompt_tokens=1000, completion_tokens=1000
    )
    # Expected: (1000 * 2.50/1e6) + (1000 * 10.00/1e6) = 0.0025 + 0.01 = 0.0125
    assert cost == pytest.approx(0.0125)


def test_llm_cost_estimator_unknown_model():
    """Tests cost estimation for a model that falls back to default pricing."""
    cost = LLMCostEstimator.estimate(
        model_name='unknown-model', prompt_tokens=1000, completion_tokens=1000
    )
    # Expected: Uses default pricing (gpt-4o): (1000 * 2.50/1e6) + (1000 * 10.00/1e6) = 0.0125
    assert cost == pytest.approx(0.0125)


def test_registry_cost_direct_argument_precedence():
    """
    Tests that arguments passed directly to estimate_cost override all other settings.
    """
    # Settings are set in fixture, pass api_key to registry
    registry = LLMRegistry(api_key='mock-key')
    cost = registry.estimate_cost(
        model_name='gpt-3.5-turbo',
        provider='llama_index',
        prompt_tokens=2000,
        completion_tokens=1000,
    )
    # Expected for gpt-3.5-turbo: (2000 * 0.50/1e6) + (1000 * 1.50/1e6) = 0.001 + 0.0015 = 0.0025
    assert cost == pytest.approx(0.0025)


def test_registry_cost_settings_precedence(monkeypatch):
    """
    Tests that Settings class attributes are used when no direct arguments are provided.
    """
    monkeypatch.setattr(settings, 'llm_provider', 'llm_gateway')
    monkeypatch.setattr(settings, 'llm_model_name', 'gpt-4')
    monkeypatch.setattr(settings, 'openai_api_key', 'mock-key')

    registry = LLMRegistry(api_key='mock-key')
    cost = registry.estimate_cost(prompt_tokens=100, completion_tokens=200)

    # Expected for gpt-4: (100 * 30.00/1e6) + (200 * 60.00/1e6) = 0.003 + 0.012 = 0.015
    assert cost == pytest.approx(0.015)


def test_registry_cost_environment_variable_precedence(monkeypatch):
    """
    Tests that environment variables are used when no direct args or Settings are set.
    """
    monkeypatch.setenv('LLM_PROVIDER', 'llama_index')
    monkeypatch.setenv('LLM_MODEL_NAME', 'o1')
    monkeypatch.setenv('OPENAI_API_KEY', 'mock-key')  # Set the env var

    # Re-evaluate Settings to pick up env vars
    monkeypatch.setattr(settings, 'llm_provider', os.environ.get('LLM_PROVIDER'))
    monkeypatch.setattr(settings, 'llm_model_name', os.environ.get('LLM_MODEL_NAME'))
    monkeypatch.setattr(settings, 'openai_api_key', os.environ.get('OPENAI_API_KEY'))

    registry = LLMRegistry()  # No explicit api_key, should use Settings
    cost = registry.estimate_cost(prompt_tokens=5000, completion_tokens=1000)

    # Expected for o1: (5000 * 15.00/1e6) + (1000 * 60.00/1e6) = 0.075 + 0.06 = 0.135
    assert cost == pytest.approx(0.135)


def test_locked_registry_cost_estimation():
    """
    Tests that a locked registry uses its own provider but respects the model argument.
    """
    # Lock the registry to 'llm_gateway'
    registry = LLMRegistry(provider='llm_gateway')

    # settings.llm_model_name is 'gpt-4o', but we override with a direct model arg
    cost = registry.estimate_cost(
        model_name='gpt-4o-mini', prompt_tokens=1000, completion_tokens=3000
    )

    # Expected for gpt-4o-mini: (1000 * 0.150/1e6) + (3000 * 0.600/1e6) = 0.00015 + 0.0018 = 0.00195
    assert cost == pytest.approx(0.00195)


def test_locked_registry_ignores_provider_argument():
    """
    Tests that a locked registry ignores the 'provider' argument in estimate_cost.
    """
    # Lock the registry to 'llm_gateway'
    registry = LLMRegistry(provider='llm_gateway')

    # This provider argument should be ignored
    cost = registry.estimate_cost(
        provider='llama_index',
        model_name='gpt-4',
        prompt_tokens=100,
        completion_tokens=100,
    )

    # Expected for gpt-4: (100 * 30.00/1e6) + (100 * 60.00/1e6) = 0.003 + 0.006 = 0.009
    assert cost == pytest.approx(0.009)


def test_get_provider_instance_not_registered():
    """
    Tests that a ValueError is raised if a non-existent provider is requested.
    """
    registry = LLMRegistry(api_key='mock-key')
    with pytest.raises(
        ValueError, match="Provider 'non_existent_provider' is not registered."
    ):
        registry.estimate_cost(
            provider='non_existent_provider', prompt_tokens=1, completion_tokens=1
        )
