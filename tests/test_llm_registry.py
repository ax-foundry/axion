import os
import warnings
from typing import Any
from unittest.mock import patch, MagicMock

import pytest
from axion._core.environment import settings
from axion.llm_registry import (
    BaseProvider,
    LLMCostEstimator,
    LLMRegistry,
    LiteLLMWrapper,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    VertexAIProvider,
    HuggingFaceProvider,
)


# =============================================================================
# Mock Classes for Testing
# =============================================================================

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


@LLMRegistry.register('mock_gateway')
class MockGatewayProvider(BaseProvider):
    """Mock provider for testing - uses default create_llm from base."""

    LITELLM_PREFIX = 'mock/'

    def __init__(self, **kwargs):
        super().__init__(api_key=kwargs.get('api_key', 'mock-api-key'), **kwargs)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        return MockLlamaIndexEmbedding(model_name=model_name, **kwargs)


@LLMRegistry.register('mock_no_embeddings')
class MockNoEmbeddingsProvider(BaseProvider):
    """Mock provider that doesn't support embeddings."""

    LITELLM_PREFIX = 'mock_no_emb/'
    SUPPORTS_EMBEDDINGS = False

    def __init__(self, **kwargs):
        super().__init__(api_key=kwargs.get('api_key', 'mock-api-key'), **kwargs)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        raise NotImplementedError('This provider does not support embeddings.')


# =============================================================================
# Fixtures
# =============================================================================

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


# =============================================================================
# Cost Estimation Tests
# =============================================================================

def test_llm_cost_estimator_emits_deprecation_warning():
    """Tests that LLMCostEstimator.estimate() emits a deprecation warning."""
    with patch('litellm.cost_per_token', return_value=(0.0025, 0.01)):
        with pytest.warns(DeprecationWarning, match='LLMCostEstimator.estimate\\(\\) is deprecated'):
            LLMCostEstimator.estimate(
                model_name='gpt-4o', prompt_tokens=1000, completion_tokens=1000
            )


def test_llm_cost_estimator_delegates_to_litellm():
    """Tests that cost estimation delegates to LiteLLM's cost_per_token."""
    with patch('litellm.cost_per_token', return_value=(0.0025, 0.01)) as mock_cost:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            cost = LLMCostEstimator.estimate(
                model_name='gpt-4o', prompt_tokens=1000, completion_tokens=1000
            )
        # litellm.cost_per_token returns tuple (prompt_cost, completion_cost)
        assert cost == pytest.approx(0.0125)
        mock_cost.assert_called_once_with(
            model='gpt-4o', prompt_tokens=1000, completion_tokens=1000
        )


def test_llm_cost_estimator_returns_zero_on_error():
    """Tests that cost estimation returns 0.0 when LiteLLM fails."""
    with patch('litellm.cost_per_token', side_effect=Exception('Unknown model')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            cost = LLMCostEstimator.estimate(
                model_name='unknown-model', prompt_tokens=1000, completion_tokens=1000
            )
        assert cost == 0.0


def test_llm_cost_estimator_strips_provider_prefix():
    """Tests that LiteLLM provider prefixes are stripped for pricing lookup."""
    with patch('litellm.cost_per_token', return_value=(0.003, 0.015)) as mock_cost:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            cost = LLMCostEstimator.estimate(
                model_name='anthropic/claude-3-5-sonnet-20241022',
                prompt_tokens=1000,
                completion_tokens=1000
            )
        # Should strip provider prefix before calling LiteLLM
        mock_cost.assert_called_once_with(
            model='claude-3-5-sonnet-20241022',
            prompt_tokens=1000,
            completion_tokens=1000
        )
        assert cost == pytest.approx(0.018)


# =============================================================================
# Provider Attribute Tests
# =============================================================================

def test_provider_litellm_prefix_attributes():
    """Tests that all providers have correct LITELLM_PREFIX values."""
    assert OpenAIProvider.LITELLM_PREFIX == ''
    assert AnthropicProvider.LITELLM_PREFIX == 'anthropic/'
    assert GeminiProvider.LITELLM_PREFIX == 'gemini/'
    assert VertexAIProvider.LITELLM_PREFIX == 'vertex_ai/'
    assert HuggingFaceProvider.LITELLM_PREFIX == ''


def test_provider_supports_embeddings_attributes():
    """Tests that all providers have correct SUPPORTS_EMBEDDINGS values."""
    assert OpenAIProvider.SUPPORTS_EMBEDDINGS is True
    assert AnthropicProvider.SUPPORTS_EMBEDDINGS is False
    assert GeminiProvider.SUPPORTS_EMBEDDINGS is True
    assert VertexAIProvider.SUPPORTS_EMBEDDINGS is True
    assert HuggingFaceProvider.SUPPORTS_EMBEDDINGS is True


def test_all_providers_inherit_from_base():
    """Tests that all providers inherit from BaseProvider."""
    providers = [
        OpenAIProvider,
        AnthropicProvider,
        GeminiProvider,
        VertexAIProvider,
        HuggingFaceProvider,
    ]
    for provider in providers:
        assert issubclass(provider, BaseProvider)


# =============================================================================
# Model Prefixing Tests
# =============================================================================

def test_openai_provider_no_prefix():
    """Tests that OpenAI provider doesn't add a prefix."""
    registry = LLMRegistry(provider='openai')
    llm = registry.get_llm('gpt-4o')
    assert llm.model == 'gpt-4o'
    assert llm._provider == 'openai'


def test_anthropic_provider_adds_prefix():
    """Tests that Anthropic provider adds 'anthropic/' prefix."""
    registry = LLMRegistry(provider='anthropic')
    llm = registry.get_llm('claude-3-5-sonnet-20241022')
    assert llm.model == 'anthropic/claude-3-5-sonnet-20241022'
    assert llm._provider == 'anthropic'


def test_anthropic_provider_no_double_prefix():
    """Tests that Anthropic provider doesn't double-prefix."""
    registry = LLMRegistry(provider='anthropic')
    llm = registry.get_llm('anthropic/claude-3-opus-20240229')
    assert llm.model == 'anthropic/claude-3-opus-20240229'


def test_gemini_provider_adds_prefix():
    """Tests that Gemini provider adds 'gemini/' prefix."""
    registry = LLMRegistry(provider='gemini')
    llm = registry.get_llm('gemini-1.5-pro')
    assert llm.model == 'gemini/gemini-1.5-pro'
    assert llm._provider == 'gemini'


def test_gemini_provider_no_double_prefix():
    """Tests that Gemini provider doesn't double-prefix."""
    registry = LLMRegistry(provider='gemini')
    llm = registry.get_llm('gemini/gemini-1.5-flash')
    assert llm.model == 'gemini/gemini-1.5-flash'


def test_vertex_ai_provider_adds_prefix():
    """Tests that Vertex AI provider adds 'vertex_ai/' prefix."""
    registry = LLMRegistry(provider='vertex_ai')
    llm = registry.get_llm('gemini-1.5-pro')
    assert llm.model == 'vertex_ai/gemini-1.5-pro'
    assert llm._provider == 'vertex_ai'


def test_vertex_ai_provider_no_double_prefix():
    """Tests that Vertex AI provider doesn't double-prefix."""
    registry = LLMRegistry(provider='vertex_ai')
    llm = registry.get_llm('vertex_ai/claude-3-5-sonnet')
    assert llm.model == 'vertex_ai/claude-3-5-sonnet'


# =============================================================================
# LiteLLMWrapper Tests
# =============================================================================

def test_litellm_wrapper_created_by_openai():
    """Tests that OpenAI provider returns LiteLLMWrapper."""
    registry = LLMRegistry(provider='openai')
    llm = registry.get_llm('gpt-4o')
    assert isinstance(llm, LiteLLMWrapper)


def test_litellm_wrapper_created_by_anthropic():
    """Tests that Anthropic provider returns LiteLLMWrapper."""
    registry = LLMRegistry(provider='anthropic')
    llm = registry.get_llm('claude-3-5-sonnet-20241022')
    assert isinstance(llm, LiteLLMWrapper)


def test_litellm_wrapper_created_by_gemini():
    """Tests that Gemini provider returns LiteLLMWrapper."""
    registry = LLMRegistry(provider='gemini')
    llm = registry.get_llm('gemini-1.5-pro')
    assert isinstance(llm, LiteLLMWrapper)


def test_litellm_wrapper_created_by_vertex_ai():
    """Tests that Vertex AI provider returns LiteLLMWrapper."""
    registry = LLMRegistry(provider='vertex_ai')
    llm = registry.get_llm('gemini-1.5-pro')
    assert isinstance(llm, LiteLLMWrapper)


def test_litellm_wrapper_temperature_default():
    """Tests that LiteLLMWrapper has default temperature of 0.0."""
    registry = LLMRegistry(provider='openai')
    llm = registry.get_llm('gpt-4o')
    assert llm.temperature == 0.0


def test_litellm_wrapper_temperature_custom():
    """Tests that custom temperature is passed through."""
    registry = LLMRegistry(provider='openai')
    llm = registry.get_llm('gpt-4o', temperature=0.7)
    assert llm.temperature == 0.7


# =============================================================================
# Embedding Tests
# =============================================================================

def test_anthropic_embeddings_raises_not_implemented():
    """Tests that Anthropic provider raises NotImplementedError for embeddings."""
    registry = LLMRegistry(provider='anthropic')
    with pytest.raises(NotImplementedError, match='Anthropic does not provide embedding models'):
        registry.get_embedding_model('text-embedding-ada-002')


def test_mock_no_embeddings_provider():
    """Tests that a provider with SUPPORTS_EMBEDDINGS=False raises NotImplementedError."""
    registry = LLMRegistry(provider='mock_no_embeddings')
    with pytest.raises(NotImplementedError, match='does not support embeddings'):
        registry.get_embedding_model('some-model')


# =============================================================================
# Registry Tests
# =============================================================================

def test_all_expected_providers_registered():
    """Tests that all expected providers are registered."""
    expected = ['openai', 'anthropic', 'gemini', 'vertex_ai', 'huggingface']
    for provider in expected:
        assert provider in LLMRegistry._registry


def test_registry_cost_direct_argument_precedence():
    """Tests that arguments passed directly to estimate_cost override all other settings."""
    with patch('litellm.cost_per_token', return_value=(0.001, 0.0015)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            registry = LLMRegistry(api_key='mock-key')
            cost = registry.estimate_cost(
                model_name='gpt-3.5-turbo',
                provider='mock_gateway',
                prompt_tokens=2000,
                completion_tokens=1000,
            )
            # LiteLLM mock returns (0.001, 0.0015) = 0.0025
            assert cost == pytest.approx(0.0025)


def test_registry_cost_settings_precedence(monkeypatch):
    """Tests that Settings class attributes are used when no direct arguments are provided."""
    monkeypatch.setattr(settings, 'llm_provider', 'mock_gateway')
    monkeypatch.setattr(settings, 'llm_model_name', 'gpt-4')
    monkeypatch.setattr(settings, 'openai_api_key', 'mock-key')

    with patch('litellm.cost_per_token', return_value=(0.003, 0.012)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            registry = LLMRegistry(api_key='mock-key')
            cost = registry.estimate_cost(prompt_tokens=100, completion_tokens=200)
            # LiteLLM mock returns (0.003, 0.012) = 0.015
            assert cost == pytest.approx(0.015)


def test_registry_cost_environment_variable_precedence(monkeypatch):
    """Tests that environment variables are used when no direct args or Settings are set."""
    monkeypatch.setenv('LLM_PROVIDER', 'mock_gateway')
    monkeypatch.setenv('LLM_MODEL_NAME', 'o1')
    monkeypatch.setenv('OPENAI_API_KEY', 'mock-key')

    # Re-evaluate Settings to pick up env vars
    monkeypatch.setattr(settings, 'llm_provider', os.environ.get('LLM_PROVIDER'))
    monkeypatch.setattr(settings, 'llm_model_name', os.environ.get('LLM_MODEL_NAME'))
    monkeypatch.setattr(settings, 'openai_api_key', os.environ.get('OPENAI_API_KEY'))

    with patch('litellm.cost_per_token', return_value=(0.075, 0.06)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            registry = LLMRegistry()
            cost = registry.estimate_cost(prompt_tokens=5000, completion_tokens=1000)
            # LiteLLM mock returns (0.075, 0.06) = 0.135
            assert cost == pytest.approx(0.135)


def test_locked_registry_cost_estimation():
    """Tests that a locked registry uses its own provider but respects the model argument."""
    with patch('litellm.cost_per_token', return_value=(0.00015, 0.0018)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            registry = LLMRegistry(provider='mock_gateway')
            cost = registry.estimate_cost(
                model_name='gpt-4o-mini', prompt_tokens=1000, completion_tokens=3000
            )
            # LiteLLM mock returns (0.00015, 0.0018) = 0.00195
            assert cost == pytest.approx(0.00195)


def test_locked_registry_ignores_provider_argument():
    """Tests that a locked registry ignores the 'provider' argument in estimate_cost."""
    with patch('litellm.cost_per_token', return_value=(0.003, 0.006)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            registry = LLMRegistry(provider='mock_gateway')
            # This provider argument should be ignored
            cost = registry.estimate_cost(
                provider='mock_no_embeddings',
                model_name='gpt-4',
                prompt_tokens=100,
                completion_tokens=100,
            )
            # LiteLLM mock returns (0.003, 0.006) = 0.009
            assert cost == pytest.approx(0.009)


def test_get_provider_instance_not_registered():
    """Tests that a ValueError is raised if a non-existent provider is requested."""
    registry = LLMRegistry(api_key='mock-key')
    with pytest.raises(
        ValueError, match="Provider 'non_existent_provider' is not registered."
    ):
        registry.estimate_cost(
            provider='non_existent_provider', prompt_tokens=1, completion_tokens=1
        )


def test_provider_switching_via_flexible_registry():
    """Tests that a flexible registry can switch between providers."""
    registry = LLMRegistry()

    # Switch to different providers
    llm_openai = registry.get_llm('gpt-4o', provider='openai')
    llm_anthropic = registry.get_llm('claude-3-5-sonnet-20241022', provider='anthropic')
    llm_gemini = registry.get_llm('gemini-1.5-pro', provider='gemini')

    assert llm_openai.model == 'gpt-4o'
    assert llm_anthropic.model == 'anthropic/claude-3-5-sonnet-20241022'
    assert llm_gemini.model == 'gemini/gemini-1.5-pro'


# =============================================================================
# Base Provider Tests
# =============================================================================

def test_base_provider_has_create_litellm_wrapper():
    """Tests that BaseProvider has _create_litellm_wrapper method."""
    assert hasattr(BaseProvider, '_create_litellm_wrapper')


def test_base_provider_create_llm_not_abstract():
    """Tests that create_llm is no longer abstract in BaseProvider."""
    # If create_llm were abstract, we couldn't instantiate a subclass without overriding it
    # MockGatewayProvider doesn't override create_llm and should work fine
    provider = MockGatewayProvider()
    llm = provider.create_llm('test-model')
    assert isinstance(llm, LiteLLMWrapper)
    assert llm.model == 'mock/test-model'  # Should have prefix applied
