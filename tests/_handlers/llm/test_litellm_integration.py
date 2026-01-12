"""
Tests for LiteLLM integration in LLMHandler.

These tests verify that:
1. LiteLLM routes to the correct provider based on model name prefix
2. The handler correctly uses litellm.acompletion instead of openai
3. Cost estimation works with LiteLLM's built-in pricing
4. Exception mapping correctly wraps LiteLLM exceptions
"""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from axion._handlers.llm.handler import LLMHandler


class SimpleInput(BaseModel):
    """Simple input model for testing."""

    query: str


class SimpleOutput(BaseModel):
    """Simple output model for testing."""

    answer: str
    confidence: float


class MockLLM:
    """Mock LLM that mimics the expected interface."""

    def __init__(self, model: str = 'gpt-4o', temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def complete(self, **kwargs):
        pass

    async def acomplete(self, **kwargs):
        pass


def create_mock_response(
    content: str = '{"answer": "test answer", "confidence": 0.95}',
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    response_cost: float = None,
):
    """Create a mock LiteLLM response object."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    # LiteLLM stores cost in _hidden_params
    mock_response._hidden_params = {'response_cost': response_cost}
    return mock_response


class TestLiteLLMModelRouting:
    """Test that model names are correctly passed to LiteLLM for routing."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'model_name,expected_provider',
        [
            ('gpt-4o', 'openai'),
            ('gpt-4o-mini', 'openai'),
            ('anthropic/claude-3-5-sonnet-20241022', 'anthropic'),
            ('anthropic/claude-3-opus-20240229', 'anthropic'),
            ('gemini/gemini-1.5-pro', 'gemini'),
            ('gemini/gemini-1.5-flash', 'gemini'),
            ('azure/gpt-4o', 'azure'),
            ('ollama/llama3', 'ollama'),
        ],
    )
    async def test_model_routing_via_prefix(self, model_name, expected_provider):
        """Verify LiteLLM receives correct model name for provider routing."""

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model=model_name)

        handler = TestHandler()

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = create_mock_response()

            # Also mock completion_cost to avoid errors
            with patch('litellm.completion_cost', return_value=0.001):
                await handler.execute({'query': 'What is 2+2?'})

            # Verify the model name was passed correctly
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs['model'] == model_name


class TestLiteLLMCostEstimation:
    """Test cost estimation with LiteLLM integration."""

    @pytest.mark.asyncio
    async def test_cost_estimation_uses_response_cost(self):
        """Verify handler uses response._hidden_params['response_cost'] as primary source."""

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')

        handler = TestHandler()
        expected_cost = 0.00123

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            # Create response with cost in _hidden_params
            mock_response = create_mock_response(response_cost=expected_cost)
            mock_acompletion.return_value = mock_response

            with patch('litellm.completion_cost') as mock_cost:
                await handler.execute({'query': 'Test query'})

                # Verify completion_cost was NOT called since _hidden_params had cost
                mock_cost.assert_not_called()

                # Verify cost estimate was set from _hidden_params
                assert handler.cost_estimate == expected_cost

    @pytest.mark.asyncio
    async def test_cost_estimation_fallback_to_completion_cost(self):
        """Verify handler falls back to completion_cost when _hidden_params is empty."""

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')

        handler = TestHandler()
        expected_cost = 0.00456

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            # Create response with no cost in _hidden_params (None/0)
            mock_response = create_mock_response(response_cost=0.0)
            mock_acompletion.return_value = mock_response

            with patch('litellm.completion_cost', return_value=expected_cost) as mock_cost:
                await handler.execute({'query': 'Test query'})

                # Verify completion_cost was called as fallback
                mock_cost.assert_called_once_with(completion_response=mock_response)

                # Verify cost estimate was set
                assert handler.cost_estimate == expected_cost

    @pytest.mark.asyncio
    async def test_cost_estimation_fallback_to_manual(self):
        """Verify handler falls back to LLMCostEstimator when all LiteLLM methods fail."""

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')

        handler = TestHandler()

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            # Create response with no cost in _hidden_params
            mock_acompletion.return_value = create_mock_response(
                prompt_tokens=1000, completion_tokens=500, response_cost=0.0
            )

            # Make completion_cost raise an exception
            with patch('litellm.completion_cost', side_effect=Exception('Unknown model')):
                with patch(
                    'axion._handlers.llm.handler.LLMCostEstimator.estimate',
                    return_value=0.0075,
                ) as mock_estimator:
                    await handler.execute({'query': 'Test query'})

                    # Verify fallback was used
                    mock_estimator.assert_called_once_with(
                        model_name='gpt-4o',
                        prompt_tokens=1000,
                        completion_tokens=500,
                    )


class TestLiteLLMExceptionMapping:
    """Test that LiteLLM exceptions are properly mapped to GenerationError."""

    @pytest.mark.asyncio
    async def test_rate_limit_error_mapping(self):
        """Verify RateLimitError is caught and wrapped."""
        import litellm.exceptions

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')
            max_retries = 1  # Reduce retries for faster test

        handler = TestHandler()

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = litellm.exceptions.RateLimitError(
                message='Rate limit exceeded',
                llm_provider='openai',
                model='gpt-4o',
            )

            from axion._core.error import GenerationError

            with pytest.raises(GenerationError) as exc_info:
                await handler.execute({'query': 'Test query'})

            assert 'Rate limit' in str(exc_info.value) or 'retry attempts failed' in str(
                exc_info.value
            )

    @pytest.mark.asyncio
    async def test_authentication_error_mapping(self):
        """Verify AuthenticationError is caught and wrapped."""
        import litellm.exceptions

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')
            max_retries = 1

        handler = TestHandler()

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = litellm.exceptions.AuthenticationError(
                message='Invalid API key',
                llm_provider='openai',
                model='gpt-4o',
            )

            from axion._core.error import GenerationError

            with pytest.raises(GenerationError) as exc_info:
                await handler.execute({'query': 'Test query'})

            assert 'Authentication' in str(exc_info.value) or 'retry attempts failed' in str(
                exc_info.value
            )


class TestLiteLLMStructuredOutput:
    """Test structured output functionality with LiteLLM."""

    @pytest.mark.asyncio
    async def test_response_format_passed_correctly(self):
        """Verify JSON schema response_format is passed to LiteLLM."""

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')

        handler = TestHandler()

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = create_mock_response()

            with patch('litellm.completion_cost', return_value=0.001):
                await handler.execute({'query': 'What is 2+2?'})

            # Verify response_format was passed
            call_kwargs = mock_acompletion.call_args.kwargs
            assert 'response_format' in call_kwargs
            assert call_kwargs['response_format']['type'] == 'json_schema'
            assert 'json_schema' in call_kwargs['response_format']

            # Verify schema matches output model
            schema = call_kwargs['response_format']['json_schema']
            assert schema['name'] == 'SimpleOutput'

    @pytest.mark.asyncio
    async def test_response_parsing(self):
        """Verify LiteLLM response is correctly parsed into output model."""

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')

        handler = TestHandler()

        expected_output = {'answer': 'The answer is 4', 'confidence': 0.99}

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = create_mock_response(
                content=json.dumps(expected_output)
            )

            with patch('litellm.completion_cost', return_value=0.001):
                result = await handler.execute({'query': 'What is 2+2?'})

            assert isinstance(result, SimpleOutput)
            assert result.answer == expected_output['answer']
            assert result.confidence == expected_output['confidence']


class TestProviderDetection:
    """Test provider detection from model name."""

    def test_provider_from_model_prefix(self):
        """Verify provider is correctly extracted from model name."""

        class TestHandler(LLMHandler):
            instruction = 'Test'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='anthropic/claude-3-5-sonnet-20241022')

        handler = TestHandler()
        metadata = handler.get_execution_metadata()

        assert metadata['provider'] == 'anthropic'
        assert metadata['model_name'] == 'anthropic/claude-3-5-sonnet-20241022'

    def test_provider_defaults_to_openai(self):
        """Verify provider defaults to openai for unprefixed models."""

        class TestHandler(LLMHandler):
            instruction = 'Test'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')

        handler = TestHandler()
        metadata = handler.get_execution_metadata()

        assert metadata['provider'] == 'openai'
        assert metadata['model_name'] == 'gpt-4o'
