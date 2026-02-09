"""
Tests for LiteLLM integration in LLMHandler.

These tests verify that:
1. LiteLLM routes to the correct provider based on model name prefix
2. The handler correctly uses litellm.acompletion instead of openai
3. Cost estimation works with LiteLLM's built-in pricing
4. Exception mapping correctly wraps LiteLLM exceptions
"""

import json
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

            with patch(
                'litellm.completion_cost', return_value=expected_cost
            ) as mock_cost:
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
            with patch(
                'litellm.completion_cost', side_effect=Exception('Unknown model')
            ):
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

            assert 'Rate limit' in str(
                exc_info.value
            ) or 'retry attempts failed' in str(exc_info.value)

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

            assert 'Authentication' in str(
                exc_info.value
            ) or 'retry attempts failed' in str(exc_info.value)


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


class TestSystemInstructionIdempotency:
    """Ensure JSON rules suffix is not duplicated."""

    def test_system_instruction_appends_marker_once(self):
        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')

        handler = TestHandler()

        s1 = handler._system_instruction()
        s2 = handler._system_instruction()

        assert s1.count('[JSON_RULES]') == 1
        assert s2.count('[JSON_RULES]') == 1

    def test_system_instruction_does_not_duplicate_if_marker_present(self):
        class TestHandler(LLMHandler):
            instruction = 'Do the thing.\n\n[JSON_RULES]\nReturn JSON only.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')

        handler = TestHandler()
        s = handler._system_instruction()

        assert s.count('[JSON_RULES]') == 1


class TestParserModeOutputSanitization:
    """Ensure parser mode strips markdown code fences from outputs."""

    @pytest.mark.asyncio
    async def test_parser_mode_strips_json_code_fences_from_logged_response(self):
        class ParserMockLLM:
            def __init__(self, model: str = 'gpt-4o', temperature: float = 0.0):
                self.model = model
                self.temperature = temperature

            def complete(self, **kwargs):
                pass

            async def acomplete(self, *args, **kwargs):
                # handler calls acomplete(prompt) positionally in parser mode
                return MagicMock(
                    text=('```json\n{\n  "answer": "ok",\n  "confidence": 0.5\n}\n```')
                )

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = ParserMockLLM(model='gpt-4o')
            as_structured_llm = False

        handler = TestHandler()
        # Intercept logged response string for assertion
        handler.tracer.log_llm_call = MagicMock()

        result = await handler.execute({'query': 'hi'})
        assert isinstance(result, SimpleOutput)

        handler.tracer.log_llm_call.assert_called()
        logged_response = handler.tracer.log_llm_call.call_args.kwargs['response']
        assert logged_response.strip().startswith('{')
        assert logged_response.strip().endswith('}')
        assert '```' not in logged_response


class TestRateLimitRetryBehavior:
    """Test rate-limit specific retry improvements."""

    @pytest.mark.asyncio
    async def test_rate_limit_error_skips_parser_fallback(self):
        """Verify structured rate-limit error re-raises instead of falling back to parser."""
        import litellm.exceptions

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')
            fallback_to_parser = True
            max_retries = 1
            max_rate_limit_retries = 1

        handler = TestHandler()

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = litellm.exceptions.RateLimitError(
                message='Rate limit exceeded',
                llm_provider='openai',
                model='gpt-4o',
            )

            from axion._core.error import GenerationError

            with pytest.raises(GenerationError):
                await handler.execute({'query': 'Test query'})

            # The parser fallback should NOT have been called —
            # only the structured call should have been attempted.
            # With max_rate_limit_retries=1 we expect exactly 1 acompletion call.
            assert mock_acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_non_rate_limit_error_still_falls_back(self):
        """Regression test: format/parse errors still trigger parser fallback."""

        class ParserFallbackLLM:
            def __init__(self):
                self.model = 'gpt-4o'
                self.temperature = 0.0

            def complete(self, **kwargs):
                pass

            async def acomplete(self, *args, **kwargs):
                return MagicMock(
                    text='{"answer": "fallback answer", "confidence": 0.7}'
                )

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = ParserFallbackLLM()
            fallback_to_parser = True
            max_retries = 1

        handler = TestHandler()

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            # Simulate a non-rate-limit error (e.g., JSON parsing failure)
            mock_acompletion.side_effect = Exception('Invalid JSON in response')

            result = await handler.execute({'query': 'Test query'})

            # Parser fallback should have succeeded
            assert isinstance(result, SimpleOutput)
            assert result.answer == 'fallback answer'

    def test_retry_delay_includes_jitter(self):
        """Verify jitter is added to rate-limit retry delays."""

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')
            rate_limit_jitter = 5.0
            rate_limit_buffer = 2.0

        handler = TestHandler()

        # Create a rate-limit error with a known reset time
        error = Exception(
            'Rate limit exceeded: Limit 30000, Used 29500, Requested 1000. '
            'Please try again in 30s.'
        )

        delays = set()
        for _ in range(20):
            delay = handler._calculate_retry_delay(attempt=0, error=error)
            delays.add(delay)

        # With jitter, we should see variation in delays
        # The base wait time should be >= 30s (from the error message)
        # All delays should be >= the base backoff
        base_backoff = handler.retry_delay * (2**0)
        for d in delays:
            assert d >= base_backoff

        # With 20 samples and jitter up to 5.0, we should see some variation
        assert len(delays) > 1, 'Expected jitter to produce varying delays'

    @pytest.mark.asyncio
    async def test_litellm_num_retries_disabled(self):
        """Verify num_retries=0 is passed to litellm.acompletion."""

        class TestHandler(LLMHandler):
            instruction = 'Answer the question.'
            input_model = SimpleInput
            output_model = SimpleOutput
            llm = MockLLM(model='gpt-4o')

        handler = TestHandler()

        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = create_mock_response()

            with patch('litellm.completion_cost', return_value=0.001):
                await handler.execute({'query': 'Test query'})

            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs['num_retries'] == 0
