import json
import re
from typing import Dict, List, Optional, Tuple, Union

import openai
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
)

# Fallback for when deepeval is not installed
try:
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError:
    DeepEvalBaseLLM = object
    import warnings

    warnings.warn(
        'deepeval is not installed. DeepEvalLLM will not be fully functional.'
    )

from axion._core.environment import settings
from axion._core.logging import get_logger
from axion._core.metadata.schema import ToolMetadata
from axion._core.networking import RateLimitInfo
from axion._core.schema import LLMRunnable
from axion._core.tracing import init_tracer, trace
from axion._core.tracing.handlers import BaseTraceHandler
from axion._handlers.llm.models import structured_outputs_models
from axion._handlers.utils import messages_to_prompt
from axion.llm_registry import LLMCostEstimator, LLMGatewayProvider

logger = get_logger(__name__)

RETRYABLE_OPENAI_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)


def calculate_retry_wait(retry_state: RetryCallState) -> float:
    """
    Calculate wait time with rate-limit awareness.

    This function replaces the standard exponential jitter backoff with
    intelligent rate-limit aware waiting:
    - For 429 errors: Extracts reset time and waits accordingly
    - For other errors: Uses exponential backoff with jitter

    Args:
        retry_state: The retry state from tenacity

    Returns:
        Number of seconds to wait before next retry
    """
    exception = retry_state.outcome.exception()

    # Try to extract rate limit info
    if exception:
        error_str = str(exception)

        # Check if this is a rate limit error
        if '429' in error_str or 'rate limit' in error_str.lower():
            rate_info = RateLimitInfo.from_error(error_str)

            if rate_info:
                wait_time = rate_info.get_wait_time(buffer_seconds=1.0)
                max_wait = 120.0  # Cap at 2 minutes
                actual_wait = min(wait_time, max_wait)

                logger.warning(
                    f'⏱️  Rate limit hit: {rate_info}. '
                    f'Waiting {actual_wait:.1f}s for reset (attempt {retry_state.attempt_number})'
                )

                return actual_wait

    # Fall back to exponential backoff with jitter for other errors
    # Formula: min(initial * (exp_base ** attempt) + random_jitter, max)
    attempt = retry_state.attempt_number - 1  # 0-indexed
    base_wait = 1.0 * (2**attempt)
    max_wait = 10.0

    actual_wait = min(base_wait, max_wait)

    logger.warning(
        f'⏱️  Retrying with exponential backoff: {actual_wait:.1f}s '
        f'(attempt {retry_state.attempt_number})'
    )

    return actual_wait


def trim_and_load_json(input_string: str) -> Dict:
    """Extract and parse JSON from a string, handling common formatting issues."""
    start = input_string.find('{')
    end = input_string.rfind('}') + 1

    if end == 0 and start != -1:
        input_string = input_string + '}'
        end = len(input_string)

    json_str = input_string[start:end] if start != -1 and end != 0 else ''
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        error_str = 'Evaluation LLM outputted an invalid JSON. Please use a better evaluation model.'
        raise ValueError(error_str)
    except Exception as e:
        raise Exception(f'An unexpected error occurred: {str(e)}')


def log_retry_error(retry_state: RetryCallState):
    """Log retry attempts for OpenAI API calls with rate-limit awareness."""
    exception = retry_state.outcome.exception()
    error_str = str(exception)

    # Check if this is a rate limit error and parse it
    if '429' in error_str or 'rate limit' in error_str.lower():
        rate_info = RateLimitInfo.from_error(error_str)
        if rate_info:
            logger.warning(
                f'⚠️  Rate limit error: {rate_info}. '
                f'Retrying attempt {retry_state.attempt_number}...'
            )
        else:
            logger.warning(
                f'⚠️  Rate limit error detected (could not parse details). '
                f'Retrying attempt {retry_state.attempt_number}...'
            )
    else:
        logger.warning(
            f'A retryable error occurred: {exception}. '
            f'Retrying attempt {retry_state.attempt_number}...'
        )


def is_retryable_error(retry_state: RetryCallState) -> bool:
    """
    Extracts the actual exception from tenacity's RetryCallState
    and then performs the checks.
    """
    exception = retry_state.outcome.exception()

    logger.debug(
        f'Received RetryCallState. The actual exception is of type: {type(exception)}'
    )
    logger.debug(f'Actual exception string value: {str(exception)}')

    if not isinstance(exception, Exception):
        logger.debug('No exception found in RetryCallState. Returning False.')
        return False

    retryable_status_codes = [429, 500, 502, 503, 504]

    if isinstance(
        exception,
        (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
    ):
        logger.debug('MATCH: Standard OpenAI error. Returning True (will retry).')
        return True

    if isinstance(exception, openai.APIError):
        status = getattr(exception, 'http_status', 0)
        detail_str = str(exception).lower()
        is_retryable_status = status in retryable_status_codes
        # Check for direct rate limit mentions or wrapped 429 errors
        is_wrapped_rate_limit = (
            '429' in detail_str
            or 'rate limit' in detail_str
            or 'llm_gatekeeper' in detail_str
            or ('500' in detail_str and '429' in detail_str)
        )

        if is_retryable_status or is_wrapped_rate_limit:
            logger.debug(
                f'MATCH: openai.APIError (status: {status}). Returning True (will retry).'
            )
            return True

    logger.warning(
        'NO MATCH: No retryable conditions met. Returning False (will NOT retry).'
    )
    return False


class DeepEvalLLM(DeepEvalBaseLLM):
    _cost_estimate: float

    def __init__(
        self,
        model_name: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        temperature: float = 0,
        tracer: Optional[BaseTraceHandler] = None,
        rate_limit_buffer: float = 1.0,
        max_retries: int = 5,
        *args,
        **kwargs,
    ):
        self.model_name = llm.model if llm else model_name or settings.llm_model_name
        self._openai_api_key = _openai_api_key
        self.base_url = base_url

        if temperature < 0:
            raise ValueError('Temperature must be >= 0.')
        self.temperature = temperature

        # Rate limit configuration
        self.rate_limit_buffer = rate_limit_buffer
        self.max_retries = max_retries

        self.args = args
        self.kwargs = kwargs
        self.llm = llm or self._create_model()
        super().__init__(self.model_name)

        tool_metadata = ToolMetadata(
            name=f'{self.__class__.__name__.lower()}',
            description=f'DeepEval LLMGateway Evaluation ({self.__class__.__name__}) Model',
            owner='AXION',
            version='1.0.0',
        )

        self.tracer = init_tracer('llm', tool_metadata, tracer)

    def _create_model(self) -> LLMRunnable:
        """Create and return the LLM model instance."""
        provider = LLMGatewayProvider(self._openai_api_key, self.base_url)
        return provider.create_llm(self.model_name, temperature=self.temperature)

    def get_model(self):
        """Legacy method for backward compatibility."""
        return self._create_model()

    def load_model(self):
        """Required by base class but not used."""
        return None

    def _serialize_response(self, response) -> str:
        """Serialize a response object to string format."""
        try:
            if hasattr(response, 'model_dump_json'):
                return response.model_dump_json()
            elif hasattr(response, 'model_dump'):
                return str(response.model_dump())
            else:
                return str(response)
        except Exception:
            return str(response)

    def _calculate_tokens_and_cost(
        self, messages_or_prompt, response_text: str
    ) -> None:
        """Calculate token counts and cost estimate."""
        if isinstance(messages_or_prompt, list):
            prompt_tokens = len(messages_to_prompt(messages_or_prompt).split())
        else:
            prompt_tokens = getattr(
                response_text, 'prompt_tokens', len(messages_or_prompt.split())
            )

        completion_tokens = (
            len(response_text.split())
            if isinstance(response_text, str)
            else getattr(response_text, 'completion_tokens', 0)
        )

        if not isinstance(prompt_tokens, int):
            prompt_tokens = 0
        if not isinstance(completion_tokens, int):
            completion_tokens = 0

        self._cost_estimate = LLMCostEstimator.estimate(
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @trace(name='build_prompt_messages', capture_args=True, capture_result=True)
    def build_prompt_messages(self, prompt: str) -> List[ChatMessage]:
        return [ChatMessage(role=MessageRole.SYSTEM, content=prompt)]

    @trace(name='generate', capture_args=True, capture_result=True)
    @retry(
        wait=calculate_retry_wait,
        retry=is_retryable_error,
        after=log_retry_error,
        stop=stop_after_attempt(5),
    )
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        """Generate synchronous response with optional schema validation."""
        if schema and self.model_name in structured_outputs_models:
            if self.model_name not in structured_outputs_models:
                raise AssertionError('Only structured models are supported')

            structured_llm = self.llm.as_structured_llm(output_cls=schema)
            messages = self.build_prompt_messages(prompt)
            response = structured_llm.chat(messages)
            response_text = self._serialize_response(response.raw)
            self._calculate_tokens_and_cost(messages, response_text)
            return response.raw
        else:
            response = self.llm.complete(prompt)
            try:
                prompt_tokens = getattr(response, 'prompt_tokens', len(prompt.split()))
                completion_tokens = getattr(
                    response, 'completion_tokens', len(response.text.split())
                )
            except Exception:
                prompt_tokens, completion_tokens = 0, 0

            self._cost_estimate = LLMCostEstimator.estimate(
                self.model_name, prompt_tokens, completion_tokens
            )

            if schema:
                json_output = trim_and_load_json(response.text)
                return schema.model_validate(json_output)
            else:
                return response

    @trace(name='generate', capture_args=True, capture_result=True)
    @retry(
        wait=calculate_retry_wait,
        retry=is_retryable_error,
        after=log_retry_error,
        stop=stop_after_attempt(5),
    )
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Generate asynchronous response with optional schema validation."""
        if schema and self.model_name in structured_outputs_models:
            if self.model_name not in structured_outputs_models:
                raise AssertionError('Only structured models are supported')

            structured_llm = self.llm.as_structured_llm(output_cls=schema)
            messages = self.build_prompt_messages(prompt)
            response = await structured_llm.achat(messages)
            response_text = self._serialize_response(response.raw)
            self._calculate_tokens_and_cost(messages, response_text)
            return response.raw
        else:
            response = await self.llm.acomplete(prompt)
            try:
                prompt_tokens = getattr(response, 'prompt_tokens', len(prompt.split()))
                completion_tokens = getattr(
                    response, 'completion_tokens', len(response.text.split())
                )
            except Exception:
                prompt_tokens, completion_tokens = 0, 0

            self._cost_estimate = LLMCostEstimator.estimate(
                self.model_name, prompt_tokens, completion_tokens
            )

            if schema:
                json_output = trim_and_load_json(response.text)
                return schema.model_validate(json_output)
            else:
                return response

    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name

    @property
    def cost_estimate(self) -> float:
        """
        Cost estimate per run. Cost is calculated based on prompt
        tokens and completion tokens on a model basis.
        """
        return self._cost_estimate
