import asyncio
import json
import random
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

import litellm
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel

from axion._core.environment import settings
from axion._core.error import GenerationError
from axion._core.logging import get_logger
from axion._core.networking import RateLimitInfo
from axion._core.schema import (
    Callbacks,
    InputModel,
    LLMRunnable,
    OutputModel,
    PromptValue,
)
from axion._core.utils import Timer, base_model_dump_json, log_execution_time
from axion._handlers.base.handler import BaseHandler
from axion._handlers.llm.schema import PromptSection
from axion._handlers.utils import generate_fake_instance, messages_to_prompt
from axion._handlers.validation import Validation
from axion.llm_registry import LLMCostEstimator

logger = get_logger(__name__)


class LLMHandler(BaseHandler, Generic[InputModel, OutputModel]):
    """
    A handler for generating structured outputs using LLMs with Pydantic models.

    This class provides a complete framework for building prompts, executing LLM calls,
    parsing responses, and handling errors with an automatic retry mechanism. It intelligently
    switches between a modern, direct-to-Pydantic generation method (like LlamaIndex's
    `as_structured_llm`), raw open ai, and a string-parsing fallback for broader compatibility.

    Key features:

    - **Dual-Mode Execution**: Automatically uses `as_structured_llm` if available, otherwise
      uses parser-based string parsing, ensuring maximum compatibility.
    - **Strongly Typed Inputs and Outputs**: Leverages Pydantic models for both input
      validation and output parsing, ensuring type safety.
    - **Automatic Prompt Construction**: Builds prompts from model schemas and examples.
    - **Rate-Limit Aware Retry Logic**: Intelligently detects 429 errors and waits for
      the appropriate reset time before retrying.
    - **Comprehensive Error Handling**: Includes retry logic with exponential backoff.
    - **Execution Metadata Tracking**: Captures detailed traces, token counts, and cost estimates.
    - **Asynchronous Support**: Built for high-performance, non-blocking operations.
    - **Lifecycle Callbacks**: Provides hooks for monitoring generation events.

    The handler builds prompts by combining several sections:

    - Instructions that guide the LLM's response
    - Output signature specifications based on the Pydantic output model
    - Example input/output pairs to demonstrate the expected behavior
    - The current input data formatted according to the input model

    Implementation steps:

    1. Define input and output Pydantic models
    2. Specify LLM instructions
    3. Provide example input/output pairs (optional but recommended)
    4. Configure an LLM instance
    5. Call execute() with validated input data

    This design separates prompt engineering from application logic while
    maintaining strong typing and validation throughout the generation process.

    Attributes:
        instruction: The main instruction to be included in the prompt.
        examples: List of example input-output pairs to demonstrate the task.
        output_model: The Pydantic model defining the expected output structure.
        input_model: The Pydantic model defining the expected input structure.
        generation_fake_sample: If True, generates a fake example when no examples are provided.
        parser: Optional parser for processing the output.
        llm: Language model instance to be used.
        rate_limit_buffer: Buffer time (in seconds) to add when waiting for rate limit resets.
        max_rate_limit_retries: Maximum retry attempts for rate limit errors.
    """

    # Core configuration
    instruction: str = None
    examples: List[Tuple[InputModel, OutputModel]] = []
    generation_fake_sample: bool = False
    as_structured_llm: bool = True
    fallback_to_parser: bool = True
    _cost_estimate: float = None

    # LLM configuration
    handler_type: str = 'llm'
    parser: Any = None
    llm: LLMRunnable = None

    # API configuration
    api_base_url: Optional[str] = None

    # Rate limit configuration
    # Buffer time to add to rate limit reset (seconds)
    rate_limit_buffer: float = 2.0
    max_rate_limit_retries: int = 10
    max_delay = 300.0  # Maximum delay of 5 minutes
    rate_limit_jitter: float = (
        5.0  # Max random jitter (seconds) to stagger concurrent retries
    )

    # Prompt caching configuration
    enable_prompt_caching: bool = False

    def __init__(self, **kwargs):
        """Initialize LLM handler."""
        super().__init__(**kwargs)
        self._validate_llm_requirements()
        self._initialize_client()

    def _initialize_client(self):
        """Configure LiteLLM settings for API calls."""
        # LiteLLM reads API keys from env vars automatically:
        # OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
        # Configure custom base URL if needed (for proxies/gateways)
        if self.api_base_url or settings.api_base_url:
            litellm.api_base = self.api_base_url or settings.api_base_url

        # Set API key if available from settings (fallback when env vars not set)
        if settings.openai_api_key:
            litellm.api_key = settings.openai_api_key

    def _validate_llm_requirements(self):
        """Validate LLM-specific requirements."""
        required_fields = ['instruction', 'llm', 'input_model', 'output_model']
        self._validate_required_fields(required_fields)
        Validation.validate_llm_model(self.llm)
        Validation.validate_io_models(
            getattr(self, 'input_model', None), getattr(self, 'output_model', None)
        )

    def async_span(self, operation_name: str, **attributes):
        """
        Override async_span to use class name for top-level 'litellm_structured' traces.

        When 'litellm_structured' is the top-level trace (new trace), replace it with
        the handler's class name (e.g., 'DetailedCriteriaGenerator').

        Args:
            operation_name: Name of the operation being tracked.
            **attributes: Additional attributes to attach to the span.

        Returns:
            Async span context manager.
        """
        # NOTE: Most executions are now wrapped in a root span by BaseHandler, so
        # nested LLM spans should retain their real operation names. We only rename
        # if this handler is used standalone and there is no active span context.
        if operation_name == 'litellm_structured' and hasattr(
            self.tracer, '_span_stack'
        ):
            try:
                if len(self.tracer._span_stack) == 0:
                    operation_name = self.name
            except Exception:
                pass

        return super().async_span(operation_name, **attributes)

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """
        Check if the error is a rate limit (429) error.

        Handles both direct 429 errors and wrapped errors like:
        - Error code: 500 - {'detail': '500: 429: {...}'}

        Args:
            error: The exception to check

        Returns:
            True if this is a rate limit error, False otherwise
        """
        error_str = str(error).lower()
        # Check for direct 429 or rate limit mentions
        if '429' in error_str or 'rate limit' in error_str:
            return True
        # Check for wrapped 500 errors containing 429
        if '500' in error_str and (
            '429' in error_str
            or 'rate_limit' in error_str
            or 'llm_gatekeeper' in error_str
        ):
            return True
        return False

    def _calculate_retry_delay(
        self, attempt: int, error: Optional[Exception] = None
    ) -> float:
        """
        Calculate the delay before the next retry attempt.

        This method implements intelligent rate-limit aware backoff:
        - For rate limit errors (429), extracts the reset time and waits accordingly
        - For other errors, uses exponential backoff

        Args:
            attempt: Current attempt number (0-indexed)
            error: The exception that triggered the retry

        Returns:
            Number of seconds to wait before retry
        """

        # Check if this is a rate limit error
        if error and self._is_rate_limit_error(error):
            rate_limit_info = RateLimitInfo.from_error(str(error))

            if rate_limit_info:
                # Use the rate limit reset time
                wait_time = rate_limit_info.get_wait_time(self.rate_limit_buffer)
                backoff_time = self.retry_delay * (2**attempt)
                wait_time = max(wait_time, backoff_time)
                # Add jitter to stagger concurrent retries (thundering herd prevention)
                jitter = random.uniform(0, self.rate_limit_jitter)
                wait_time += jitter
                logger.warning_highlight(
                    f'⏱️  Rate limit hit: {rate_limit_info}. '
                    f'Waiting {wait_time:.1f}s for reset (attempt {attempt + 1}/{self.max_rate_limit_retries})'
                )
                return min(wait_time, self.max_delay)

        # Fall back to exponential backoff for other errors
        delay = self.retry_delay * (2**attempt)
        delay = min(delay, self.max_delay)

        logger.warning_highlight(
            f'⏱️  Retrying with exponential backoff: {delay:.1f}s '
            f'(attempt {attempt + 1}/{self.max_retries})'
        )

        return delay

    def _system_instruction(self) -> str:
        """
        Return a model-agnostic system instruction that is compatible with JSON-only
        user inputs (optionally prefixed with a small 'Input:' anchor) and JSON-only
        assistant outputs.
        """
        instruction = (self.instruction or '').strip()

        # Keep this short and provider-agnostic; rely on response_format/schema where supported.
        suffix = (
            '\n\n'
            '[JSON_RULES]\n'
            'You will receive JSON inputs as the user message content.\n'
            'Return ONLY a JSON object matching the required output schema.\n'
            'Do not include any additional keys, commentary, or markdown formatting.\n'
            'Examples may be synthetic; treat them only as demonstrations of structure.'
        )

        if not instruction:
            return suffix.strip()

        # Idempotency: if our rules (or equivalent) are already present, don't append.
        if '[json_rules]' in instruction.lower():
            return instruction

        return f'{instruction}{suffix}'

    @staticmethod
    def _input_anchor(json_str: str) -> str:
        """
        Add a tiny 'execute' anchor that helps weaker models interpret the JSON
        as the input to process, without adding verbose instruction text.
        """
        return f'Input:\n{json_str}'

    @staticmethod
    def _strip_markdown_code_fences(text: Optional[str]) -> Optional[str]:
        """
        Strip surrounding markdown code fences from model output.

        Common model behavior in parser mode is returning:
          ```json
          { ... }
          ```

        We remove the outer fence (``` or ~~~) and any language tag, returning the
        inner content. If no outer fence is present, return a trimmed string.
        """
        if text is None:
            return None

        s = text.strip()
        for fence in ('```', '~~~'):
            if s.startswith(fence):
                first_nl = s.find('\n')
                if first_nl == -1:
                    return ''
                inner = s[first_nl + 1 :]
                end = inner.rfind(fence)
                if end != -1:
                    inner = inner[:end]
                return inner.strip()
        return s

    def _build_message_dicts(
        self, data: Optional[InputModel] = None
    ) -> List[Dict[str, str]]:
        """Build messages as role/content dicts for LLM API calls."""
        messages = [{'role': 'system', 'content': self._system_instruction()}]
        # Add examples as conversation pairs
        for idx, (input_example, output_example) in enumerate[
            Tuple[InputModel, OutputModel]
        ](self.examples):
            messages.append(
                {
                    'role': 'user',
                    'content': self._input_anchor(base_model_dump_json(input_example)),
                }
            )
            messages.append(
                {'role': 'assistant', 'content': base_model_dump_json(output_example)}
            )

        # Add fake example if needed
        if not self.examples and self.generation_fake_sample:
            fake_input = generate_fake_instance(self.input_model)
            fake_output = generate_fake_instance(self.output_model)
            messages.append(
                {
                    'role': 'user',
                    'content': self._input_anchor(base_model_dump_json(fake_input)),
                }
            )
            messages.append(
                {'role': 'assistant', 'content': base_model_dump_json(fake_output)}
            )

        # Add actual input
        if data:
            content = self._input_anchor(base_model_dump_json(data))
        else:
            # Use a JSON-shaped placeholder to keep message format consistent for debugging.
            content = self._input_anchor(
                base_model_dump_json(self.input_model.model_construct())
            )

        messages.append({'role': 'user', 'content': content})

        return messages

    def _get_cache_control_injection_points(
        self, messages: List[Dict[str, str]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Return LiteLLM cache_control injection points when prompt caching is enabled.

        This enables provider-level prompt caching (Anthropic explicit, OpenAI automatic)
        by marking the system message and the last assistant message (few-shot boundary)
        as cache breakpoints.
        """
        if not self.enable_prompt_caching:
            return None

        points: List[Dict[str, Any]] = [{'location': 'message', 'role': 'system'}]

        # If there are assistant messages (from examples), mark the last one
        # to cache the entire prefix through all few-shot examples
        has_assistant = any(m['role'] == 'assistant' for m in messages)
        if has_assistant:
            points.append({'location': 'message', 'role': 'assistant', 'index': -1})

        return points

    async def _execute_structured_call(
        self,
        messages: List[Dict[str, str]],
        attempt: int = 1,
        input_data: Optional[InputModel] = None,
    ) -> OutputModel:
        """Execute structured output call using LiteLLM (supports OpenAI, Claude, Gemini, etc.)."""
        model_name = getattr(self.llm, 'model', 'openai/gpt-4o')
        # Determine provider from model name prefix (e.g., "anthropic/claude-3" -> "anthropic")
        if '/' in model_name:
            provider = model_name.split('/')[0]
        else:
            provider = getattr(getattr(self.llm, 'metadata', {}), 'provider', 'openai')

        async with self.async_span(
            'litellm_structured',
            model=model_name,
            mode='litellm_structured',
            provider=provider,
            attempt=attempt,
        ) as span:
            input_dump = None
            if input_data is not None:
                try:
                    # Store as a dict for easy downstream consumption.
                    input_dump = json.loads(base_model_dump_json(input_data))
                except Exception:
                    # Fall back to the string form if parsing fails for any reason.
                    input_dump = base_model_dump_json(input_data)

            span.set_input(input_dump)  # User input only
            try:
                schema = self.output_model.model_json_schema(mode='serialization')
                response_format = {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': self.output_model.__name__,
                        'schema': schema,
                        'strict': True,
                    },
                }

                with Timer() as timer:
                    # Make the LiteLLM API call - routes to correct provider based on model name
                    # Extract API key from LLM wrapper to ensure provider-agnostic auth
                    api_key = getattr(self.llm, '_api_key', None)
                    call_kwargs = {
                        'model': model_name,
                        'messages': messages,
                        'response_format': response_format,
                        'temperature': getattr(self.llm, 'temperature', 0.0),
                        'api_key': api_key,
                        'num_retries': 0,  # Disable LiteLLM's internal retry; Axion handles retries
                    }

                    cache_points = self._get_cache_control_injection_points(messages)
                    if cache_points is not None:
                        call_kwargs['cache_control_injection_points'] = cache_points

                    response = await litellm.acompletion(**call_kwargs)

                # Extract the JSON response
                response_text = response.choices[0].message.content

                # Parse and validate with Pydantic
                parsed_output = self.output_model.model_validate_json(response_text)

                # Track token usage and cost
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                # Estimate cost using LiteLLM's built-in pricing (supports 100+ models)
                # Priority: response_cost > completion_cost > LLMCostEstimator
                try:
                    # Primary: Use response_cost from LiteLLM (real-time pricing)
                    self.cost_estimate = response._hidden_params.get(
                        'response_cost', 0.0
                    )
                    if not self.cost_estimate:
                        # Fallback: Use completion_cost function
                        self.cost_estimate = litellm.completion_cost(
                            completion_response=response
                        )
                except Exception:
                    # Final fallback: Manual estimation for unsupported/custom models
                    self.cost_estimate = LLMCostEstimator.estimate(
                        model_name=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )

                span.set_attribute('prompt_tokens', prompt_tokens)
                span.set_attribute('completion_tokens', completion_tokens)
                span.set_attribute('total_tokens', prompt_tokens + completion_tokens)
                span.set_attribute('latency', timer.elapsed_time)
                span.set_attribute('cost_estimate', self.cost_estimate)

                # Log cached tokens for prompt caching observability
                cached_tokens = getattr(
                    getattr(usage, 'prompt_tokens_details', None),
                    'cached_tokens',
                    None,
                )
                if cached_tokens is not None:
                    span.set_attribute('cached_tokens', cached_tokens)

                try:
                    # Log the call to tracer
                    self.tracer.log_llm_call(
                        name='litellm_call',
                        model=model_name,
                        prompt=json.dumps(messages),  # Full messages for observability
                        response=response_text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency=timer.elapsed_time,
                        cost_estimate=self.cost_estimate,
                        provider=provider,
                    )
                except:
                    pass

                span.set_output(parsed_output)  # Capture LLM output
                return parsed_output

            except litellm.exceptions.RateLimitError as e:
                # Map LiteLLM rate limit errors to our error handling
                span.set_attribute('error', str(e))
                span.set_attribute('error_type', 'RateLimitError')
                raise GenerationError(f'Rate limit exceeded: {str(e)}') from e
            except litellm.exceptions.APIConnectionError as e:
                span.set_attribute('error', str(e))
                span.set_attribute('error_type', 'APIConnectionError')
                raise GenerationError(f'API connection failed: {str(e)}') from e
            except litellm.exceptions.AuthenticationError as e:
                span.set_attribute('error', str(e))
                span.set_attribute('error_type', 'AuthenticationError')
                raise GenerationError(f'Authentication failed: {str(e)}') from e
            except Exception as e:
                span.set_attribute('error', str(e))
                span.set_attribute('error_type', type(e).__name__)
                error_msg = f'LiteLLM structured call failed: {str(e)}'
                raise GenerationError(error_msg) from e

    def _build_chat_messages(
        self, data: Optional[InputModel] = None
    ) -> List[ChatMessage]:
        """Build messages for LlamaIndex chat interface."""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self._system_instruction())
        ]

        # Add examples as conversation pairs
        for input_example, output_example in self.examples:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=self._input_anchor(base_model_dump_json(input_example)),
                )
            )
            messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=base_model_dump_json(output_example),
                )
            )

        # Add fake example if needed
        if not self.examples and self.generation_fake_sample:
            fake_input = generate_fake_instance(self.input_model)
            fake_output = generate_fake_instance(self.output_model)
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=self._input_anchor(base_model_dump_json(fake_input)),
                )
            )
            messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=base_model_dump_json(fake_output),
                )
            )

        # Add actual input
        if data:
            content = self._input_anchor(base_model_dump_json(data))
        else:
            content = self._input_anchor(
                base_model_dump_json(self.input_model.model_construct())
            )

        messages.append(ChatMessage(role=MessageRole.USER, content=content))

        return messages

    async def _execute_structured_llm_call(
        self, messages: List[ChatMessage]
    ) -> OutputModel:
        """Execute LLM call using as_structured_llm (modern LlamaIndex approach)."""
        model_name = getattr(self.llm, 'model', 'unknown')

        # Define provider before try block so it's available in exception handler
        provider = getattr(getattr(self.llm, 'metadata', {}), 'provider', 'unknown')

        async with self.async_span(
            'structured_llm_execution', model=model_name, mode='as_structured_llm'
        ) as span:
            span.set_input({'messages': messages, 'model': model_name})
            try:
                with Timer() as timer:
                    structured_llm = self.llm.as_structured_llm(self.output_model)
                    response = await structured_llm.achat(messages)

                parsed_output = response.raw

                # Estimate cost
                if hasattr(response, 'raw') and hasattr(response.raw, 'usage'):
                    usage = response.raw.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
                    self.cost_estimate = LLMCostEstimator.estimate(
                        model_name=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                else:
                    prompt_tokens = 0
                    completion_tokens = 0
                    self.cost_estimate = 0.0

                span.set_attribute('prompt_tokens', prompt_tokens)
                span.set_attribute('completion_tokens', completion_tokens)
                span.set_attribute('latency', timer.elapsed_time)
                span.set_attribute('cost_estimate', self.cost_estimate)

                # Log the call to tracer
                prompt_str = messages_to_prompt(messages)
                self.tracer.log_llm_call(
                    name='structured_llm_call',
                    model=model_name,
                    prompt=prompt_str,
                    response=str(response),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency=timer.elapsed_time,
                    cost_estimate=self.cost_estimate,
                    provider=provider,
                )

                span.set_output(parsed_output)  # Capture LLM output
                return parsed_output

            except Exception as e:
                span.set_attribute('error', str(e))
                span.set_attribute('error_type', type(e).__name__)
                raise GenerationError(f'Structured LLM call failed: {str(e)}') from e

    def _build_parser_prompt(self, data: InputModel) -> str:
        """Build prompt string for parser-based approach."""
        sections = []

        # Add instruction
        sections.append(
            PromptSection(name='instruction', content=self.instruction, priority=1)
        )

        # Add output signature (schema for expected output)
        output_signature = self.output_model.model_json_schema()
        sections.append(
            PromptSection(
                name='output_signature',
                content=f'The output must conform to the following schema:\n{json.dumps(output_signature, indent=2)}',
                priority=2,
            )
        )

        # Add examples
        if self.examples:
            examples_text = 'Here are some examples:\n\n'
            for idx, (input_ex, output_ex) in enumerate(self.examples, 1):
                examples_text += f'Example {idx}:\n'
                examples_text += f'Input: {base_model_dump_json(input_ex)}\n'
                examples_text += f'Output: {base_model_dump_json(output_ex)}\n\n'
            sections.append(
                PromptSection(name='examples', content=examples_text, priority=3)
            )
        elif self.generation_fake_sample:
            fake_input = generate_fake_instance(self.input_model)
            fake_output = generate_fake_instance(self.output_model)
            examples_text = 'Here is a FAKE example for structure only (do not use the actual content):\n\n'
            examples_text += f'Input: {base_model_dump_json(fake_input)}\n'
            examples_text += f'Output: {base_model_dump_json(fake_output)}\n'
            sections.append(
                PromptSection(name='fake_example', content=examples_text, priority=3)
            )

        # Add the current input
        sections.append(
            PromptSection(
                name='input',
                content=f'Now process this input:\n{base_model_dump_json(data)}',
                priority=4,
            )
        )

        # Sort and concatenate sections
        sections.sort(key=lambda x: x.priority)
        return '\n\n'.join(section.content for section in sections)

    async def _execute_parser_llm_call(
        self,
        prompt: str,
        prompt_value: Optional[PromptValue] = None,
        input_data: Optional[InputModel] = None,
    ) -> Union[str, OutputModel]:
        """Execute LLM call and optionally parse output for parser-based approach."""
        model_name = getattr(self.llm, 'model', 'unknown')

        # Define provider before try block so it's available in exception handler
        provider = getattr(getattr(self.llm, 'metadata', {}), 'provider', 'unknown')

        async with self.async_span(
            'parser_llm_execution', model=model_name, mode='parser_based'
        ) as span:
            input_dump = None
            if input_data is not None:
                try:
                    input_dump = json.loads(base_model_dump_json(input_data))
                except Exception:
                    input_dump = base_model_dump_json(input_data)

            # User input only (mirrors structured span behavior)
            span.set_input(input_dump)
            try:
                with Timer() as timer:
                    response = await self.llm.acomplete(prompt)

                response_text = self._strip_markdown_code_fences(
                    getattr(response, 'text', None)
                )

                prompt_tokens = len(prompt.split())
                completion_tokens = len((response_text or '').split())

                # Estimate cost
                self.cost_estimate = LLMCostEstimator.estimate(
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                span.set_attribute('prompt_tokens', prompt_tokens)
                span.set_attribute('completion_tokens', completion_tokens)
                span.set_attribute('latency', timer.elapsed_time)
                span.set_attribute('cost_estimate', self.cost_estimate)

                # Log the call to tracer
                self.tracer.log_llm_call(
                    name='parser_llm_call',
                    model=model_name,
                    prompt=prompt,
                    response=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency=timer.elapsed_time,
                    cost_estimate=self.cost_estimate,
                    provider=provider,
                )

                span.set_output(response_text)  # Capture sanitized LLM output

                # If prompt_value is provided, also parse within this span
                if prompt_value is not None:
                    (
                        parsed_output,
                        parse_metadata,
                    ) = await self._parse_and_validate_parser(
                        response_text, prompt_value, span=span
                    )
                    return parsed_output

                return response_text or ''

            except Exception as e:
                span.set_attribute('error', str(e))
                span.set_attribute('error_type', type(e).__name__)

                self.tracer.log_llm_call(
                    name='parser_llm_call_error',
                    model=model_name,
                    prompt=prompt,
                    response='',
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=0,
                    latency=timer.elapsed_time,
                    error=str(e),
                    provider=provider,
                )
                raise GenerationError(f'LLM call failed: {str(e)}') from e

    async def _parse_and_validate_parser(
        self, output_string: str, prompt_value: PromptValue, span=None
    ) -> Tuple[OutputModel, dict]:
        """Parse and validate LLM output using parser-based approach with unified tracing."""
        # Initialize LLM parser if needed
        if not self.parser:
            from axion._core.parsers.parse.parser import AIOutputParser

            self.parser = AIOutputParser(
                llm=self.llm, output_model=self.output_model, tracer=self.tracer
            )

        try:
            with Timer() as parse_timer:
                parsed_output, parse_metadata = await self.parser.parse_output_string(
                    output_string=output_string, prompt_value=prompt_value
                )

            if span:
                span.set_attribute('parse_success', True)
                span.set_attribute('parse_time', parse_timer.elapsed_time)
                span.set_attribute(
                    'parser_max_retries', getattr(self.parser, 'max_retries', 0)
                )
                span.set_attribute('parser_model_name', self.output_model.__name__)
                span.set_attribute('parse_attempts', parse_metadata.get('attempts', 1))

            return parsed_output, parse_metadata

        except Exception as e:
            if span:
                span.set_attribute('parse_success', False)
                span.set_attribute('parse_error', str(e))
                span.set_attribute('error_type', type(e).__name__)
            logger.error_highlight(f'Failed to parse/validate output: {str(e)}')
            raise

    async def _execute_with_retry(self, input_data: InputModel) -> OutputModel:
        """
        Execute LLM generation with rate-limit aware retry logic.

        This method implements intelligent retry behavior:
        - Detects rate limit (429) errors
        - Extracts reset time from error messages
        - Waits for the appropriate duration before retrying
        - Falls back to exponential backoff for non-rate-limit errors

        Execution paths:
        - Structured mode (default): Uses LiteLLM for native structured output
          (supports OpenAI, Anthropic, Google, and 100+ providers)
        - Parser mode: Fallback for models without structured output support
        """
        processed_data = self.process_input(input_data)
        last_exception = None

        attempt = 0
        while True:
            try:
                if self.as_structured_llm:
                    # Unified path via LiteLLM (handles all providers)
                    messages = self._build_message_dicts(processed_data)
                    try:
                        result = await self._execute_structured_call(
                            messages, attempt=attempt + 1, input_data=processed_data
                        )
                    except Exception as structured_error:
                        if self.fallback_to_parser and not self._is_rate_limit_error(
                            structured_error
                        ):
                            logger.warning_highlight(
                                f'Structured call failed: {str(structured_error)}. '
                                f'Falling back to parser mode.'
                            )

                            # Fallback to parser mode
                            prompt = self._build_parser_prompt(processed_data)
                            prompt_value = PromptValue(text=prompt)
                            result = await self._execute_parser_llm_call(
                                prompt, prompt_value, input_data=processed_data
                            )
                        else:
                            raise structured_error
                else:
                    # Parser fallback for models without structured output
                    prompt = self._build_parser_prompt(processed_data)
                    prompt_value = PromptValue(text=prompt)
                    result = await self._execute_parser_llm_call(
                        prompt, prompt_value, input_data=processed_data
                    )

                # Process the output through the base handler
                return self.process_output(result, input_data)

            except Exception as e:
                last_exception = e
                is_rate_limit = self._is_rate_limit_error(e)

                max_attempts = self.max_retries
                if is_rate_limit:
                    max_attempts = max(self.max_retries, self.max_rate_limit_retries)

                if attempt >= max_attempts - 1:
                    # Final attempt failed
                    logger.error_highlight(
                        f'❌ All {max_attempts} attempts failed. Final error: {str(e)}'
                    )
                    break
                else:
                    # Calculate delay with rate-limit awareness
                    retry_delay = self._calculate_retry_delay(attempt, e)

                    # Log with appropriate context
                    if is_rate_limit:
                        rate_info = RateLimitInfo.from_error(str(e))
                        if rate_info:
                            logger.warning_highlight(
                                f'⚠️  Rate limit exceeded (attempt {attempt + 1}): '
                                f'{rate_info}. Waiting {retry_delay:.1f}s before retry...'
                            )
                        else:
                            logger.warning_highlight(
                                f'⚠️  Rate limit error detected (attempt {attempt + 1}). '
                                f'Waiting {retry_delay:.1f}s before retry...'
                            )
                    else:
                        logger.warning_highlight(
                            f'Attempt {attempt + 1} failed: {str(e)}. '
                            f'Retrying in {retry_delay}s...'
                        )

                    attempt += 1
                    await asyncio.sleep(retry_delay)

        # All attempts failed
        raise GenerationError('All retry attempts failed.') from last_exception

    @log_execution_time
    async def execute(
        self,
        input_data: Union[InputModel, Dict[str, str]],
        callbacks: Callbacks = None,
    ) -> OutputModel:
        """
        Execute the handler to generate a structured output from the given input.

        This is the main entry point for the handler. It orchestrates the entire
        generation process, including input validation, execution with retries,
        and output formatting, all within a monitored and traced context.

        Uses the BaseHandler's generation context and unified tracer
        for seamless observability without redundant logging logic.

        Args:
            input_data (Union[InputModel, Dict[str, str]]): The data to be processed,
                either as a dictionary or a pre-validated Pydantic model instance.
            callbacks (Callbacks, optional): A list of callback objects to be notified
                of lifecycle events during generation. Defaults to None.

        Returns:
            OutputModel: An instance of the handler's defined `output_model` containing
                the structured result from the LLM.
        """
        async with self._generation_context(callbacks or []):
            formatted_input = self.format_input_data(input_data)
            input_dump = (
                formatted_input.model_dump()
                if isinstance(formatted_input, BaseModel)
                else formatted_input
            )

            if hasattr(self, 'tracer'):
                self.tracer.metadata.input_data = input_dump
                # Note: Don't set_input on current_span here - it may be a parent span
                # The child span created in _execute_structured_call will set its own input

            result = await self._execute_with_retry(formatted_input)

            if hasattr(self, 'tracer'):
                try:
                    result_dump = result.model_dump()
                    self.tracer.metadata.output_data = result_dump
                    # Note: Don't set_output on current_span here - it may be a parent span
                    # The child span created in _execute_structured_call handles its own output
                except AttributeError:
                    pass

            return result

    def set_instruction(self, instruction: str):
        """
        Set a new instruction string for the handler.

        Args:
            instruction (str): The updated task instruction that guides the LLM's behavior.
        """
        self.instruction = instruction

    def set_examples(self, examples: List[Tuple[InputModel, OutputModel]]):
        """
        Replace all current examples with a new set.

        Args:
            examples (List[Tuple[InputModel, OutputModel]]): A list of example input-output pairs
                used for few-shot prompting.
        """
        self.examples = examples

    def add_examples(self, examples: List[Tuple[InputModel, OutputModel]]):
        """
        Add new example input-output pairs to the existing list of examples.

        Args:
            examples (List[Tuple[InputModel, OutputModel]]): One or more examples to add
                to the current list, extending the few-shot prompting context.
        """
        self.examples.extend(examples)

    @property
    def cost_estimate(self):
        """
        Gets the cost estimate per run.
        Cost is calculated based on prompt and completion tokens per model.
        """
        return self._cost_estimate

    @cost_estimate.setter
    def cost_estimate(self, value):
        """
        Sets the cost estimate value.

        Args:
            value (float): The estimated cost to assign.
        """
        self._cost_estimate = value

    @cost_estimate.deleter
    def cost_estimate(self):
        """
        Deletes the cost estimate value.
        """
        del self._cost_estimate

    def display_prompt(self, query_input: Union[dict, InputModel] = None, **kwargs):
        """
        Displays the fully constructed prompt that will be sent to the LLM.

        This method is useful for debugging and understanding how the handler
        is communicating with the language model. It adapts its output based
        on whether the handler is using the modern chat-based approach or the
        parser-based single-string prompt format.

        Args:
            query_input (Union[dict, InputModel], optional): The input data to be
                included in the prompt. If None, a placeholder is used. Defaults to None.
        """
        from axion._core.display import display_prompt

        if query_input:
            query_input = self.format_input_data(query_input)

        if self.as_structured_llm:
            # LiteLLM structured mode
            messages = self._build_message_dicts(query_input)
            prompt_text = '\n---\n'.join(
                [f'## {m["role"].upper()}\n\n{m["content"]}' for m in messages]
            )
        else:
            # Parser fallback mode
            if query_input is None:
                query_input = self.input_model.model_construct()
            prompt_text = self._build_parser_prompt(query_input)

        display_prompt(prompt_text, query=query_input, **kwargs)

    def display_llm_statistics(self):
        """Display LLM execution statistics from the tracer."""
        if hasattr(self, 'tracer') and hasattr(self.tracer, 'display_llm_statistics'):
            self.tracer.display_llm_statistics()
        else:
            logger.info('No tracer with statistics display found.')

    def get_execution_metadata(self) -> Dict[str, Any]:
        """
        Get comprehensive execution metadata, including handler-specific details.
        """
        model_name = getattr(self.llm, 'model', 'unknown')
        # Determine provider from model name prefix (e.g., "anthropic/claude-3" -> "anthropic")
        if '/' in str(model_name):
            provider = model_name.split('/')[0]
        else:
            provider = getattr(getattr(self.llm, 'metadata', {}), 'provider', 'openai')

        base_metadata = super().get_execution_metadata()
        return {
            **base_metadata,
            'handler_type': 'llm',
            'model_name': model_name,
            'provider': provider,
            'examples_count': len(self.examples),
            'has_parser': self.parser is not None,
            'execution_mode': 'litellm_structured'
            if self.as_structured_llm
            else 'parser_mode',
            'generation_fake_sample': self.generation_fake_sample,
            'fallback_to_parser': self.fallback_to_parser,
            'rate_limit_aware': True,
            'rate_limit_buffer': self.rate_limit_buffer,
        }
