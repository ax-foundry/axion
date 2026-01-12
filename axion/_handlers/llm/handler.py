import asyncio
import json
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

import litellm
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
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel

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
    rate_limit_buffer: float = 1.0
    max_delay = 120.0  # Maximum delay of 2 minutes

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
                logger.warning_highlight(
                    f'⏱️  Rate limit hit: {rate_limit_info}. '
                    f'Waiting {wait_time:.1f}s for reset (attempt {attempt + 1}/{self.max_retries})'
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

    def _build_openai_messages(
        self, data: Optional[InputModel] = None
    ) -> List[Dict[str, str]]:
        """Build messages in OpenAI format for raw API calls."""
        messages = [{'role': 'system', 'content': self.instruction}]
        # Add examples as conversation pairs
        for idx, (input_example, output_example) in enumerate[Tuple[InputModel, OutputModel]](self.examples):
            messages.append(
                {
                    'role': 'user',
                    'content': f'Example input:\n{base_model_dump_json(input_example)}',
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
                    'content': f'FAKE example for structure only:\n{base_model_dump_json(fake_input)}',
                }
            )
            messages.append(
                {'role': 'assistant', 'content': base_model_dump_json(fake_output)}
            )

        # Add actual input
        if data:
            content = (
                'Process the following input and provide the output in the required structured format. '
                'Do not include any other text or explanations.\n\n'
                f'Input:\n{base_model_dump_json(data)}'
            )
        else:
            content = (
                'Process the following input and provide the output in the required structured format. '
                'Do not include any other text or explanations.\n\n'
                'Input:\n<INPUT_DATA_WILL_GO_HERE>'
            )

        messages.append({'role': 'user', 'content': content})

        return messages

    async def _execute_structured_call(
        self, messages: List[Dict[str, str]]
    ) -> OutputModel:
        """Execute structured output call using LiteLLM (supports OpenAI, Claude, Gemini, etc.)."""
        model_name = getattr(self.llm, 'model', 'gpt-4o')

        # Determine provider from model name prefix (e.g., "anthropic/claude-3" -> "anthropic")
        if '/' in model_name:
            provider = model_name.split('/')[0]
        else:
            provider = getattr(getattr(self.llm, 'metadata', {}), 'provider', 'openai')

        async with self.async_span(
            'litellm_structured_execution',
            model=model_name,
            mode='litellm_structured',
            provider=provider,
        ) as span:
            span.set_input({'messages': messages, 'model': model_name})
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
                    response = await litellm.acompletion(
                        model=model_name,
                        messages=messages,
                        response_format=response_format,
                        temperature=getattr(self.llm, 'temperature', 0.0),
                    )

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
                    self.cost_estimate = response._hidden_params.get('response_cost', 0.0)
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

                try:
                    # Log the call to tracer
                    self.tracer.log_llm_call(
                        model=model_name,
                        prompt=json.dumps(messages),
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
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=self.instruction)]

        # Add examples as conversation pairs
        for input_example, output_example in self.examples:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=f'Example input:\n{base_model_dump_json(input_example)}',
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
                    content=f'FAKE example for structure only:\n{base_model_dump_json(fake_input)}',
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
            content = (
                'Process the following input and provide the output in the required structured format. '
                'Do not include any other text or explanations.\n\n'
                f'Input:\n{base_model_dump_json(data)}'
            )
        else:
            content = (
                'Process the following input and provide the output in the required structured format. '
                'Do not include any other text or explanations.\n\n'
                'Input:\n<INPUT_DATA_WILL_GO_HERE>'
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

    async def _execute_parser_llm_call(self, prompt: str) -> str:
        """Execute LLM call and return raw string output for parser-based approach."""
        model_name = getattr(self.llm, 'model', 'unknown')

        # Define provider before try block so it's available in exception handler
        provider = getattr(getattr(self.llm, 'metadata', {}), 'provider', 'unknown')

        async with self.async_span(
            'parser_llm_execution', model=model_name, mode='parser_based'
        ) as span:
            span.set_input({'prompt': prompt, 'model': model_name})
            try:
                with Timer() as timer:
                    response = await self.llm.acomplete(prompt)

                prompt_tokens = len(prompt.split())
                completion_tokens = len(response.text.split())

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
                    model=model_name,
                    prompt=prompt,
                    response=response.text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency=timer.elapsed_time,
                    cost_estimate=self.cost_estimate,
                    provider=provider,
                )

                span.set_output(response.text)  # Capture LLM output
                return response.text

            except Exception as e:
                span.set_attribute('error', str(e))
                span.set_attribute('error_type', type(e).__name__)

                self.tracer.log_llm_call(
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
        self, output_string: str, prompt_value: PromptValue
    ) -> OutputModel:
        """Parse and validate LLM output using parser-based approach with unified tracing."""
        async with self.async_span(
            'parse_and_validate_parser', output_length=len(output_string)
        ) as span:
            # Initialize LLM parser if needed
            if not self.parser:
                from axion._core.parsers.parse.parser import AIOutputParser

                self.parser = AIOutputParser(
                    llm=self.llm, output_model=self.output_model, tracer=self.tracer
                )

            try:
                with Timer() as parse_timer:
                    parsed_output = await self.parser.parse_output_string(
                        output_string=output_string, prompt_value=prompt_value
                    )

                span.set_attribute('parse_success', True)
                span.set_attribute('parse_time', parse_timer.elapsed_time)
                span.set_attribute(
                    'parser_max_retries', getattr(self.parser, 'max_retries', 0)
                )
                span.set_attribute('model_name', self.output_model.__name__)

                return parsed_output

            except Exception as e:
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
        async with self.async_span(
            'execute_with_retry',
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        ) as span:
            span.set_input(input_data)  # Capture retry-level input
            processed_data = self.process_input(input_data)
            last_exception = None

            # Determine execution mode: structured (LiteLLM) or parser fallback
            execution_mode = 'litellm_structured' if self.as_structured_llm else 'parser_mode'

            span.set_attribute('execution_mode', execution_mode)
            span.set_attribute('fallback_enabled', self.fallback_to_parser)
            span.set_attribute('rate_limit_aware', True)

            for attempt in range(self.max_retries):
                # Each attempt gets its own span
                async with self.async_span(
                    f'attempt_{attempt + 1}',
                    attempt_number=attempt + 1,
                    auto_trace=False,
                ) as attempt_span:
                    try:
                        if self.as_structured_llm:
                            # Unified path via LiteLLM (handles all providers)
                            attempt_span.set_attribute('mode', 'litellm_structured')
                            messages = self._build_openai_messages(processed_data)
                            try:
                                result = await self._execute_structured_call(messages)
                            except Exception as structured_error:
                                if self.fallback_to_parser:
                                    logger.warning_highlight(
                                        f'Structured call failed: {str(structured_error)}. '
                                        f'Falling back to parser mode.'
                                    )
                                    attempt_span.set_attribute('fallback_to_parser', True)
                                    attempt_span.set_attribute(
                                        'structured_error', str(structured_error)
                                    )

                                    # Fallback to parser mode
                                    prompt = self._build_parser_prompt(processed_data)
                                    prompt_value = PromptValue(text=prompt)
                                    output_string = await self._execute_parser_llm_call(
                                        prompt
                                    )
                                    result = await self._parse_and_validate_parser(
                                        output_string, prompt_value
                                    )
                                else:
                                    raise structured_error
                        else:
                            # Parser fallback for models without structured output
                            attempt_span.set_attribute('mode', 'parser_mode')
                            prompt = self._build_parser_prompt(processed_data)
                            prompt_value = PromptValue(text=prompt)
                            output_string = await self._execute_parser_llm_call(prompt)
                            result = await self._parse_and_validate_parser(
                                output_string, prompt_value
                            )

                        # Process the output through the base handler
                        final_result = self.process_output(result, input_data)

                        attempt_span.add_trace(
                            'attempt_success',
                            f'Attempt {attempt + 1} succeeded',
                            {
                                'attempt_number': attempt + 1,
                                'total_attempts': attempt + 1,
                            },
                        )

                        span.set_attribute('successful_attempt', attempt + 1)
                        span.set_attribute('total_attempts', attempt + 1)
                        span.set_output(final_result)  # Capture retry-level output

                        return final_result

                    except Exception as e:
                        last_exception = e

                        attempt_span.add_trace(
                            'attempt_failed',
                            f'Attempt {attempt + 1} failed: {str(e)}',
                            {'attempt_number': attempt + 1, 'error': str(e)},
                        )

                        if attempt == self.max_retries - 1:
                            # Final attempt failed
                            span.set_attribute('all_attempts_failed', True)
                            span.set_attribute('total_attempts', self.max_retries)
                            logger.error_highlight(
                                f'❌ All {self.max_retries} attempts failed. Final error: {str(e)}'
                            )
                            break
                        else:
                            # Calculate delay with rate-limit awareness
                            retry_delay = self._calculate_retry_delay(attempt, e)

                            # Log with appropriate context
                            if self._is_rate_limit_error(e):
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

                            attempt_span.set_attribute('retry_delay', retry_delay)
                            await asyncio.sleep(retry_delay)

            # All attempts failed
            span.set_attribute('all_attempts_failed', True)
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
                if hasattr(self.tracer, 'current_span') and self.tracer.current_span:
                    span = self.tracer.current_span
                    span.set_input(formatted_input)  # Capture full input data
                    span.set_attribute('input_type', type(formatted_input).__name__)
                    span.set_attribute('input_size', len(str(input_dump)))
                    span.set_attribute('handler_type', 'llm')
                    span.set_attribute(
                        'execution_mode',
                        'litellm_structured'
                        if self.as_structured_llm
                        else 'parser_mode',
                    )

            result = await self._execute_with_retry(formatted_input)

            if hasattr(self, 'tracer'):
                try:
                    result_dump = result.model_dump()
                    self.tracer.metadata.output_data = result_dump

                    if (
                        hasattr(self.tracer, 'current_span')
                        and self.tracer.current_span
                    ):
                        span = self.tracer.current_span
                        span.set_output(result)  # Capture full output data
                        span.set_attribute('output_type', type(result).__name__)
                        span.set_attribute('output_size', len(str(result_dump)))
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
            # LiteLLM structured mode - uses OpenAI message format
            messages = self._build_openai_messages(query_input)
            prompt_text = '\n---\n'.join(
                [f"## {m['role'].upper()}\n\n{m['content']}" for m in messages]
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
            'execution_mode': 'litellm_structured' if self.as_structured_llm else 'parser_mode',
            'generation_fake_sample': self.generation_fake_sample,
            'fallback_to_parser': self.fallback_to_parser,
            'rate_limit_aware': True,
            'rate_limit_buffer': self.rate_limit_buffer,
        }
