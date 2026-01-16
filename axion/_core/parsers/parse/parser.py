import asyncio
from typing import Type

from axion._core.error import ParsingError
from axion._core.logging import get_logger
from axion._core.schema import LLMRunnable, PromptValue
from axion._core.tracing.handlers import BaseTraceHandler
from axion._core.parsers.fix_output import FixOutputFormat
from axion._core.parsers.parse.json_parser import extract_json_from_text
from pydantic import BaseModel, ValidationError

logger = get_logger(__name__)


class AIOutputParser:
    """
    Parse and validate LLM-generated text into a structured Pydantic model.

    This core method handles the complete parsing pipeline, including:

    - JSON extraction from potentially malformed text
    - Schema validation against the specified output model
    - Automatic error recovery through retry mechanisms
    - Output correction using specialized fixing prompts

    The parsing process employs a resilient approach with multiple attempts,
    allowing for recovery from common LLM output formatting issues such as
    incomplete JSON, incorrect field types, or missing required fields.
    Each retry uses the fixing prompt to request corrections based on the
    specific validation errors encountered.

    Environment Configuration:
        Configure tracing via .env file - inherits from parent tracer configuration.
    """

    handler_type = 'llm'
    generation_fake_sample = False

    def __init__(
        self,
        llm: LLMRunnable,
        output_model: Type[BaseModel],
        tracer: BaseTraceHandler,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        """
        Initialize the AIOutputParser.

        Args:
            llm: A pre-configured LLM model.
            output_model: The Pydantic model to validate output against
            tracer: Unified tracer for observability
            max_retries: Maximum number of retry attempts for parsing
            retry_delay: Delay between retry attempts in seconds
        """
        super().__init__()

        self.llm = llm
        self.output_model = output_model
        self.tracer = tracer
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fix_prompt = FixOutputFormat(llm=self.llm, tracer=self.tracer)

    async def _attempt_fix(
        self,
        output_string: str,
        prompt_value: PromptValue,
        attempt: int,
    ) -> str:
        """
        Attempt to fix malformed output using the fix prompt.

        Args:
            output_string: The malformed output string
            prompt_value: Original prompt for context
            attempt: Current attempt number

        Returns:
            Fixed output string
        """
        async with self.tracer.async_span(
            'fix_output_format',
            attempt_number=attempt + 1,
            output_length=len(output_string),
        ) as span:
            logger.debug(
                f'Attempting to fix output format (attempt {attempt + 1}/{self.max_retries})'
            )

            try:
                fixed_output = await self.fix_prompt._execute_parser_llm_call(
                    prompt_value.to_string()
                )

                span.set_attribute('fix_success', True)
                span.set_attribute('fixed_output_length', len(fixed_output))

                return fixed_output

            except Exception as e:
                span.set_attribute('fix_success', False)
                span.set_attribute('fix_error', str(e))

                logger.error(
                    f'Error fixing output format on attempt {attempt + 1}: {str(e)}'
                )
                raise

    async def _validate_output(self, json_str: str) -> BaseModel:
        """
        Validate JSON string against the output model.

        Args:
            json_str: JSON string to validate

        Returns:
            Validated Pydantic model instance
        """
        with self.tracer.span(
            'validate_output',
            json_length=len(json_str),
            model_name=self.output_model.__name__,
        ) as span:
            try:
                validated_output = self.output_model.model_validate_json(json_str)
                span.set_attribute('validation_success', True)
                span.set_attribute(
                    'validated_fields', len(self.output_model.model_fields)
                )
                return validated_output

            except ValidationError as e:
                span.set_attribute('validation_success', False)
                span.set_attribute('validation_errors', len(e.errors()))
                span.set_attribute('error_details', str(e))
                raise

    async def parse_output_string(
        self,
        output_string: str,
        prompt_value: PromptValue,
    ) -> BaseModel:
        """
        Parse and validate the output string, with automatic retries and fixes.

        Args:
            output_string: Raw output string to parse
            prompt_value: Original prompt for context

        Returns:
            Validated Pydantic model instance
        """
        async with self.tracer.async_span(
            'parse_with_retries',
            max_retries=self.max_retries,
            output_length=len(output_string),
            model_name=self.output_model.__name__,
        ) as main_span:
            current_output = output_string
            last_error = None

            for attempt in range(1, self.max_retries + 1):
                # Create child span for each attempt
                async with self.tracer.async_span(
                    f'parse_attempt_{attempt}',
                    attempt_number=attempt,
                    current_output_length=len(current_output),
                ) as attempt_span:
                    try:
                        # Use synchronous span for JSON extraction (quick operation)
                        with self.tracer.span('extract_json') as extract_span:
                            json_str = extract_json_from_text(current_output)
                            extract_span.set_attribute(
                                'extracted_json_length', len(json_str)
                            )

                        result = await self._validate_output(json_str)

                        # Log success
                        attempt_span.set_attribute('parse_success', True)
                        attempt_span.set_attribute('json_length', len(json_str))

                        main_span.set_attribute('successful_attempt', attempt)
                        main_span.set_attribute('total_attempts', attempt)

                        logger.debug(f'Successfully parsed output on attempt {attempt}')
                        return result

                    except (ParsingError, ValidationError) as e:
                        last_error = e

                        # Log failure
                        attempt_span.set_attribute('parse_success', False)
                        attempt_span.set_attribute('error_type', type(e).__name__)
                        attempt_span.set_attribute('error_message', str(e)[:200])

                        logger.warning(
                            f'Parse attempt {attempt} failed: {str(e)[:200]}'
                        )

                        if attempt == self.max_retries:
                            break  # Exit loop without fix attempt

                        try:
                            current_output = await self._attempt_fix(
                                current_output, prompt_value, attempt
                            )

                            attempt_span.set_attribute('fix_attempted', True)
                            attempt_span.set_attribute(
                                'fixed_output_length', len(current_output)
                            )

                            logger.debug(f'Fix attempt {attempt} completed')
                            await asyncio.sleep(self.retry_delay)

                        except Exception as fix_error:
                            attempt_span.set_attribute('fix_attempted', True)
                            attempt_span.set_attribute('fix_success', False)
                            attempt_span.set_attribute(
                                'fix_error', str(fix_error)[:200]
                            )

                            logger.error(
                                f'Fix attempt {attempt} failed: {str(fix_error)}'
                            )
                            last_error = fix_error

            # All attempts failed
            error_msg = (
                f'Failed to parse output after {self.max_retries} attempts. '
                f'Last error: {str(last_error)}'
            )

            main_span.set_attribute('all_attempts_failed', True)
            main_span.set_attribute('total_attempts', self.max_retries)
            main_span.set_attribute('final_error_type', type(last_error).__name__)

            logger.error(error_msg)
            raise ParsingError(error_msg)

    def get_parsing_statistics(self) -> dict:
        """Return parsing statistics from the tracer, if available."""
        return (
            self.tracer.get_llm_statistics()
            if hasattr(self.tracer, 'get_llm_statistics')
            else {}
        )

    def display_parsing_traces(self):
        """Display parsing traces for debugging."""
        if hasattr(self.tracer, 'display_traces'):
            self.tracer.display_traces()

    def get_span_context(self) -> tuple:
        """Get current span context."""
        return self.tracer.get_current_span_context()

    def __repr__(self):
        """String representation of the parser."""
        return f'AIOutputParser(model={self.output_model.__name__}, max_retries={self.max_retries})'
