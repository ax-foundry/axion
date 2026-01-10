from typing import Optional

from axion._core.schema import LLMRunnable
from axion._core.tracing.handlers import BaseTraceHandler
from axion._handlers.llm.handler import LLMHandler
from axion.llm_registry import LLMRegistry
from pydantic import BaseModel


class OutputStringAndPrompt(BaseModel):
    output_string: str
    prompt_value: str


class StringIO(BaseModel):
    text: str

    def __hash__(self):
        return hash(self.text)


class FixOutputFormat(LLMHandler[OutputStringAndPrompt, StringIO]):
    instruction = """
    The provided output string does not match the required JSON schema. Please fix the output to comply with the schema.
    The output should:
    1. Be valid JSON
    2. Match the provided schema structure
    3. Contain all required fields
    4. Use correct data types for each field

    Original schema is provided in the input.
    """
    description = 'Fix the output to comply with the schema.'
    owner = 'AI Engineering'
    input_model = OutputStringAndPrompt
    output_model = StringIO
    generation_fake_sample = False

    def __init__(
        self,
        llm: Optional[LLMRunnable] = None,
        model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        tracer: Optional[BaseTraceHandler] = None,
    ):
        """
        Initialize the model.

        Args:
            llm: A pre-configured LLM. If not provided, a default is loaded from the registry.
            model_name: Name of the LLM model to use
            llm_provider: The LLM provider to use
            tracer: Optional tracer for observability
        """

        if llm is not None:
            self.llm = llm
        else:
            registry = LLMRegistry(llm_provider)
            self.llm = registry.get_llm(model_name)

        self.tracer = tracer
