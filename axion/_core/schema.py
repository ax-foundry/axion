import inspect
import json
from dataclasses import dataclass, fields
from enum import Enum
from textwrap import fill
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from axion._core.error import CustomValidationError
from axion._core.uuid import uuid7
from llama_index.core.base.llms.base import CompletionResponse
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic.config import ConfigDict

InputModel = TypeVar('InputModel', bound=BaseModel)
OutputModel = TypeVar('OutputModel', bound=BaseModel)
Callbacks = TypeVar('Callbacks')
E = TypeVar('E', bound='RichEnum')


###################################
# BASE MODELS
###################################
class RichBaseModel(BaseModel):
    """Base class for all prompt input and outputs"""

    model_config = ConfigDict(
        extra='forbid',
        # Required for OpenAI structured outputs (strict mode) which requires
        # ALL properties to be in the 'required' array, even those with defaults
        json_schema_serialization_defaults_required=True,
    )

    def __repr__(self) -> str:
        """Returns a detailed JSON representation for debugging."""
        return f'{self.__class__.__name__}(\n{self.model_dump_json(indent=4, exclude_none=True, by_alias=True)}\n)'

    def __str__(self) -> str:
        """Returns a more concise, user-friendly string representation."""
        return f'{self.__class__.__name__}: {json.dumps(self.model_dump(exclude_none=True, by_alias=True), indent=2)}'

    def __getitem__(self, key: str):
        """Allows dictionary-like access to attributes."""
        return getattr(self, key, None)

    def to_dict(self, exclude_none: bool = True, **kwargs):
        """Converts the model to a dictionary."""
        return self.model_dump(by_alias=True, exclude_none=exclude_none, **kwargs)

    def to_json(self, exclude_none: bool = True) -> str:
        """Converts the model to a json."""
        return json.dumps(
            self.model_dump(exclude_none=exclude_none, by_alias=True), indent=2
        )

    def clean_model_dump(
        self, exclude_keys: List[str] = None
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Returns a dictionary of the cleaned model.

        Args:
            exclude_keys (List[str], optional): Keys to exclude from output. Defaults to None.

        Returns:
            Dict[str, Any]: Cleaned dictionary representation of the model.
        """
        # special to remove any ID fields
        exclude_keys = exclude_keys or ['id']

        def _clean(value: Union[Dict, List], exclude: set) -> Union[Dict, List]:
            if isinstance(value, dict):
                cleaned = {}
                for k, v in value.items():
                    if k in exclude:
                        continue
                    cleaned_v = _clean(v, exclude)
                    if cleaned_v not in (None, '', [], {}) and cleaned_v != {}:
                        cleaned[k] = cleaned_v
                return cleaned
            elif isinstance(value, list):
                return [
                    _clean(item, exclude)
                    for item in value
                    if item not in (None, '', [], {})
                ]
            return value

        raw = self.model_dump(exclude_none=True, by_alias=True)
        return _clean(raw, set(exclude_keys or []))

    def clean_model_dump_json(
        self, exclude_keys: List[str] = None, indent: int = 2
    ) -> str:
        """
        Returns a JSON string of the cleaned model.

        Args:
            exclude_keys (List[str], optional): Keys to exclude from output. Defaults to None.
            indent: Indent value for JSON. Defaults to 2

        Returns:
            str: Cleaned JSON string.
        """
        cleaned_dict = self.clean_model_dump(exclude_keys=exclude_keys)
        return json.dumps(cleaned_dict, indent=indent)

    def summary(self) -> str:
        """Returns a quick summary of the key fields."""
        data = self.model_dump(exclude_none=True)
        summary_str = ', '.join(f'{k}: {v}' for k, v in data.items())
        return f'{self.__class__.__name__}({summary_str})'

    def pretty(self) -> str:
        """Returns a formatted string representation of the object."""
        data = self.model_dump(exclude_none=True)

        if not data:
            return f'📄 {self.__class__.__name__} (empty)'

        lines = [
            f'📄 {self.__class__.__name__}',
            '─' * (len(self.__class__.__name__) + 2),
        ]

        def format_long_string(key: str, value: str) -> str:
            wrapped = fill(
                value, width=70, initial_indent='  │ ', subsequent_indent='  │ '
            )
            return f'• {key}:\n{wrapped}'

        def format_list(key: str, value: list) -> list[str]:
            output = [f'• {key}: list({len(value)} items)']
            for i, item in enumerate(value[:3]):
                preview = (
                    item
                    if not isinstance(item, str) or len(item) <= 50
                    else f'{item[:47]}...'
                )
                output.append(f'  [{i}]: {preview}')
            if len(value) > 3:
                output.append(f'  ... and {len(value) - 3} more')
            return output

        def format_dict(key: str, value: dict) -> list[str]:
            output = [f'• {key}: dict({len(value)} items)']
            for i, (k, v) in enumerate(list(value.items())[:3]):
                preview = (
                    v if not isinstance(v, str) or len(v) <= 50 else f'{v[:47]}...'
                )
                output.append(f'  {k}: {preview}')
            if len(value) > 3:
                output.append(f'  ... and {len(value) - 3} more')
            return output

        for key, value in data.items():
            if isinstance(value, str) and len(value) > 150:
                lines.append(format_long_string(key, value))
            elif isinstance(value, list):
                lines.extend(format_list(key, value))
            elif isinstance(value, dict):
                lines.extend(format_dict(key, value))
            else:
                lines.append(f'• {key}: {value!r}')

        return '\n'.join(lines)

    @classmethod
    def format_validation_error(cls, error: ValidationError) -> str:
        """Format validation errors with helpful context and suggestions."""
        error_details = []

        for err in error.errors():
            field = '.'.join(str(loc) for loc in err['loc'])
            error_msg = err['msg']
            error_type = err['type']

            # Get field description if available
            field_description = ''
            try:
                if field in cls.model_fields:
                    field_obj = cls.model_fields[field]
                    if hasattr(field_obj, 'description') and field_obj.description:
                        field_description = f' - {field_obj.description}'
            except Exception:
                pass

            # Enhanced suggestions based on error type
            suggestion = ''
            if error_type == 'literal_error':
                allowed_values = err.get('ctx', {}).get('expected', [])
                if allowed_values:
                    suggestion = f"\n  Allowed values: {', '.join(repr(v) for v in allowed_values)}"

                    # Check for possible typos
                    input_value = err.get('input')
                    if input_value and isinstance(input_value, (str, int, float)):
                        import difflib

                        close_matches = difflib.get_close_matches(
                            str(input_value), [str(v) for v in allowed_values]
                        )
                        if close_matches:
                            suggestion += f'\n  Did you mean? {close_matches[0]!r}'

            elif error_type == 'missing':
                suggestion = '\n  This field is required and must be provided.'

                # Add helpful context about what's expected
                if hasattr(cls, 'model_fields') and field in cls.model_fields:
                    field_info = cls.model_fields[field]
                    if hasattr(field_info, 'annotation'):
                        suggestion += (
                            f'\n  Expected type: {field_info.annotation.__name__}'
                        )

            elif error_type == 'type_error':
                expected_type = err.get('ctx', {}).get('expected_type', 'unknown type')
                suggestion = f'\n  Expected type: {expected_type}'

            # Add details to the error message
            detail = (
                f'❌ {field}{field_description}:\n' f'  Error: {error_msg}{suggestion}'
            )
            error_details.append(detail)

        # Add example of correct usage if available
        example_usage = cls._get_example_usage()
        if example_usage:
            error_details.append(f'\nExample usage:\n{example_usage}')

        # Combine all errors into one formatted message
        class_name = cls.__name__
        final_message = (
            f'ValidationError: {len(error_details)} error(s) for {class_name}:\n\n'
            + '\n\n'.join(error_details)
        )

        return final_message

    @classmethod
    def _get_example_usage(cls) -> str:
        """Generate an example of correct usage based on field types."""
        try:
            model_fields = cls.model_fields
            example = {}

            for field_name, field_info in model_fields.items():
                # Generate example values based on field type
                annotation = getattr(field_info, 'annotation', None)
                if not annotation:
                    example[field_name] = '...'
                    continue

                annotation_name = getattr(annotation, '__name__', str(annotation))

                if annotation == str or annotation_name == 'str':
                    example[field_name] = f'example_{field_name}'
                elif annotation == int or annotation_name == 'int':
                    example[field_name] = 42
                elif annotation == float or annotation_name == 'float':
                    example[field_name] = 3.14
                elif annotation == bool or annotation_name == 'bool':
                    example[field_name] = True
                elif annotation == list or annotation_name.startswith('list'):
                    example[field_name] = []
                elif annotation == dict or annotation_name.startswith('dict'):
                    example[field_name] = {}
                else:
                    example[field_name] = '...'

            example_str = f'{cls.__name__}(**{json.dumps(example, indent=2)})'
            return example_str
        except Exception as e:
            return f'# Error generating example: {str(e)}'

    def __init__(self, **data):
        """Custom initialization with better error handling."""
        try:
            super().__init__(**data)
        except ValidationError as e:
            # Format the error message
            formatted_error = self.__class__.format_validation_error(e)

            # Raise a custom exception with the formatted message
            raise CustomValidationError(formatted_error, e) from None

    @classmethod
    def validate_data(cls, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate data without creating an instance, returning a list of errors."""
        try:
            cls(**data)
            return []  # No errors if validation succeeds
        except ValidationError as e:
            return e.errors()

    def display(self, title: str = None, **kwargs) -> None:
        """
        Display pydantic prompt.
        """
        from axion._core.display import display_pydantic

        title = title or self.__class__.__name__
        display_pydantic(self, title, **kwargs)


@dataclass
class RichSerializer:
    """
    Base class for objects that can be serialized to/from dictionaries and JSON.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, including only non-None fields."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                if hasattr(value, 'to_dict') and callable(value.to_dict):
                    result[f.name] = value.to_dict()
                elif isinstance(value, list) and value and hasattr(value[0], 'to_dict'):
                    result[f.name] = [item.to_dict() for item in value]
                elif (
                    isinstance(value, dict)
                    and value
                    and hasattr(next(iter(value.values())), 'to_dict')
                ):
                    result[f.name] = {k: v.to_dict() for k, v in value.items()}
                elif isinstance(value, Enum):
                    result[f.name] = value.value
                else:
                    result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RichSerializer':
        """Create instance from dictionary, handling nested objects."""
        kwargs: Dict[str, Any] = {}
        type_hints = get_type_hints(cls)

        for name, hint in type_hints.items():
            if name not in data:
                continue

            value = data[name]
            if value is None:
                kwargs[name] = None
                continue

            # Handle special types
            origin = get_origin(hint)
            args = get_args(hint)

            if origin is list and args and hasattr(args[0], 'from_dict'):
                kwargs[name] = [args[0].from_dict(item) for item in value]
            elif origin is dict and len(args) == 2 and hasattr(args[1], 'from_dict'):
                kwargs[name] = {k: args[1].from_dict(v) for k, v in value.items()}
            elif hasattr(hint, 'from_dict') and callable(getattr(hint, 'from_dict')):
                kwargs[name] = hint.from_dict(value)
            elif hasattr(hint, '__origin__') and hint.__origin__ is Optional:
                # Handle Optional types
                inner_type = hint.__args__[0]
                if hasattr(inner_type, 'from_dict') and callable(
                    getattr(inner_type, 'from_dict')
                ):
                    kwargs[name] = inner_type.from_dict(value)
                else:
                    kwargs[name] = value
            elif hasattr(hint, '__origin__') and hint.__origin__ is list:
                # Handle lists of simple types
                kwargs[name] = value
            elif inspect.isclass(hint) and issubclass(hint, Enum):
                kwargs[name] = hint(value)
            else:
                kwargs[name] = value

        return cls(**kwargs)

    def save_to_json(self, file_path: str) -> None:
        """Save to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str) -> 'RichSerializer':
        """Load from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class RichEnum(Enum):
    """
    Enhanced Enum class with utility methods for case-insensitive lookups,
    value/name checks, string conversions, and IDE-friendly Literal support.
    """

    @classmethod
    def keys(cls) -> List[str]:
        """Return a list of all enum member names."""
        return [member.name for member in cls]

    @classmethod
    def values(cls) -> List[Any]:
        """Return a list of all enum member values."""
        return [member.value for member in cls]

    @classmethod
    def get_literal(cls) -> Any:
        """Return a Literal type of all enum values (for static typing support)."""
        return Literal[tuple(cls.values())]  # used for manual type hints

    @classmethod
    def get_literal_values(cls) -> List[str]:
        """Return a list of all enum values as string literals (for type generation)."""
        return list(map(str, cls.values()))

    @classmethod
    def from_str(cls: Type[E], string: str, default: Optional[E] = None) -> E:
        """
        Retrieve enum member by string value (case-insensitive for strings).
        """
        if string is None:
            if default is not None:
                return default
            raise ValueError(f'Cannot look up None in {cls.__name__}')

        for member in cls:
            val = member.value
            if string == val or (
                isinstance(val, str) and string.lower() == val.lower()
            ):
                return member

        if default is not None:
            return default

        raise KeyError(f"'{string}' not found in {cls.__name__}")

    @classmethod
    def has_value(cls, value: Any) -> bool:
        """Check if a value exists in the enum (case-insensitive for strings)."""
        if value is None:
            return False

        return any(
            value == member.value
            or (
                isinstance(value, str)
                and isinstance(member.value, str)
                and value.lower() == member.value.lower()
            )
            for member in cls
        )

    @classmethod
    def has_name(cls, name: str) -> bool:
        """Check if a name exists in the enum (case-insensitive)."""
        if name is None:
            return False

        return any(name.lower() == member.name.lower() for member in cls)

    def __str__(self) -> str:
        return str(self.value)


###################################
# BASE MESSAGES
###################################


class BaseMessage(RichBaseModel):
    """Base model for any message in a conversation."""

    role: str
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class HumanMessage(BaseMessage):
    """A message from a user."""

    role: Literal['user'] = 'user'
    content: str


class ToolCall(RichBaseModel):
    """Represents the AI's request to call a tool."""

    id: str = Field(default_factory=lambda: f'call_{uuid7()}')
    name: str
    args: Dict[str, Any]
    tags: List[str] = Field(
        default_factory=list,
        description="Descriptive tags for categorizing the tool call (e.g., 'RAG', 'GUARDRAIL').",
    )
    request_latency: Optional[float] = Field(
        default=None, description='Time taken for the LLM to generate this tool call.'
    )


class ToolMessage(BaseMessage):
    """The result from a tool execution."""

    role: Literal['tool'] = 'tool'
    tool_call_id: str
    content: str
    is_error: bool = Field(
        default=False,
        description='Indicates if the tool execution resulted in an error.',
    )
    tool_output: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            'The full, structured output from the tool, for tracing and evaluation. '
            "The 'content' field should be the string representation for the LLM."
        ),
    )

    # This maps to the `executionLatency` from `invokedActions`
    latency: Optional[float] = Field(
        default=None, description='Time taken for the tool to execute, in seconds.'
    )

    # This maps to the parsed text chunks from `summary.searchResults`
    retrieved_content: Optional[List[str]] = Field(
        default=None, description='List of parsed text chunks retrieved by the tool.'
    )

    # This maps to the parsed sources from `summary.searchResults`
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description='List of parsed sources, e.g., [{"id": "doc1", "url": "...", "score": 0.9}]',
    )

    @field_validator('content')
    def _set_content_if_not_provided(cls, v, values):
        """
        Automatically generate a string 'content' from the structured
        fields if 'content' isn't provided explicitly.
        """
        if v is not None:
            return v

        # If content is None, create a summary
        if values.get('retrieved_content'):
            return f"Retrieved {len(values['retrieved_content'])} content chunk(s)."
        if values.get('tool_output'):
            try:
                # Try to serialize the raw output
                return json.dumps(values['tool_output'])
            except TypeError:
                return str(values['tool_output'])
        return '[Tool execution completed]'


class AIMessage(BaseMessage):
    """A message from the AI, which can contain text and/or tool calls."""

    role: Literal['assistant'] = 'assistant'
    tool_calls: Optional[List[ToolCall]] = None

    # This field stores the `llmEvents` block from the trace,
    # which contains the prompts and raw LLM responses.
    llm_events: Optional[Any] = Field(
        default=None,
        description='The list of LLM events (prompts/responses) for debugging.',
    )

    # This maps to the `topic` from `lastExecution`
    topic: Optional[str] = Field(
        default=None, description='The topic detected by the planner.'
    )


###################################
# Mimic langchain StringPromptValue
# from langchain_core.prompt_values import StringPromptValue
###################################
class PromptValue:
    def __init__(self, text: str):
        self.text = text

    def to_string(self) -> str:
        """Return the prompt as a string."""
        return self.text

    def to_messages(self) -> List[RichSerializer]:
        """Return the prompt as a list of messages."""
        return [HumanMessage(content=self.text)]


###################################
# LLM and Embedding Model Protocols
###################################
class LLMRunnable(Protocol):
    """
    Protocol for LlamaIndex LLM models.
    """

    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse: ...

    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse: ...


class EmbeddingRunnable(Protocol):
    """
    Protocol for LlamaIndex embedding models.
    """

    async def aget_text_embedding(self, text: str, **kwargs) -> List[float]: ...

    def get_text_embedding(self, texts: List[str], **kwargs) -> List[List[float]]: ...
