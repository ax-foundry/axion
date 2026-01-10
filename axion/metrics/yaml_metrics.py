import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from axion._core.config import Config, ConfigurationError
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult
from axion.metrics.execution_environment import (
    create_execution_environment,
)
from pydantic import BaseModel, ValidationError, model_validator

try:
    from RestrictedPython import compile_restricted_exec, safe_globals

    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    compile_restricted_exec = None
    safe_globals = None

KNOWN_FIELDS = {
    'instruction',
    'heuristic',
    'examples',
    'model_name',
    'embed_model_name',
    'threshold',
    'llm_provider',
    'required_fields',
    'optional_fields',
    'name',
    'description',  # Added new fields
}

HANDLER_FUNCTION_NAME = 'evaluate'


def validate_metric_yaml_structure(config: Dict[str, Any]) -> None:
    """
    Validates the structure of a metric YAML configuration.

    Args:
        config: The loaded YAML configuration dictionary.
    """
    has_instruction = 'instruction' in config
    has_heuristic = 'heuristic' in config
    if has_instruction == has_heuristic:
        raise ConfigurationError(
            "Configuration must contain exactly one of 'instruction' or 'heuristic'.\n"
            "- 'instruction' will be used as the LLM-as-a-judge prompt.\n"
            "- 'heuristic' will be used for custom python function."
        )
    if unknown_fields := set(config.keys()) - KNOWN_FIELDS:
        fields_str = ', '.join(sorted(unknown_fields))
        logger.warning(f'Unknown configuration fields will be ignored: {fields_str}')


class HandlerCreationError(Exception):
    """Raised when handler class creation fails."""

    pass


class ConfigValidationError(Exception):
    """Raised when validation of the YAML configuration fails."""

    pass


logger = logging.getLogger(__name__)


class MetricExampleDefinition(BaseModel):
    """
    Defines an example input/output pair for metrics.
    """

    input: Dict[str, Any]
    output: Dict[str, Any]

    @model_validator(mode='after')
    def validate_metric_example(self) -> 'MetricExampleDefinition':
        """Ensure metric example has required fields."""
        # Validate output has required MetricEvaluationResult fields
        if 'score' not in self.output:
            raise ValueError('Metric example output must include "score" field')

        if 'explanation' not in self.output:
            raise ValueError('Metric example output must include "explanation" field')

        score = self.output['score']
        if not isinstance(score, (int, float)):
            raise ValueError('Metric example output "score" must be a number')

        return self


class MetricConfiguration(BaseModel):
    """
    Metric configuration with either instruction OR heuristic, plus optional examples and config.

    Attributes:
        instruction: The evaluation instruction/prompt (required if no heuristic)
        heuristic: Python code for heuristic evaluation (required if no instruction)
        examples: List of example input/output pairs (optional)
        model_name: Optional model name for LLM
        embed_model_name: Optional embedding model name
        threshold: Optional threshold override
        llm_provider: Optional LLM provider
        required_fields: Optional list of required field names for evaluation
        optional_fields: Optional list of optional field names for evaluation
        name: Optional name for the metric
        description: Optional description for the metric
    """

    instruction: Optional[str] = None
    heuristic: Optional[str] = None
    examples: Optional[List[MetricExampleDefinition]] = None
    model_name: Optional[str] = None
    embed_model_name: Optional[str] = None
    threshold: Optional[float] = None
    llm_provider: Optional[str] = None
    required_fields: Optional[List[str]] = None
    optional_fields: Optional[List[str]] = None
    name: Optional[str] = None
    description: Optional[str] = None

    @model_validator(mode='after')
    def validate_examples_if_present(self) -> 'MetricConfiguration':
        """Validate that either instruction or heuristic is provided, and examples if present."""
        # Must have either instruction or heuristic, but not both
        if not self.instruction and not self.heuristic:
            raise ValueError('Either "instruction" or "heuristic" must be provided')

        if self.instruction and self.heuristic:
            raise ValueError(
                'Cannot provide both "instruction" and "heuristic" - choose one'
            )

        if self.examples:
            for i, example in enumerate(self.examples):
                if not example.input or not example.output:
                    raise ValueError(
                        f'Example #{i + 1} must have both input and output'
                    )

        # Validate field lists if provided
        if self.required_fields is not None:
            if not isinstance(self.required_fields, list):
                raise ValueError('required_fields must be a list of strings')
            if not all(isinstance(field, str) for field in self.required_fields):
                raise ValueError('All required_fields must be strings')

        if self.optional_fields is not None:
            if not isinstance(self.optional_fields, list):
                raise ValueError('optional_fields must be a list of strings')
            if not all(isinstance(field, str) for field in self.optional_fields):
                raise ValueError('All optional_fields must be strings')

        return self


def compile_heuristic_code(heuristic_code: str) -> Any:
    """
    Safely compile heuristic code using RestrictedPython.

    Args:
        heuristic_code: The Python code string to compile

    Returns:
        Compiled code object

    Raises:
        ConfigValidationError: If code compilation fails
    """
    if RESTRICTED_PYTHON_AVAILABLE:
        try:
            # Use RestrictedPython to compile with security restrictions
            compiled_code = compile_restricted_exec(heuristic_code)
            if compiled_code.errors:
                error_msg = '; '.join(compiled_code.errors)
                raise ConfigValidationError(
                    f'Heuristic code compilation errors: {error_msg}'
                )

            return compiled_code.code
        except Exception as e:
            raise ConfigValidationError(f'Failed to compile heuristic code: {str(e)}')
    else:
        # Fallback to regular compile (less secure)
        try:
            return compile(heuristic_code, '<heuristic>', 'exec')
        except Exception as e:
            raise ConfigValidationError(f'Failed to compile heuristic code: {str(e)}')


def load_metric_from_yaml(yaml_path: Union[str, Path]) -> Type[BaseMetric]:
    """
    Load and create a metric class from a YAML configuration file.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        A dynamically generated metric class
    """
    try:
        # Load and validate the configuration
        config = Config(yaml_path).config
        yaml_path_str = str(yaml_path)
        logger.info(f'Loading metric configuration from: {yaml_path_str}')

        # Validate the overall structure first
        validate_metric_yaml_structure(config)

        # Parse into MetricConfiguration for validation
        definition = MetricConfiguration(**config)

        logger.debug(
            f"Successfully parsed metric definition with {'heuristic' if definition.heuristic else 'instruction'}"
        )

        # Create the dynamic metric class following BaseMetric pattern exactly
        class YAMLMetric(BaseMetric[DatasetItem, MetricEvaluationResult]):
            """Dynamically generated metric class based on YAML configuration."""

            instruction = (
                definition.instruction
                or 'Heuristic-based evaluation (no LLM instruction)'
            )
            _yaml_path = yaml_path_str
            _heuristic_function = None

            # Set class-level name and description if provided
            if definition.name:
                _metric_name = definition.name
            if definition.description:
                _metric_description = definition.description

            examples = []
            if definition.examples:
                examples = [
                    (
                        DatasetItem(**example.input),
                        MetricEvaluationResult(**example.output),
                    )
                    for example in definition.examples
                ]

            if definition.heuristic:
                try:
                    compiled_code = compile_heuristic_code(definition.heuristic)
                    heuristic_globals = create_execution_environment()
                    exec(compiled_code, heuristic_globals)

                    # Look for the HANDLER_FUNCTION_NAME in the executed code
                    if HANDLER_FUNCTION_NAME in heuristic_globals:
                        _heuristic_function = heuristic_globals[HANDLER_FUNCTION_NAME]
                        logger.info(
                            'Successfully compiled heuristic function with RestrictedPython'
                            if RESTRICTED_PYTHON_AVAILABLE
                            else 'Successfully compiled heuristic function with fallback'
                        )
                    else:
                        logger.warning(
                            f"Heuristic code provided but no '{HANDLER_FUNCTION_NAME}' function found"
                        )

                except ConfigValidationError:
                    # Re-raise validation errors as-is
                    raise
                except Exception as e:
                    logger.error(f'Failed to compile heuristic: {str(e)}')
                    raise ConfigValidationError(f'Invalid heuristic code: {str(e)}')

            def __init__(self, **kwargs):
                """Initialize with YAML config options and allow runtime overrides."""
                # Set defaults from the YAML definition
                default_kwargs = {
                    'model_name': definition.model_name,
                    'embed_model_name': definition.embed_model_name,
                    'threshold': definition.threshold,
                    'llm_provider': definition.llm_provider,
                    'required_fields': definition.required_fields,
                    'optional_fields': definition.optional_fields,
                    'metric_name': definition.name,
                    'metric_description': definition.description,
                }

                default_kwargs = {
                    k: v for k, v in default_kwargs.items() if v is not None
                }
                final_kwargs = {**default_kwargs, **kwargs}
                # set class name if passed
                if definition.name:
                    self.__class__.__name__ = definition.name
                super().__init__(**final_kwargs)

            @property
            def name(self) -> str:
                """Return the name of the metric from YAML config, instance config, or fallback to class name."""
                # Check YAML definition first
                if definition.name:
                    return definition.name
                # Then check if there's a config object with name
                if (
                    hasattr(self, 'config')
                    and hasattr(self.config, 'name')
                    and self.config.name
                ):
                    return self.config.name
                # Then check instance-level metric_name
                if hasattr(self, '_metric_name') and self._metric_name:
                    return self._metric_name
                # Finally fallback to class name
                return self.__class__.__name__

            @property
            def description(self) -> str:
                """Return the description of the metric from YAML config, instance config, or fallback to class name."""
                # Check YAML definition first
                if definition.description:
                    return definition.description
                # Then check if there's a config object with description
                if (
                    hasattr(self, 'config')
                    and hasattr(self.config, 'description')
                    and self.config.description
                ):
                    return self.config.description
                # Then check class-level description
                if hasattr(self, '_metric_description') and self._metric_description:
                    return self._metric_description
                # Finally fallback to class name
                return self.__class__.__name__

            async def execute(
                self, item: DatasetItem, callbacks=None
            ) -> MetricEvaluationResult:
                """Execute the metric evaluation, using heuristic if available, otherwise LLM."""
                # If heuristic is defined, use it (no LLM fallback)
                if self._heuristic_function:
                    try:
                        result = YAMLMetric._heuristic_function(item)

                        if isinstance(result, MetricEvaluationResult):
                            return result
                        elif isinstance(result, (int, float)):
                            return MetricEvaluationResult(
                                score=float(result),
                                explanation=f'Heuristic evaluation returned score: {result}',
                            )
                        elif isinstance(result, dict):
                            return MetricEvaluationResult(**result)
                        else:
                            logger.warning(
                                f'Heuristic returned unexpected type: {type(result)}'
                            )
                            return MetricEvaluationResult(
                                score=0.0,
                                explanation=f'Heuristic returned invalid type: {type(result)}',
                            )

                    except Exception as e:
                        logger.error(f'Heuristic evaluation failed: {str(e)}')
                        return MetricEvaluationResult(
                            score=0.0,
                            explanation=f'Heuristic evaluation failed: {str(e)}',
                        )

                if definition.instruction:
                    self._validate_required_metric_fields(item)
                    return await super().execute(item, callbacks)
                else:
                    raise ValueError(
                        'No heuristic function available and no instruction provided'
                    )

            def get_config_summary(cls) -> Dict[str, Any]:
                """Return a summary of the metric configuration."""
                return {
                    'instruction_length': (
                        len(cls.instruction) if definition.instruction else 0
                    ),
                    'num_examples': len(cls.examples),
                    'config_path': cls._yaml_path,
                    'owner': cls.owner,
                    'evaluation_type': (
                        'heuristic' if cls._heuristic_function else 'llm'
                    ),
                    'has_heuristic': cls._heuristic_function is not None,
                    'yaml_config': {
                        'model_name': definition.model_name,
                        'embed_model_name': definition.embed_model_name,
                        'threshold': definition.threshold,
                        'llm_provider': definition.llm_provider,
                        'required_fields': definition.required_fields,
                        'optional_fields': definition.optional_fields,
                        'name': definition.name,
                        'description': definition.description,
                        'has_heuristic': definition.heuristic is not None,
                        'has_instruction': definition.instruction is not None,
                    },
                }

            @classmethod
            def reload_from_yaml(cls) -> Type[BaseMetric[Any, Any]]:
                """Reload the metric from the YAML file."""
                return load_metric_from_yaml(cls._yaml_path)

            def __repr__(self):
                """Get a string representation of the metric."""
                eval_type = 'heuristic' if self._heuristic_function else 'llm'
                name_info = f'name={definition.name or self.__class__.__name__}'
                field_info = ''
                if definition.required_fields or definition.optional_fields:
                    req_count = (
                        len(definition.required_fields)
                        if definition.required_fields
                        else 0
                    )
                    opt_count = (
                        len(definition.optional_fields)
                        if definition.optional_fields
                        else 0
                    )
                    field_info = f', req_fields={req_count}, opt_fields={opt_count}'
                return f'{self.__class__.__name__}({name_info}, type={eval_type}, examples={len(self.examples)}{field_info})'

        # Add docstring
        evaluation_type = 'heuristic' if definition.heuristic else 'LLM'
        content_preview = (
            definition.heuristic if definition.heuristic else definition.instruction
        )

        field_summary = ''
        if definition.required_fields or definition.optional_fields:
            req_count = (
                len(definition.required_fields) if definition.required_fields else 0
            )
            opt_count = (
                len(definition.optional_fields) if definition.optional_fields else 0
            )
            field_summary = f'\n        Required Fields: {req_count}\n        Optional Fields: {opt_count}'

        name_info = f'\n        Name: {definition.name}' if definition.name else ''
        description_info = (
            f'\n        Description: {definition.description}'
            if definition.description
            else ''
        )

        YAMLMetric.__doc__ = f"""
        Dynamically generated metric from YAML configuration.

        Evaluation Type: {evaluation_type}{name_info}{description_info}
        Content Preview: {content_preview}
        Examples: {len(definition.examples) if definition.examples else 0}{field_summary}
        Config Path: {yaml_path_str}

        Generated dynamically from YAML configuration.
        """

        logger.debug(f'Successfully created metric from: {yaml_path_str}')

        return YAMLMetric

    except ValidationError as e:
        logger.error(f'Validation error: {str(e)}')
        raise ConfigValidationError(f'Error validating metric configuration: {str(e)}')
    except ConfigurationError as e:
        logger.error(f'Configuration error: {str(e)}')
        raise ConfigValidationError(str(e))
    except Exception as e:
        logger.error(f'Unexpected error creating metric: {str(e)}', exc_info=True)
        raise HandlerCreationError(f'Failed to create metric: {str(e)}')
