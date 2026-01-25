from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from axion._core.logging import get_logger
from axion._core.schema import (
    Callbacks,
    EmbeddingRunnable,
    InputModel,
    LLMRunnable,
    OutputModel,
)
from axion._handlers.llm.handler import LLMHandler
from axion.dataset import DatasetItem
from axion.error import MetricRegistryError
from axion.llm_registry import LLMRegistry, MockLLM
from axion.metrics.schema import MetricConfig, MetricEvaluationResult
from axion.metrics.utils import validate_required_metric_fields

logger = get_logger(__name__)


# Ideally this should be two base classes
# 1. non-LLM based metrics (BaseMetric)
# 2. LLM based metrics (BaseLLMMetric)
# Keeping it simple for external users
class BaseMetric(LLMHandler, Generic[InputModel, OutputModel]):
    """
    Base class for all metric evaluation classes, inheriting from LLMHandler.
    """

    _default_instructions = 'JUDGE INSTRUCTIONS'
    _input_item = DatasetItem

    input_model: Type[InputModel] = DatasetItem
    output_model: Type[OutputModel] = MetricEvaluationResult
    owner: str = 'AI Engineering'
    instruction: str = _default_instructions
    requires_embeddings: bool = False
    inverse_scoring_metric: bool = False
    shares_internal_cache: bool = False
    cost_estimate: float
    examples: List[Tuple[InputModel, OutputModel]] = []

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        embed_model_name: Optional[str] = None,
        embed_model: Optional[EmbeddingRunnable] = None,
        threshold: float = None,
        llm_provider: Optional[str] = None,
        required_fields: Optional[List[str]] = None,
        optional_fields: Optional[List[str]] = None,
        metric_name: Optional[str] = None,
        metric_description: Optional[str] = None,
        name: Optional[str] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the metric with optional LLM and embedding model.

        Args:
            model_name: Name of the LLM model to use
            llm: A pre-configured LLM model. If not provided, a default is loaded from the registry.
            embed_model_name: Name of the embedding model to use
            embed_model: A pre-configured embedding model handler (if needed).
            threshold: The threshold to consider a score as 'passing'. Will overwrite default.
            llm_provider: The LLM provider to use
            required_fields: List of required field names for evaluation
            optional_fields: List of optional field names for evaluation
            metric_name: Optional name for the metric instance (alias: name)
            metric_description: Optional description for the metric instance
            name: Alias for metric_name (for convenience)
            field_mapping: Optional mapping from canonical field names to source paths.
                e.g., {'actual_output': 'additional_output.summary'} will resolve
                'actual_output' from item.additional_output['summary'].
            **kwargs: Additional keyword arguments passed to the parent LLMHandler (e.g., logger config).
        """
        self._set_llm(llm, model_name, llm_provider)
        self._set_embedding_model(embed_model, embed_model_name)
        self._threshold = threshold
        self._required_fields = required_fields
        self._optional_fields = optional_fields
        # Name takes precedence
        self._metric_name = name or metric_name
        self._metric_description = metric_description
        self._field_mapping = field_mapping or {}

        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Return the name of the metric from instance, config, or fallback to class name."""
        # Check instance-level metric_name first
        if hasattr(self, '_metric_name') and self._metric_name:
            return self._metric_name
        if (
            hasattr(self, 'config')
            and hasattr(self.config, 'name')
            and self.config.name
        ):
            return self.config.name
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """Return the description of the metric from instance, config, or fallback to class name."""
        # Check instance-level metric_description first
        if hasattr(self, '_metric_description') and self._metric_description:
            return self._metric_description
        if (
            hasattr(self, 'config')
            and hasattr(self.config, 'description')
            and self.config.description
        ):
            return self.config.description
        return self.__class__.__name__

    def _set_llm(
        self,
        llm: Optional[LLMRunnable],
        model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> None:
        """
        Sets the LLM instance and related metadata.

        If an LLM instance is provided, it is used directly. Otherwise, one is loaded from
        the LLM registry using the specified model name and provider.

        Args:
            llm (Optional[LLMRunnable]): The LLM instance to use. If not provided, it will be
                resolved from the registry.
            model_name (Optional[str]): The name of the model to use when resolving from the registry.
            llm_provider (Optional[str]): The name of the LLM provider
        """
        if self.instruction == self._default_instructions:
            logger.debug(
                'Non-LLM-based metric detected. Initializing MockLLM instance.'
            )
            self.llm = MockLLM()
        elif llm is not None:
            self.llm = llm
        else:
            registry = LLMRegistry(llm_provider)
            self.llm = registry.get_llm(model_name)

        self.model_name = model_name or getattr(self.llm, 'model', 'unknown')
        self.llm_provider = llm_provider or getattr(self.llm, 'llm_provider', 'unknown')

    def _set_embedding_model(
        self,
        embed_model: Optional[EmbeddingRunnable],
        embed_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> None:
        """
        Sets the embedding model instance and related metadata.

        If `requires_embeddings` is False, embedding is skipped. If an embedding model instance
        is provided, it is used directly. Otherwise, the embedding model is loaded from the
        registry using the specified model name and provider.

        Args:
            embed_model (Optional[EmbeddingRunnable]): The embedding model instance to use.
                If not provided, one is resolved from the registry.
            embed_model_name (Optional[str]): The name of the embedding model to resolve if not provided directly.
            llm_provider (Optional[str]): The provider name used for loading from the embedding model registry.

        Returns:
            None
        """
        if not self.requires_embeddings:
            self.embed_model = None
            self.embed_model_name = None
            return

        if embed_model is not None:
            self.embed_model = embed_model
        else:
            registry = LLMRegistry(llm_provider)
            self.embed_model = registry.get_embedding_model(embed_model_name)

        self.embed_model_name = embed_model_name or getattr(
            self.embed_model, 'embed_model', 'unknown'
        )

    async def execute(
        self,
        item: Union[DatasetItem, dict],
        callbacks: Callbacks = None,
        **kwargs,
    ) -> MetricEvaluationResult:
        """
        Execute the metric evaluation for a single dataset item.

        Args:
            item: Input dataset item containing necessary fields for evaluation.
            callbacks: Optional callback handler for events/logging.

        Returns:
            An evaluation result conforming to the output model.
        """
        if hasattr(self, 'config') or self._required_fields:
            self._validate_required_metric_fields(item)
        self._input_item = self.get_evaluation_fields(item)
        return await super().execute(self._input_item, callbacks)

    def get_evaluation_fields(
        self, item: Union[DatasetItem, dict]
    ) -> Union[DatasetItem, InputModel]:
        """
        Extracts the appropriate evaluation fields from the dataset item.

        Priority is given to explicitly set required and optional fields on the instance.
        If not defined, configuration-based fields are used. If none are available,
        the item's default evaluation fields are returned.

        Args:
            item (DatasetItem): The input dataset item.

        Returns:
            DatasetItem | InputModel: A dataset item containing only the relevant fields for evaluation.
        """
        if isinstance(item, dict):
            item = DatasetItem(**item)
        fields = self.required_fields + self.optional_fields
        if fields:
            return item.subset(fields)
        if isinstance(item, DatasetItem):
            return item.evaluation_fields()
        return item

    def _validate_required_metric_fields(self, item: Union[DatasetItem, dict]) -> None:
        """
        Validate that the input item contains all required fields for this metric.

        Args:
            item (DatasetItem): The input dataset item.
        """
        if isinstance(item, dict):
            item = DatasetItem(**item)
        validate_required_metric_fields(
            item, self.required_fields, self.name, self._field_mapping
        )

    def get_field(self, item: DatasetItem, field_name: str, default: Any = None) -> Any:
        """
        Resolve a field value from DatasetItem, respecting field_mapping overrides.

        If a mapping is defined for the given field_name, this method resolves the value
        from the mapped source path. Otherwise, it returns the attribute directly from
        the item.

        Args:
            item: The DatasetItem to extract from
            field_name: Canonical field name (e.g., 'actual_output')
            default: Value to return if field is not found

        Returns:
            The resolved field value

        Example:
            # With field_mapping={'actual_output': 'additional_output.summary'}
            value = self.get_field(item, 'actual_output')  # Gets item.additional_output['summary']
        """
        source_path = self._field_mapping.get(field_name)

        if source_path:
            return self._resolve_path(item, source_path, default)
        else:
            return getattr(item, field_name, default)

    def _resolve_path(self, item: DatasetItem, path: str, default: Any = None) -> Any:
        """
        Resolve dot-notation path to get nested values from a DatasetItem.

        Supports both attribute access (for DatasetItem fields) and dictionary key access
        (for nested dictionaries like additional_output or additional_input).

        Args:
            item: The DatasetItem to resolve the path from
            path: Dot-notation path (e.g., 'additional_output.summary')
            default: Value to return if path cannot be resolved

        Returns:
            The resolved value, or default if not found

        Example:
            # item.additional_output = {'summary': 'Hello world'}
            _resolve_path(item, 'additional_output.summary')  # Returns 'Hello world'
        """
        parts = path.split('.')
        current = item

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
            if current is None:
                return default

        return current

    def get_mapped_fields(self, item: DatasetItem) -> Dict[str, Any]:
        """
        Return all required and optional fields with resolved values.

        This convenience method resolves all configured fields (both required and optional)
        from the DatasetItem, applying any field mappings.

        Args:
            item: The DatasetItem to extract fields from

        Returns:
            Dictionary mapping field names to their resolved values

        Example:
            # With required_fields=['actual_output', 'expected_output']
            # and field_mapping={'actual_output': 'additional_output.summary'}
            fields = metric.get_mapped_fields(item)
            # Returns {'actual_output': <resolved value>, 'expected_output': <resolved value>}
        """
        fields = {}
        for field in (self.required_fields or []) + (self.optional_fields or []):
            fields[field] = self.get_field(item, field)
        return fields

    def set_instruction(self, instruction: str):
        """
        Set a new instruction string for the metric.

        Args:
            instruction (str): The updated task instruction that guides the metric’s behavior or LLM prompt.
        """
        self.instruction = instruction

    def set_examples(self, examples: List[Tuple[DatasetItem, MetricEvaluationResult]]):
        """
        Replace all current examples with a new set.

        Args:
            examples (List[Tuple[DatasetItem, EvaluationResult]]): A list of example input-output pairs
                used for few-shot prompting or metric calibration.

        Example:
            [
                (
                    DatasetItem(
                        expected_output='....',
                        actual_output='...',
                    ),
                    MetricEvaluationResult(
                        score=...,
                        explanation="...",
                    ),
                ),
            ]
        """
        self.examples = examples

    def add_examples(self, examples: List[Tuple[DatasetItem, MetricEvaluationResult]]):
        """
        Add new example input-output pairs to the existing list of examples.

        Args:
            examples (List[Tuple[DatasetItem, EvaluationResult]]): One or more examples to add
                to the current list, extending few-shot prompting context.

        Example:
            [
                (
                    DatasetItem(
                        expected_output='....',
                        actual_output='...',
                    ),
                    MetricEvaluationResult(
                        score=...,
                        explanation="...",
                    ),
                ),
            ]
        """
        self.examples.extend(examples)

    @property
    def threshold(self):
        """Metric passing threshold."""
        return self._threshold or self.config.default_threshold

    @property
    def input_item(self):
        """Access the final DatasetItem passed the metric"""
        return self._input_item

    @property
    def required_fields(self) -> list:
        """
        Returns the required fields for evaluation.

        Falls back to configuration if instance-level fields are not explicitly set.
        """
        if hasattr(self, '_required_fields') and self._required_fields is not None:
            return self._required_fields
        if hasattr(self, 'config') and hasattr(self.config, 'required_fields'):
            return self.config.required_fields
        return []

    @required_fields.setter
    def required_fields(self, fields: list):
        """Sets the required fields for evaluation."""
        self._required_fields = fields

    @property
    def optional_fields(self) -> list:
        """
        Returns the optional fields for evaluation.

        Falls back to configuration if instance-level fields are not explicitly set.
        """
        if hasattr(self, '_optional_fields') and self._optional_fields is not None:
            return self._optional_fields
        if hasattr(self, 'config') and hasattr(self.config, 'optional_fields'):
            return self.config.optional_fields
        return []

    @optional_fields.setter
    def optional_fields(self, fields: list):
        """Sets the optional fields for evaluation."""
        self._optional_fields = fields

    def compute_cost_estimate(self, sub_models: List['BaseMetric']):
        """
        Computes the total estimated cost, including the current model and any sub-models.

        Args:
            sub_models (List[BaseMetric]): List of sub-models that may have a cost_estimate.
        """
        current_cost = float(getattr(self, 'cost_estimate', 0.0) or 0.0)
        sub_model_cost = sum(
            model.cost_estimate
            for model in sub_models
            if hasattr(model, 'cost_estimate')
            and isinstance(model.cost_estimate, float)
        )

        total_cost = current_cost + sub_model_cost
        self.cost_estimate = total_cost

    def display_prompt(self, item: Union[dict, InputModel] = None, **kwargs):
        """
        Displays the fully constructed prompt that will be sent to the LLM.

        Args:
            item (Union[dict, InputModel], optional): The input data to be
                included in the prompt. If None, a placeholder is used. Defaults to None.
        """
        if item:
            item = self.get_evaluation_fields(item)
        super().display_prompt(item, **kwargs)

    def __repr__(self):
        """Get a string representation of the handler."""
        return f'{self.__class__.__name__}(name={self.name}, description={self.description})'


def metric(
    name: str,
    description: str,
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None,
    key: Optional[str] = None,
    default_threshold: float = 0.5,
    score_range: tuple[Union[int, float], Union[int, float]] = (0, 1),
    tags: Optional[List[str]] = None,
) -> Callable[[Type[BaseMetric]], Type[BaseMetric]]:
    """
    Decorator to define and register a metric class with declarative configuration.

    Args:
        name: Human-readable name of the metric.
        description: Description of what the metric measures.
        required_fields: Fields that must be present in the DatasetItem to evaluate this metric.
        optional_fields: Optional fields the metric may use if available.
        key: Optional. A unique programmatic identifier for the metric.
             If not provided, it's generated from the name.
        default_threshold: The default threshold to consider a score as 'passing'.
        score_range: Tuple representing the valid score range for this metric.
        tags: Searchable tags to group or filter metrics.

    Returns:
        A class decorator that attaches config and registers the metric in the MetricRegistry.

    Raises:
        TypeError: If the decorated class is not a subclass of BaseMetric.
    """

    def decorator(cls: Type[BaseMetric]) -> Type[BaseMetric]:
        if not issubclass(cls, BaseMetric):
            raise TypeError(
                f'@metric decorator can only be applied to subclasses of BaseMetric, got: {cls.__name__}'
            )

        final_key = key if key is not None else '_'.join(name.lower().split())

        cls.config = MetricConfig(
            key=final_key,
            name=name,
            description=description,
            required_fields=required_fields,
            optional_fields=optional_fields or [],
            default_threshold=default_threshold,
            tags=tags or [],
            score_range=score_range,
        )

        metric_registry.register(cls)
        return cls

    return decorator


class MetricRegistry:
    """
    Registry for storing and retrieving metric classes.
    """

    _registry: Dict[str, Type[BaseMetric]] = {}
    _initial_state: Dict[str, Type[BaseMetric]] = None

    def finalize_initial_state(self) -> None:
        """Call this after all built-in metrics are registered."""
        if self._initial_state is None:
            self._initial_state = self._registry.copy()
            logger.debug(
                f'Captured initial state with {len(self._initial_state)} metrics'
            )

    def register(self, metric_class: Type[BaseMetric]) -> None:
        """
        Register a metric class into the registry.

        Args:
            metric_class: A class inheriting from BaseMetric with a valid config.
        """
        config = getattr(metric_class, 'config', None)
        if not isinstance(config, MetricConfig):
            raise MetricRegistryError(
                f"Metric '{metric_class.__name__}' must have a valid MetricConfig."
            )

        key = config.key
        if key in self._registry:
            logger.warning(
                f"Metric with key '{key}' is already registered. Overwriting"
            )

        self._registry[key] = metric_class

    @classmethod
    def reset(cls) -> None:
        """Reset registry to original state."""
        if cls._initial_state is not None:
            cls._registry.clear()
            cls._registry.update(cls._initial_state)
            logger.info(
                f'Reset to initial state with {len(cls._initial_state)} metrics'
            )

    def get(self, key: str, error: bool = True) -> Optional[Type[BaseMetric]]:
        """
        Retrieve a registered metric class by key.

        Args:
            key: The unique key of the metric.
            error: If True, raise an error if the key is not found.
                   If False, return None instead.

        Returns:
            The registered metric class, or None if not found and error=False.
        """
        try:
            return self._registry[key]
        except KeyError:
            if error:
                raise KeyError(
                    f"Metric '{key}' not found. Available keys: {list(self._registry.keys())}"
                )
            return None

    def find(self, query: str) -> List[Type[BaseMetric]]:
        """
        Search for metrics whose name, description, or tags match a query.

        Args:
            query: Case-insensitive search string.

        Returns:
            A list of matching metric classes.
        """
        query = query.lower()
        return [
            cls
            for cls in self._registry.values()
            if query in cls.config.name.lower()
            or query in cls.config.description.lower()
            or any(query in tag.lower() for tag in cls.config.tags)
        ]

    def get_compatible_metrics(self, item: DatasetItem) -> List[Type[BaseMetric]]:
        """
        Return all metrics compatible with a given DatasetItem.

        Args:
            item: The dataset item to test against.

        Returns:
            A list of compatible metric classes.
        """
        available_fields = {
            field for field, value in item.model_dump().items() if value is not None
        }

        return [
            cls
            for cls in self._registry.values()
            if set(cls.config.required_fields).issubset(available_fields)
        ]

    def get_metric_descriptions(self) -> Dict[str, str]:
        """Return {metric_name: description} from the registry."""
        return {
            name: getattr(getattr(metric, 'config', None), 'description', '') or ''
            for name, metric in self._registry.items()
        }

    def display(self, show_examples: bool = False) -> None:
        """
        Display a summary of all registered metrics.

        Args:
            show_examples: Show custom LLM examples
        """

        from axion.docs.display_registry import (
            create_metric_display,
        )

        metric_display = create_metric_display()
        metric_display.display(self._registry)
        if show_examples:
            MetricRegistry.display_examples()

    def display_table(self) -> None:
        """
        Display a formatted table of all registered metrics.
        """
        from axion.docs.display_registry import display_table

        display_table(self._registry)

    @classmethod
    def display_examples(cls):
        from axion.docs.metric import (
            heuristic_metric_template,
            multi_turn_metric_template,
            single_turn_metric_template,
            yaml_metric_template,
        )
        from axion.docs.render import create_multi_usage_modal_card

        cards = create_multi_usage_modal_card(
            key='custom_metrics',
            title='Building Custom Metrics',
            description='Choose the right template for your evaluation needs. '
            'Single-turn metrics evaluate simple input-output pairs, '
            'while multi-turn metrics assess entire conversation flows '
            'with context and dialogue quality.',
            usage_templates=[
                (single_turn_metric_template, '📝 Single-Turn Metric Template'),
                (multi_turn_metric_template, '💬 Multi-Turn Conversation Template'),
                (heuristic_metric_template, '⚡ Heuristic Template'),
                (yaml_metric_template, '📄 YAML Driven Template'),
            ],
            max_width='1350px',
        )
        from IPython.display import HTML, display

        display(HTML(cards))

    def __iter__(self) -> Iterator[Type[BaseMetric]]:
        return iter(self._registry.values())

    def __len__(self) -> int:
        return len(self._registry)


# For general use
metric_registry = MetricRegistry()
