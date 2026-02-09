import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, Field, field_serializer, model_validator

from axion._core.schema import RichBaseModel
from axion._core.types import MetricCategory

T = TypeVar('T', bound=BaseModel)

DEFAULT_EXPLANATION = 'Check signals for additional details'


class MetricConfig(RichBaseModel):
    """
    Configuration model for all static metadata required to register and use a metric.
    This config is attached to each metric class at registration time and used for
    introspection, filtering, and UI display.
    """

    key: str = Field(
        ...,
        description='Unique identifier for the metric (typically a lowercase, snake_case version of the name).',
    )

    name: str = Field(..., description='Human-readable name of the metric.')

    description: str = Field(
        ...,
        description="Detailed explanation of what the metric evaluates and how it's used.",
    )

    metric_category: MetricCategory = Field(
        default=MetricCategory.SCORE,
        description='The category of the metric output: SCORE (numeric), ANALYSIS (structured insights), or CLASSIFICATION (labels).',
    )

    default_threshold: Optional[float] = Field(
        default=0.5,
        description='Default threshold used to determine if a score is considered a passing score. Optional for ANALYSIS metrics.',
    )

    score_range: Optional[Tuple[Union[int, float], Union[int, float]]] = Field(
        default=(0, 1),
        description='Tuple representing the valid score range for this metric (e.g., 0 to 1). Optional for ANALYSIS metrics.',
    )

    required_fields: List[str] = Field(
        default_factory=list,
        description='List of required fields that must be present in a DatasetItem for the metric to run.',
    )

    optional_fields: List[str] = Field(
        default_factory=list,
        description='Optional fields the metric can use if available, but does not require.',
    )

    tags: List[str] = Field(
        default_factory=list,
        description='List of searchable tags used for categorization, filtering, or grouping.',
    )

    @model_validator(mode='after')
    def validate_score_config(self) -> 'MetricConfig':
        """Validate that SCORE metrics have threshold and score_range set."""
        if self.metric_category == MetricCategory.SCORE:
            if self.default_threshold is None:
                self.default_threshold = 0.5
            if self.score_range is None:
                self.score_range = (0, 1)
        return self


class MetricEvaluationResult(RichBaseModel):
    """
    Standardized output model for the result of a single evaluation metric.

    This model holds the final score, an explanation of how the score was derived,
    and any additional metadata useful for debugging or traceability.

    For analysis metrics (metric_category=ANALYSIS), score can be None.
    The score will be normalized to np.nan downstream for consistent handling.

    The `metadata` and `signals` fields are excluded from JSON schema generation
    to ensure compatibility with OpenAI structured output (which requires
    additionalProperties: false for all objects).
    """

    score: Optional[Union[int, float]] = Field(
        default=None,
        description="The final score assigned by the metric. Must be within the metric's defined score range. Optional for ANALYSIS metrics.",
    )

    explanation: Optional[str] = Field(
        default=DEFAULT_EXPLANATION,
        description='Explanation of how the score was derived. May be plain text, dict, or another model.',
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description='Additional structured data useful for inspection, debugging, or traceability.',
    )

    signals: Optional[Any] = Field(
        default=None,
        description='A dictionary of dynamically generated signals providing a granular breakdown of the metric.',
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> Dict[str, Any]:
        """
        Generate JSON schema excluding fields incompatible with OpenAI structured output.

        The `metadata` and `signals` fields use Dict[str, Any] and Any types which
        cannot satisfy OpenAI's additionalProperties: false requirement.
        """
        schema = super().model_json_schema(*args, **kwargs)

        # Remove incompatible fields from schema
        if 'properties' in schema:
            schema['properties'].pop('metadata', None)
            schema['properties'].pop('signals', None)

        # Remove from required if present
        if 'required' in schema:
            schema['required'] = [
                f for f in schema['required'] if f not in ('metadata', 'signals')
            ]

        return schema

    @field_serializer('explanation', when_used='json')
    def serialize_explanation(self, value: Any) -> str:
        """
        Serialize the explanation field to a clean JSON string for consistent output formatting.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, BaseModel):
            return value.model_dump_json(indent=2)
        try:
            return json.dumps(value, indent=2)
        except TypeError:
            return str(value)


@dataclass
class SignalDescriptor(Generic[T]):
    """A declarative blueprint for defining a single, atomic signal within a metric result.

    Each signal represents a distinct piece of structured insight extracted from a model’s
    output (often a Pydantic result object). This provides a clean and type-safe way for
    metrics to define how specific sub-values are derived, displayed, and scored.

    The most visible signal(s) can be flagged with `headline_display=True`, allowing UIs
    or reports to surface them as key summary indicators (e.g., the primary "relevance" score
    for a statement or the overall metric outcome).

    Attributes:
        name: Unique name of the signal (e.g., `"overall_score"` or `"is_relevant"`).
        extractor: Callable that takes the metric’s result model and extracts the signal’s value.
        description: Optional human-readable explanation of what this signal represents.
        score_mapping: Optional dictionary mapping categorical values to numerical scores
                       (e.g., `{ "correct": 1.0, "incorrect": 0.0 }`).
        group: Optional label for logically grouping related signals (e.g., per-statement or per-turn).
        headline_display: If True, marks this signal as a headline or key display value for dashboards or summaries.
    """

    name: str
    extractor: Callable[[T], Any]
    description: Optional[str] = None
    score_mapping: Optional[Dict[str, float]] = None
    group: Optional[str] = None
    headline_display: bool = False

    def __post_init__(self):
        """Auto-generate a default description if one is not provided."""
        if not self.description:
            self.description = f'Signal: {self.name}'


class SubMetricResult(RichBaseModel):
    """A sub-metric extracted from a multi-metric evaluation.

    Used by metrics with `is_multi_metric=True` to define how a single
    evaluation result explodes into multiple sub-metric scores.
    """

    name: str = Field(
        ...,
        description="Sub-metric name (e.g., 'engagement', 'frustration')",
    )

    score: Optional[float] = Field(
        default=None,
        description='The computed score for this sub-metric.',
    )

    explanation: Optional[str] = Field(
        default=None,
        description='Explanation of how this sub-metric score was derived.',
    )

    threshold: Optional[float] = Field(
        default=None,
        description='Optional threshold for this specific sub-metric. If not set, inherits from parent metric.',
    )

    metric_category: Optional[MetricCategory] = Field(
        default=None,
        description=(
            'Category for this sub-metric: SCORE (numeric pass/fail), ANALYSIS (qualitative insights), '
            'or CLASSIFICATION (labels). If not set, defaults to SCORE when score is present, '
            'otherwise inherits from parent metric.'
        ),
    )

    group: Optional[str] = Field(
        default=None,
        description="Logical grouping for related sub-metrics (e.g., 'sentiment', 'behavioral').",
    )

    signals: Optional[Any] = Field(
        default=None,
        description='Optional signals for this sub-metric. If set, propagated to the MetricScore signals column.',
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description='Additional structured data specific to this sub-metric.',
    )
