import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, Field, field_serializer

from axion._core.schema import RichBaseModel

T = TypeVar('T', bound=BaseModel)

# Default explanation value used in MetricEvaluationResult
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

    default_threshold: float = Field(
        default=0.5,
        description='Default threshold used to determine if a score is considered a passing score.',
    )

    score_range: Tuple[Union[int, float], Union[int, float]] = Field(
        default=(0, 1),
        description='Tuple representing the valid score range for this metric (e.g., 0 to 1).',
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


class MetricEvaluationResult(RichBaseModel):
    """
    Standardized output model for the result of a single evaluation metric.

    This model holds the final score, an explanation of how the score was derived,
    and any additional metadata useful for debugging or traceability.
    """

    score: Union[int, float] = Field(
        ...,
        description="The final score assigned by the metric. Must be within the metric's defined score range.",
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
