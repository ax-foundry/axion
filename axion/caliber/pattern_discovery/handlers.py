from typing import List, Optional, Type

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from axion._core.schema import LLMRunnable, StrictBaseModel
from axion._handlers.llm.handler import LLMHandler
from axion.llm_registry import LLMRegistry


class AnnotationNote(StrictBaseModel):
    """Single annotation note for clustering input."""

    record_id: str = Field(description='Unique identifier for the record')
    notes: str = Field(description='The annotation notes from the evaluator')


class ClusteringInput(StrictBaseModel):
    """Input model for pattern clustering."""

    annotations: List[AnnotationNote] = Field(
        description='List of annotation notes to cluster'
    )


class PatternCategory(StrictBaseModel):
    """A single discovered pattern category."""

    category: str = Field(
        description='Short descriptive name for this pattern (2-4 words)'
    )
    record_ids: List[str] = Field(description='Record IDs that belong to this category')
    description: str = Field(
        description='Brief description of what characterizes this category'
    )


class ClusteringOutput(StrictBaseModel):
    """Structured output from pattern clustering LLM."""

    patterns: List[PatternCategory] = Field(
        description='List of discovered pattern categories'
    )
    uncategorized: List[str] = Field(
        default_factory=list,
        description='Record IDs that do not fit any category',
    )


class LabelInput(StrictBaseModel):
    """Input for label refinement."""

    examples: List[str] = Field(description='Example notes from the cluster')
    keywords: str = Field(description='Statistical keywords from BERTopic')


class LabelOutput(BaseModel):
    """Output from label refinement."""

    model_config = ConfigDict(populate_by_name=True, extra='forbid')

    category_name: str = Field(
        description='Concise 2-4 word category name',
        validation_alias=AliasChoices('category_name', 'category', 'name', 'label'),
        serialization_alias='category_name',
    )


DEFAULT_CLUSTERING_INSTRUCTION = """You are an expert at analyzing human evaluation annotations and discovering patterns.

Analyze the provided annotation notes from human evaluators and group them into 3-6 meaningful categories based on common themes, issues, or patterns.

For each category:
1. Provide a short, descriptive name (2-4 words)
2. List all record IDs that belong to this category
3. Write a brief description of what characterizes this category

Guidelines:
- Look for semantic similarity in the notes, not just keyword matches
- Categories should be mutually exclusive where possible
- If a note could fit multiple categories, assign it to the most specific one
- Notes that don't fit any clear pattern should be marked as uncategorized
"""

DEFAULT_EVIDENCE_CLUSTERING_INSTRUCTION = """You are an expert at analyzing text evidence and discovering patterns.

Analyze the provided items and group them into 3-6 meaningful categories based on common themes, issues, or patterns.

For each category:
1. Provide a short, descriptive name (2-4 words)
2. List all record IDs that belong to this category
3. Write a brief description of what characterizes this category

Guidelines:
- Cluster primarily by text meaning; use metadata hints as secondary signals
- Categories should be mutually exclusive where possible
- If an item could fit multiple categories, assign it to the most specific one
- Items that don't fit any clear pattern should be marked as uncategorized
"""

DEFAULT_DISTILLATION_INSTRUCTION_TEMPLATE = """You are an expert at synthesizing clusters of related evidence into actionable learnings.

Given a cluster of related items, produce one or more learning artifacts that capture the key insight.

For each learning:
1. Title: A concise 2-8 word actionable title
2. Content: A synthesized insight in prose form
3. Tags: Categorical tags for organization
4. Confidence: 0.0-1.0 reflecting how well-supported the learning is
5. Supporting item IDs: ONLY cite item_ids from the provided cluster's item_ids list
6. Recommended actions: Actionable bullet points (at least one for high-confidence learnings)
7. Counterexamples: Item IDs that contradict the main pattern (if any, from this cluster only)
8. Scope: When this learning applies
9. When not to apply: Limitations of the insight

IMPORTANT:
- Only cite item_ids from the provided cluster's item_ids list
- Include at least one recommended_action for high-confidence (>= 0.7) learnings
- Produce at most {max_learnings_per_cluster} learnings per cluster
"""


def default_distillation_instruction(max_learnings_per_cluster: int = 3) -> str:
    if max_learnings_per_cluster < 1:
        raise ValueError('max_learnings_per_cluster must be >= 1')
    return DEFAULT_DISTILLATION_INSTRUCTION_TEMPLATE.format(
        max_learnings_per_cluster=max_learnings_per_cluster
    )


DEFAULT_DISTILLATION_INSTRUCTION = default_distillation_instruction(3)


def _set_handler_llm(
    handler,
    llm: Optional[LLMRunnable],
    model_name: Optional[str],
    llm_provider: Optional[str],
) -> None:
    """Set up LLM on a handler instance."""
    if llm is not None:
        handler.llm = llm
    else:
        registry = LLMRegistry(llm_provider)
        handler.llm = registry.get_llm(model_name)

    handler.model_name = model_name or getattr(handler.llm, 'model', None)
    handler.llm_provider = llm_provider or getattr(handler.llm, '_provider', None)


class PatternClusteringHandler(LLMHandler[ClusteringInput, ClusteringOutput]):
    """LLM handler for clustering annotation notes into patterns."""

    input_model: Type[ClusteringInput] = ClusteringInput
    output_model: Type[ClusteringOutput] = ClusteringOutput
    instruction: str = DEFAULT_CLUSTERING_INSTRUCTION
    generation_fake_sample: bool = False

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        llm_provider: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        if instruction:
            self.instruction = instruction
        _set_handler_llm(self, llm, model_name, llm_provider)
        super().__init__(**kwargs)


class LabelRefinementHandler(LLMHandler[LabelInput, LabelOutput]):
    """LLM handler for refining BERTopic labels."""

    input_model: Type[LabelInput] = LabelInput
    output_model: Type[LabelOutput] = LabelOutput
    instruction: str = (
        'Generate a concise 2-4 word category name that captures the '
        'common theme of these annotation notes.\n\n'
        'Based on the example notes and statistical keywords, create a '
        'human-readable category name.\n'
        'The name should be descriptive and capture the essence of what '
        'these annotations have in common.\n\n'
        'Return ONLY a JSON object with this exact shape:\n'
        '{"category_name": "<2-4 word name>"}'
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        llm_provider: Optional[str] = None,
        **kwargs,
    ):
        _set_handler_llm(self, llm, model_name, llm_provider)
        super().__init__(**kwargs)


class EvidenceNote(StrictBaseModel):
    """Single evidence note for clustering input."""

    item_id: str
    text: str
    context: Optional[str] = None


class EvidenceClusteringInput(StrictBaseModel):
    """Input model for evidence clustering."""

    items: List[EvidenceNote]


class ClusterForDistillation(StrictBaseModel):
    """A cluster ready for distillation into learnings."""

    category: str
    description: str
    item_ids: List[str]
    example_texts: List[str]
    metadata_summary: Optional[str] = None


class DistillationInput(StrictBaseModel):
    """Input for distilling a single cluster into learnings."""

    cluster: ClusterForDistillation
    domain_context: Optional[str] = None


class LearningArtifactOutput(StrictBaseModel):
    """Pydantic output model for a single learning artifact from LLM."""

    title: str
    content: str
    tags: List[str]
    confidence: float
    supporting_item_ids: List[str]
    recommended_actions: List[str] = Field(default_factory=list)
    counterexamples: List[str] = Field(default_factory=list)
    scope: Optional[str] = None
    when_not_to_apply: Optional[str] = None


class DistillationOutput(StrictBaseModel):
    """Structured output from distillation LLM."""

    learnings: List[LearningArtifactOutput]


class EvidenceClusteringHandler(LLMHandler[EvidenceClusteringInput, ClusteringOutput]):
    """LLM handler for clustering evidence items into patterns.

    Reuses ``ClusteringOutput`` (uses ``record_ids`` for item IDs).
    """

    input_model: Type[EvidenceClusteringInput] = EvidenceClusteringInput
    output_model: Type[ClusteringOutput] = ClusteringOutput
    instruction: str = DEFAULT_EVIDENCE_CLUSTERING_INSTRUCTION
    generation_fake_sample: bool = False

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        llm_provider: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        if instruction:
            self.instruction = instruction
        _set_handler_llm(self, llm, model_name, llm_provider)
        super().__init__(**kwargs)


class DistillationHandler(LLMHandler[DistillationInput, DistillationOutput]):
    """LLM handler for distilling a cluster into learning artifacts."""

    input_model: Type[DistillationInput] = DistillationInput
    output_model: Type[DistillationOutput] = DistillationOutput
    instruction: str = DEFAULT_DISTILLATION_INSTRUCTION
    generation_fake_sample: bool = False

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        llm_provider: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        if instruction:
            self.instruction = instruction
        _set_handler_llm(self, llm, model_name, llm_provider)
        super().__init__(**kwargs)
