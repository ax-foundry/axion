"""
Pattern discovery for CaliberHQ workflow.

Discovers patterns in evaluation annotations using LLM-based and/or topic modeling approaches.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import AliasChoices, BaseModel, Field

from axion._core.logging import get_logger
from axion._core.schema import LLMRunnable
from axion._handlers.llm.handler import LLMHandler
from axion.llm_registry import LLMRegistry

logger = get_logger(__name__)


class ClusteringMethod(str, Enum):
    """Available clustering methods."""

    LLM = 'llm'  # LLM-based semantic clustering
    BERTOPIC = 'bertopic'  # BERTopic topic modeling (no LLM needed)
    HYBRID = 'hybrid'  # BERTopic clustering + LLM label refinement


class AnnotationNote(BaseModel):
    """Single annotation note for clustering input."""

    record_id: str = Field(description='Unique identifier for the record')
    notes: str = Field(description='The annotation notes from the evaluator')


class ClusteringInput(BaseModel):
    """Input model for pattern clustering."""

    annotations: List[AnnotationNote] = Field(
        description='List of annotation notes to cluster'
    )


class PatternCategory(BaseModel):
    """A single discovered pattern category."""

    category: str = Field(
        description='Short descriptive name for this pattern (2-4 words)'
    )
    record_ids: List[str] = Field(description='Record IDs that belong to this category')
    description: str = Field(
        description='Brief description of what characterizes this category'
    )


class ClusteringOutput(BaseModel):
    """Structured output from pattern clustering LLM."""

    patterns: List[PatternCategory] = Field(
        description='List of discovered pattern categories'
    )
    uncategorized: List[str] = Field(
        default_factory=list, description='Record IDs that do not fit any category'
    )


@dataclass
class AnnotatedItem:
    """A single annotated item with optional notes."""

    record_id: str
    score: int  # Binary: 0 or 1
    notes: Optional[str] = None
    timestamp: Optional[str] = None
    query: Optional[str] = None
    actual_output: Optional[str] = None


@dataclass
class DiscoveredPattern:
    """A discovered pattern/category from clustering."""

    category: str
    description: str
    count: int
    record_ids: List[str]
    examples: List[str] = field(default_factory=list)
    confidence: Optional[float] = None


@dataclass
class PatternDiscoveryResult:
    """Complete result from pattern discovery."""

    patterns: List[DiscoveredPattern]
    uncategorized: List[str]
    total_analyzed: int
    method: ClusteringMethod
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        """
        Initialize the pattern clustering handler.

        Args:
            model_name: Name of the LLM model (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
            llm: Pre-configured LLM instance
            llm_provider: LLM provider ('openai', 'anthropic')
            instruction: Custom instruction to override default
        """
        if instruction:
            self.instruction = instruction

        # Set up LLM using same pattern as BaseMetric
        self._set_llm(llm, model_name, llm_provider)

        super().__init__(**kwargs)

    def _set_llm(
        self,
        llm: Optional[LLMRunnable],
        model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> None:
        """Set the LLM instance from provided params or registry."""
        if llm is not None:
            self.llm = llm
        elif model_name is not None or llm_provider is not None:
            registry = LLMRegistry(llm_provider)
            self.llm = registry.get_llm(model_name)
        else:
            # Default to registry default
            registry = LLMRegistry(llm_provider)
            self.llm = registry.get_llm(model_name)

        self.model_name = model_name or getattr(self.llm, 'model', None)
        self.llm_provider = llm_provider or getattr(self.llm, '_provider', None)


class LabelInput(BaseModel):
    """Input for label refinement."""

    examples: List[str] = Field(description='Example notes from the cluster')
    keywords: str = Field(description='Statistical keywords from BERTopic')


class LabelOutput(BaseModel):
    """Output from label refinement."""

    # Note: in parser-fallback mode (or with weaker models), the LLM sometimes returns
    # keys like "category" or "name" instead of the exact "category_name". We accept
    # those on input to make HYBRID mode resilient, while still serializing as
    # "category_name" to match the intended schema.
    category_name: str = Field(
        description='Concise 2-4 word category name',
        validation_alias=AliasChoices('category_name', 'category', 'name', 'label'),
        serialization_alias='category_name',
    )


class LabelRefinementHandler(LLMHandler[LabelInput, LabelOutput]):
    """LLM handler for refining BERTopic labels."""

    input_model: Type[LabelInput] = LabelInput
    output_model: Type[LabelOutput] = LabelOutput
    instruction: str = """Generate a concise 2-4 word category name that captures the common theme of these annotation notes.

Based on the example notes and statistical keywords, create a human-readable category name.
The name should be descriptive and capture the essence of what these annotations have in common.

Return ONLY a JSON object with this exact shape:
{"category_name": "<2-4 word name>"}"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        llm_provider: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the label refinement handler."""
        # Set up LLM
        if llm is not None:
            self.llm = llm
        elif model_name is not None or llm_provider is not None:
            registry = LLMRegistry(llm_provider)
            self.llm = registry.get_llm(model_name)
        else:
            registry = LLMRegistry(llm_provider)
            self.llm = registry.get_llm(model_name)

        self.model_name = model_name or getattr(self.llm, 'model', None)
        self.llm_provider = llm_provider or getattr(self.llm, '_provider', None)

        super().__init__(**kwargs)


class PatternDiscovery:
    """
    Discovers patterns in evaluation annotations using LLM-based clustering.

    This class leverages LLMHandler for structured output, automatic retries,
    and consistent LLM configuration with the rest of axion.

    Example:
        >>> from axion.caliber import PatternDiscovery, AnnotatedItem
        >>>
        >>> annotations = {
        ...     'rec_1': AnnotatedItem(record_id='rec_1', score=0, notes='Missing context'),
        ...     'rec_2': AnnotatedItem(record_id='rec_2', score=0, notes='Lacks detail'),
        ... }
        >>> # Using model_name/provider (recommended)
        >>> discovery = PatternDiscovery(model_name='gpt-4o', llm_provider='openai')
        >>> result = await discovery.discover(annotations)
        >>>
        >>> # Or with pre-configured LLM
        >>> from axion.llm_registry import LLMRegistry
        >>> llm = LLMRegistry('anthropic').get_llm('claude-sonnet-4-20250514')
        >>> discovery = PatternDiscovery(llm=llm)
        >>> result = await discovery.discover(annotations)
    """

    # -------------------------------------------------------------------------
    # Defaults / constants (avoid magic numbers throughout implementation)
    # -------------------------------------------------------------------------
    BERTOPIC_MIN_DOCUMENTS: int = 5
    # Backwards-compatible alias for an old typo (internal, but safe).
    BERTTOPIC_MIN_DOCUMENTS: int = BERTOPIC_MIN_DOCUMENTS

    MAX_EXAMPLES_PER_PATTERN: int = 3
    EXAMPLE_PREVIEW_CHARS: int = 100
    TOPIC_NAME_WORDS: int = 3
    TOPIC_DESCRIPTION_WORDS: int = 5

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        llm_provider: Optional[str] = None,
        instruction: Optional[str] = None,
        max_notes: int = 50,
        min_category_size: int = 2,
        bertopic_embedding_model: Any = 'all-MiniLM-L6-v2',
    ):
        """
        Initialize PatternDiscovery.

        Args:
            model_name: Name of the LLM model (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
            llm: Pre-configured LLM instance
            llm_provider: LLM provider ('openai', 'anthropic')
            instruction: Custom instruction to override default clustering prompt
            max_notes: Max notes per clustering call (sampled if exceeded)
            min_category_size: Min records to form a category
            bertopic_embedding_model: BERTopic embedding model (string or model object)
        """
        self._model_name = model_name
        self._llm = llm
        self._llm_provider = llm_provider
        self._instruction = instruction
        self._max_notes = max_notes
        self._min_category_size = min_category_size
        self._bertopic_embedding_model = bertopic_embedding_model

        # Lazily initialized handlers
        self._clustering_handler: Optional[PatternClusteringHandler] = None
        self._label_handler: Optional[LabelRefinementHandler] = None

    def _get_clustering_handler(self) -> PatternClusteringHandler:
        """Get or create the clustering handler."""
        if self._clustering_handler is None:
            self._clustering_handler = PatternClusteringHandler(
                model_name=self._model_name,
                llm=self._llm,
                llm_provider=self._llm_provider,
                instruction=self._instruction,
            )
        return self._clustering_handler

    def _get_label_handler(self) -> LabelRefinementHandler:
        """Get or create the label refinement handler."""
        if self._label_handler is None:
            self._label_handler = LabelRefinementHandler(
                model_name=self._model_name,
                llm=self._llm,
                llm_provider=self._llm_provider,
            )
        return self._label_handler

    async def discover(
        self,
        annotations: Union[Dict[str, AnnotatedItem], Dict[str, Dict]],
        method: ClusteringMethod = ClusteringMethod.LLM,
    ) -> PatternDiscoveryResult:
        """
        Discover patterns asynchronously.

        Args:
            annotations: Dict mapping record_id to AnnotatedItem or dict
            method: Clustering method to use

        Returns:
            PatternDiscoveryResult with discovered patterns
        """
        items = self._normalize_annotations(annotations)

        if method == ClusteringMethod.LLM:
            return await self._cluster_with_llm(items)
        elif method == ClusteringMethod.BERTOPIC:
            return await self._cluster_with_bertopic(items)
        elif method == ClusteringMethod.HYBRID:
            return await self._cluster_hybrid(items)
        else:
            raise ValueError(f'Unknown clustering method: {method}')

    def _normalize_annotations(
        self, annotations: Union[Dict[str, AnnotatedItem], Dict[str, Dict]]
    ) -> Dict[str, AnnotatedItem]:
        """Convert dict annotations to AnnotatedItem instances."""
        result = {}
        for record_id, item in annotations.items():
            if isinstance(item, AnnotatedItem):
                result[record_id] = item
            elif isinstance(item, dict):
                result[record_id] = AnnotatedItem(
                    record_id=item.get('record_id', record_id),
                    score=item.get('score', 0),
                    notes=item.get('notes'),
                    timestamp=item.get('timestamp'),
                    query=item.get('query'),
                    actual_output=item.get('actual_output'),
                )
            else:
                raise TypeError(f'Invalid annotation type: {type(item)}')
        return result

    async def _cluster_with_llm(
        self, items: Dict[str, AnnotatedItem]
    ) -> PatternDiscoveryResult:
        """Cluster annotations using LLM-based semantic analysis with structured output."""
        # Filter items with notes
        items_with_notes = {rid: item for rid, item in items.items() if item.notes}

        if not items_with_notes:
            return PatternDiscoveryResult(
                patterns=[],
                uncategorized=list(items.keys()),
                total_analyzed=len(items),
                method=ClusteringMethod.LLM,
                metadata={'error': 'No items with notes to cluster'},
            )

        # Sample if too many
        record_ids = list(items_with_notes.keys())
        if len(record_ids) > self._max_notes:
            import random

            record_ids = random.sample(record_ids, self._max_notes)
            items_with_notes = {rid: items_with_notes[rid] for rid in record_ids}

        # Build structured input
        input_data = ClusteringInput(
            annotations=[
                AnnotationNote(record_id=rid, notes=items_with_notes[rid].notes)
                for rid in record_ids
            ]
        )

        # Execute with LLMHandler (structured output, retries, tracing)
        handler = self._get_clustering_handler()
        output: ClusteringOutput = await handler.execute(input_data)

        # Convert to result format
        patterns = []
        for p in output.patterns:
            if len(p.record_ids) >= self._min_category_size:
                patterns.append(
                    DiscoveredPattern(
                        category=p.category,
                        description=p.description,
                        count=len(p.record_ids),
                        record_ids=p.record_ids,
                        examples=[
                            items_with_notes[rid].notes[: self.EXAMPLE_PREVIEW_CHARS]
                            for rid in p.record_ids[: self.MAX_EXAMPLES_PER_PATTERN]
                            if rid in items_with_notes and items_with_notes[rid].notes
                        ],
                    )
                )

        # Sort by count
        patterns.sort(key=lambda p: p.count, reverse=True)

        return PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=output.uncategorized,
            total_analyzed=len(items),
            method=ClusteringMethod.LLM,
            metadata={
                'model': handler.model_name,
                'provider': handler.llm_provider,
            },
        )

    async def _cluster_with_bertopic(
        self, items: Dict[str, AnnotatedItem]
    ) -> PatternDiscoveryResult:
        """Cluster annotations using BERTopic topic modeling."""
        try:
            from bertopic import BERTopic
            from bertopic.representation import KeyBERTInspired
        except ImportError:
            raise ImportError(
                'BERTopic not installed. Run: pip install axion[bertopic]'
            )

        items_with_notes = {rid: item for rid, item in items.items() if item.notes}
        record_ids = list(items_with_notes.keys())
        notes = [items_with_notes[rid].notes for rid in record_ids]

        if len(notes) < self.BERTOPIC_MIN_DOCUMENTS:
            return PatternDiscoveryResult(
                patterns=[],
                uncategorized=record_ids,
                total_analyzed=len(items),
                method=ClusteringMethod.BERTOPIC,
                metadata={
                    'error': f'Too few documents for BERTopic (min {self.BERTOPIC_MIN_DOCUMENTS})'
                },
            )

        representation_model = KeyBERTInspired()
        topic_model = BERTopic(
            embedding_model=self._bertopic_embedding_model,
            representation_model=representation_model,
            nr_topics='auto',
            min_topic_size=self._min_category_size,
            verbose=False,
        )

        topics, probs = topic_model.fit_transform(notes)
        topic_info = topic_model.get_topic_info()

        patterns: List[DiscoveredPattern] = []
        uncategorized: List[str] = []

        for idx, row in topic_info.iterrows():
            topic_id = row['Topic']

            if topic_id == -1:
                topic_record_ids = [
                    record_ids[i] for i, t in enumerate(topics) if t == -1
                ]
                uncategorized.extend(topic_record_ids)
                continue

            topic_record_ids = [
                record_ids[i] for i, t in enumerate(topics) if t == topic_id
            ]

            if not topic_record_ids:
                continue

            topic_words = topic_model.get_topic(topic_id)
            category_name = self._format_topic_name(topic_words)

            examples = []
            for rid in topic_record_ids[: self.MAX_EXAMPLES_PER_PATTERN]:
                note = items_with_notes[rid].notes
                if note:
                    examples.append(note[: self.EXAMPLE_PREVIEW_CHARS])

            topic_probs = [probs[i] for i, t in enumerate(topics) if t == topic_id]
            avg_confidence = (
                sum(topic_probs) / len(topic_probs) if topic_probs else None
            )

            patterns.append(
                DiscoveredPattern(
                    category=category_name,
                    description=(
                        'Topics: '
                        + ', '.join(
                            [w for w, _ in topic_words[: self.TOPIC_DESCRIPTION_WORDS]]
                        )
                    ),
                    count=len(topic_record_ids),
                    record_ids=topic_record_ids,
                    examples=examples,
                    confidence=avg_confidence,
                )
            )

        patterns.sort(key=lambda p: p.count, reverse=True)

        return PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=uncategorized,
            total_analyzed=len(items),
            method=ClusteringMethod.BERTOPIC,
            metadata={'num_topics': len(patterns)},
        )

    def _format_topic_name(self, topic_words: List[tuple]) -> str:
        """Format BERTopic words into a readable category name."""
        if not topic_words:
            return 'Unknown Pattern'
        top_words = [word.title() for word, _ in topic_words[: self.TOPIC_NAME_WORDS]]
        return ' / '.join(top_words)

    async def _cluster_hybrid(
        self, items: Dict[str, AnnotatedItem]
    ) -> PatternDiscoveryResult:
        """Use BERTopic for clustering, LLM for better category names."""
        result = await self._cluster_with_bertopic(items)

        if not result.patterns:
            result.method = ClusteringMethod.HYBRID
            return result

        # Use LLM to generate better category names via structured output
        handler = self._get_label_handler()

        for pattern in result.patterns:
            try:
                input_data = LabelInput(
                    examples=pattern.examples,
                    keywords=pattern.description,
                )
                output: LabelOutput = await handler.execute(input_data)
                pattern.category = output.category_name
            except Exception as e:
                logger.warning(f'Failed to refine label for pattern: {e}')
                # Keep original BERTopic name on error

        result.method = ClusteringMethod.HYBRID
        result.metadata['label_model'] = handler.model_name
        return result
