from typing import Any, Dict, List, Optional, Sequence, Union

from axion._core.logging import get_logger
from axion._core.tracing import trace
from axion.caliber.pattern_discovery._compat import (
    AnnotatedItem,
    annotations_to_evidence,
    normalize_annotations,
)
from axion.caliber.pattern_discovery._utils import (
    ExcerptFn,
    MetadataConfig,
    deterministic_sample,
    format_metadata_header,
)
from axion.caliber.pattern_discovery.handlers import (
    ClusteringOutput,
    EvidenceClusteringHandler,
    EvidenceClusteringInput,
    EvidenceNote,
    LabelInput,
    LabelOutput,
    LabelRefinementHandler,
)
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    EvidenceItem,
    PatternDiscoveryResult,
)

logger = get_logger(__name__)


class PatternDiscovery:
    """
    Discovers patterns in evaluation annotations using LLM-based clustering.

    This class leverages LLMHandler for structured output, automatic retries,
    and consistent LLM configuration with the rest of axion.

    Supports both the legacy ``discover()`` API (AnnotatedItem dicts) and the
    new ``discover_from_evidence()`` API (EvidenceItem sequences/dicts).

    Example:
        >>> from axion.caliber import PatternDiscovery, AnnotatedItem
        >>>
        >>> annotations = {
        ...     'rec_1': AnnotatedItem(record_id='rec_1', score=0, notes='Missing context'),
        ...     'rec_2': AnnotatedItem(record_id='rec_2', score=0, notes='Lacks detail'),
        ... }
        >>> discovery = PatternDiscovery(model_name='gpt-4o', llm_provider='openai')
        >>> result = await discovery.discover(annotations)
    """

    BERTOPIC_MIN_DOCUMENTS: int = 5
    BERTTOPIC_MIN_DOCUMENTS: int = BERTOPIC_MIN_DOCUMENTS

    MAX_EXAMPLES_PER_PATTERN: int = 3
    EXAMPLE_PREVIEW_CHARS: int = 100
    TOPIC_NAME_WORDS: int = 3
    TOPIC_DESCRIPTION_WORDS: int = 5

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm=None,
        llm_provider: Optional[str] = None,
        instruction: Optional[str] = None,
        max_notes: int = 50,
        min_category_size: int = 2,
        bertopic_embedding_model: Any = 'all-MiniLM-L6-v2',
        metadata_config: Optional[MetadataConfig] = None,
        excerpt_fn: Optional[ExcerptFn] = None,
        seed: Optional[int] = None,
    ):
        self._model_name = model_name
        self._llm = llm
        self._llm_provider = llm_provider
        self._instruction = instruction
        self._max_notes = max_notes
        self._min_category_size = min_category_size
        self._bertopic_embedding_model = bertopic_embedding_model
        self._metadata_config = metadata_config or MetadataConfig()
        self._excerpt_fn = excerpt_fn
        self._seed = seed

        # Lazily initialized handlers
        self._evidence_clustering_handler: Optional[EvidenceClusteringHandler] = None
        self._label_handler: Optional[LabelRefinementHandler] = None

    def _get_evidence_clustering_handler(self) -> EvidenceClusteringHandler:
        if self._evidence_clustering_handler is None:
            self._evidence_clustering_handler = EvidenceClusteringHandler(
                model_name=self._model_name,
                llm=self._llm,
                llm_provider=self._llm_provider,
                instruction=self._instruction,
            )
        return self._evidence_clustering_handler

    def _get_label_handler(self) -> LabelRefinementHandler:
        if self._label_handler is None:
            self._label_handler = LabelRefinementHandler(
                model_name=self._model_name,
                llm=self._llm,
                llm_provider=self._llm_provider,
            )
        return self._label_handler

    @trace(name='PatternDiscovery.discover', capture_args=True)
    async def discover(
        self,
        annotations: Union[Dict[str, AnnotatedItem], Dict[str, Dict]],
        method: ClusteringMethod = ClusteringMethod.LLM,
    ) -> PatternDiscoveryResult:
        """
        Backward-compatible entry point.

        Normalizes AnnotatedItem dicts into EvidenceItem dicts and
        delegates to ``discover_from_evidence()``.
        """
        items = self._normalize_annotations(annotations)
        evidence = annotations_to_evidence(items)
        return await self._discover_internal(evidence, method)

    def _normalize_annotations(
        self, annotations: Union[Dict[str, AnnotatedItem], Dict[str, Dict]]
    ) -> Dict[str, AnnotatedItem]:
        return normalize_annotations(annotations)

    @trace(name='PatternDiscovery.discover_from_evidence', capture_args=True)
    async def discover_from_evidence(
        self,
        evidence: Union[Sequence[EvidenceItem], Dict[str, EvidenceItem]],
        method: ClusteringMethod = ClusteringMethod.LLM,
    ) -> PatternDiscoveryResult:
        """Discover patterns from generic evidence items."""
        evidence_dict = self._to_evidence_dict(evidence)
        return await self._discover_internal(evidence_dict, method)

    @staticmethod
    def _to_evidence_dict(
        evidence: Union[Sequence[EvidenceItem], Dict[str, EvidenceItem]],
    ) -> Dict[str, EvidenceItem]:
        if isinstance(evidence, dict):
            for key, item in evidence.items():
                if key != item.id:
                    raise ValueError(
                        f'Dict key {key!r} does not match item.id {item.id!r}'
                    )
            return evidence
        return {item.id: item for item in evidence}

    async def _discover_internal(
        self,
        evidence: Dict[str, EvidenceItem],
        method: ClusteringMethod,
    ) -> PatternDiscoveryResult:
        if method == ClusteringMethod.LLM:
            return await self._cluster_evidence_with_llm(evidence)
        elif method == ClusteringMethod.BERTOPIC:
            return await self._cluster_evidence_with_bertopic(evidence)
        elif method == ClusteringMethod.HYBRID:
            return await self._cluster_evidence_hybrid(evidence)
        else:
            raise ValueError(f'Unknown clustering method: {method}')

    @trace(name='PatternDiscovery._cluster_evidence_with_llm')
    async def _cluster_evidence_with_llm(
        self, evidence: Dict[str, EvidenceItem]
    ) -> PatternDiscoveryResult:
        items_with_text = {eid: item for eid, item in evidence.items() if item.text}

        if not items_with_text:
            return PatternDiscoveryResult(
                patterns=[],
                uncategorized=list(evidence.keys()),
                total_analyzed=len(evidence),
                method=ClusteringMethod.LLM,
                metadata={'error': 'No items with text to cluster'},
            )

        if len(items_with_text) > self._max_notes:
            items_with_text = deterministic_sample(
                items_with_text, self._max_notes, self._seed
            )

        meta_cfg = self._metadata_config
        notes: List[EvidenceNote] = []
        for eid, item in items_with_text.items():
            context = None
            if meta_cfg.include_in_clustering and item.metadata:
                context = format_metadata_header(item.metadata, meta_cfg)
            notes.append(EvidenceNote(item_id=eid, text=item.text, context=context))

        input_data = EvidenceClusteringInput(items=notes)
        handler = self._get_evidence_clustering_handler()
        output: ClusteringOutput = await handler.execute(input_data)

        patterns = self._output_to_patterns(output, items_with_text)

        return PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=output.uncategorized,
            total_analyzed=len(evidence),
            method=ClusteringMethod.LLM,
            metadata={
                'model': handler.model_name,
                'provider': handler.llm_provider,
            },
        )

    def _output_to_patterns(
        self,
        output: ClusteringOutput,
        items: Dict[str, EvidenceItem],
    ) -> List[DiscoveredPattern]:
        patterns = []
        for p in output.patterns:
            if len(p.record_ids) >= self._min_category_size:
                examples = []
                for rid in p.record_ids[: self.MAX_EXAMPLES_PER_PATTERN]:
                    item = items.get(rid)
                    if item and item.text:
                        examples.append(item.text[: self.EXAMPLE_PREVIEW_CHARS])
                patterns.append(
                    DiscoveredPattern(
                        category=p.category,
                        description=p.description,
                        count=len(p.record_ids),
                        record_ids=p.record_ids,
                        examples=examples,
                    )
                )
        patterns.sort(key=lambda p: p.count, reverse=True)
        return patterns

    @trace(name='PatternDiscovery._cluster_evidence_with_bertopic')
    async def _cluster_evidence_with_bertopic(
        self, evidence: Dict[str, EvidenceItem]
    ) -> PatternDiscoveryResult:
        try:
            from bertopic import BERTopic
            from bertopic.representation import KeyBERTInspired
        except ImportError:
            raise ImportError(
                'BERTopic not installed. Run: pip install axion[bertopic]'
            )

        items_with_text = {eid: item for eid, item in evidence.items() if item.text}
        record_ids = list(items_with_text.keys())
        texts = [items_with_text[rid].text for rid in record_ids]

        if len(texts) < self.BERTOPIC_MIN_DOCUMENTS:
            return PatternDiscoveryResult(
                patterns=[],
                uncategorized=record_ids,
                total_analyzed=len(evidence),
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

        topics, probs = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()

        patterns: List[DiscoveredPattern] = []
        uncategorized: List[str] = []

        for idx, row in topic_info.iterrows():
            topic_id = row['Topic']

            if topic_id == -1:
                topic_ids_list = [
                    record_ids[i] for i, t in enumerate(topics) if t == -1
                ]
                uncategorized.extend(topic_ids_list)
                continue

            topic_ids_list = [
                record_ids[i] for i, t in enumerate(topics) if t == topic_id
            ]

            if not topic_ids_list:
                continue

            topic_words = topic_model.get_topic(topic_id)
            category_name = self._format_topic_name(topic_words)

            examples = []
            for rid in topic_ids_list[: self.MAX_EXAMPLES_PER_PATTERN]:
                text = items_with_text[rid].text
                if text:
                    examples.append(text[: self.EXAMPLE_PREVIEW_CHARS])

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
                    count=len(topic_ids_list),
                    record_ids=topic_ids_list,
                    examples=examples,
                    confidence=avg_confidence,
                )
            )

        patterns.sort(key=lambda p: p.count, reverse=True)

        return PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=uncategorized,
            total_analyzed=len(evidence),
            method=ClusteringMethod.BERTOPIC,
            metadata={'num_topics': len(patterns)},
        )

    def _format_topic_name(self, topic_words: List[tuple]) -> str:
        if not topic_words:
            return 'Unknown Pattern'
        top_words = [word.title() for word, _ in topic_words[: self.TOPIC_NAME_WORDS]]
        return ' / '.join(top_words)

    @trace(name='PatternDiscovery._cluster_evidence_hybrid')
    async def _cluster_evidence_hybrid(
        self, evidence: Dict[str, EvidenceItem]
    ) -> PatternDiscoveryResult:
        result = await self._cluster_evidence_with_bertopic(evidence)

        if not result.patterns:
            result.method = ClusteringMethod.HYBRID
            return result

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

        result.method = ClusteringMethod.HYBRID
        result.metadata['label_model'] = handler.model_name
        return result
