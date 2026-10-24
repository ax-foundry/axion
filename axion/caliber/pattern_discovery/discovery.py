from typing import Any, Dict, List, Optional, Sequence, Set, Union

from axion._core.logging import get_logger
from axion._core.tracing import init_tracer, trace
from axion.caliber.pattern_discovery._compat import (
    AnnotatedItem,
    annotations_to_evidence,
    normalize_annotations,
)
from axion.caliber.pattern_discovery._utils import (
    ExcerptFn,
    MetadataConfig,
    build_coverage,
    deterministic_sample,
    format_metadata_header,
    normalize_evidence,
    repair_cluster_assignments,
)
from axion.caliber.pattern_discovery.handlers import (
    DEFAULT_EVIDENCE_CLUSTERING_INSTRUCTION,
    ClusteringOutput,
    EvidenceClusteringHandler,
    EvidenceClusteringInput,
    EvidenceNote,
    LabelInput,
    LabelOutput,
    LabelRefinementHandler,
    default_evidence_clustering_instruction,
)
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    EvidenceItem,
    PatternDiscoveryResult,
)

logger = get_logger(__name__)


class LowCoverageError(RuntimeError):
    """Raised when the model placed too few submitted items and the caller opted in."""

    def __init__(self, message: str, coverage: Dict[str, Any]) -> None:
        super().__init__(message)
        self.coverage = coverage


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
        >>> discovery = PatternDiscovery(model_name='gpt-6-sol', llm_provider='openai')
        >>> result = await discovery.discover(annotations)
    """

    BERTOPIC_MIN_DOCUMENTS: int = 5
    BERTTOPIC_MIN_DOCUMENTS: int = BERTOPIC_MIN_DOCUMENTS

    # Mirrors BERTopic's built-in UMAP defaults (bertopic._bertopic constructs
    # UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')). UMAP's k-NN
    # graph requires n_samples > n_neighbors, so any corpus with n_docs <= n_neighbors
    # collapses to a zero-row array and fit_transform raises "Found array with
    # 0 sample(s)". The small-corpus boundary is therefore exactly the *effective*
    # n_neighbors (see ``bertopic_n_neighbors`` below) — not a magic constant — so
    # raising n_neighbors scales the guard automatically.
    BERTOPIC_DEFAULT_N_NEIGHBORS: int = 15
    BERTOPIC_DEFAULT_N_COMPONENTS: int = 5
    # Back-compat alias for callers that referenced the boundary directly; the boundary
    # equals the default n_neighbors.
    BERTOPIC_SMALL_CORPUS_THRESHOLD: int = BERTOPIC_DEFAULT_N_NEIGHBORS

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
        bertopic_n_neighbors: Optional[int] = None,
        small_corpus_llm_fallback: bool = True,
        metadata_config: Optional[MetadataConfig] = None,
        excerpt_fn: Optional[ExcerptFn] = None,
        seed: Optional[int] = None,
        tracer: Optional[Any] = None,
        # New params stay at the end so existing positional callers are unaffected.
        disjoint_clusters: bool = False,
        clustering_excerpt_chars: Optional[int] = None,
        min_coverage: Optional[float] = None,
        raise_on_low_coverage: bool = False,
    ):
        self._model_name = model_name
        self._llm = llm
        self._llm_provider = llm_provider
        self._instruction = instruction
        self._max_notes = max_notes
        self._min_category_size = min_category_size
        self._bertopic_embedding_model = bertopic_embedding_model
        # Effective UMAP n_neighbors; also the small-corpus boundary. Defaults to
        # BERTopic's own default so behavior is unchanged unless a caller overrides it.
        self._bertopic_n_neighbors = (
            bertopic_n_neighbors
            if bertopic_n_neighbors is not None
            else self.BERTOPIC_DEFAULT_N_NEIGHBORS
        )
        # For HYBRID on a corpus at/below the small-corpus boundary (n_neighbors),
        # cluster with the LLM instead of a clamped BERTopic run. Topic modeling on so
        # few docs is degenerate (near-zero-variance embeddings → matmul warnings), and
        # LLM grouping handles small sets better. Explicit BERTOPIC still clamps-and-runs.
        self._small_corpus_llm_fallback = small_corpus_llm_fallback
        self._metadata_config = metadata_config or MetadataConfig()
        self._excerpt_fn = excerpt_fn
        self._seed = seed
        # Cluster-output integrity. disjoint_clusters=True keeps an item's first
        # assignment only; the default lets overlapping themes share items.
        self._disjoint_clusters = disjoint_clusters
        # Per-note cap on the text sent to the clustering call. None = full text.
        self._clustering_excerpt_chars = clustering_excerpt_chars
        # Model-coverage floor (share of submitted items the model placed in a
        # pattern or in uncategorized). None = no check; below it, warn or raise.
        if min_coverage is not None and not 0.0 <= min_coverage <= 1.0:
            raise ValueError('min_coverage must be between 0.0 and 1.0')
        self._min_coverage = min_coverage
        self._raise_on_low_coverage = raise_on_low_coverage
        self.tracer = init_tracer('base', tracer=tracer)

        # Lazily initialized handlers. Clustering handlers are cached per rendered
        # instruction: with the default instruction the category range scales with
        # corpus size (see ``evidence_category_range``), so different corpus sizes can
        # need different handler instances within one PatternDiscovery lifetime.
        self._evidence_clustering_handlers: Dict[str, EvidenceClusteringHandler] = {}
        self._label_handler: Optional[LabelRefinementHandler] = None

    def _get_evidence_clustering_handler(
        self, n_items: Optional[int] = None
    ) -> EvidenceClusteringHandler:
        if self._instruction is not None:
            instruction = self._instruction  # caller-supplied: never rescaled
        elif n_items is not None:
            instruction = default_evidence_clustering_instruction(n_items)
        else:
            instruction = DEFAULT_EVIDENCE_CLUSTERING_INSTRUCTION
        handler = self._evidence_clustering_handlers.get(instruction)
        if handler is None:
            handler = EvidenceClusteringHandler(
                model_name=self._model_name,
                llm=self._llm,
                llm_provider=self._llm_provider,
                instruction=instruction,
                tracer=self.tracer,
            )
            self._evidence_clustering_handlers[instruction] = handler
        return handler

    def _get_label_handler(self) -> LabelRefinementHandler:
        if self._label_handler is None:
            self._label_handler = LabelRefinementHandler(
                model_name=self._model_name,
                llm=self._llm,
                llm_provider=self._llm_provider,
                tracer=self.tracer,
            )
        return self._label_handler

    @trace(name='discover', capture_args=True)
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

    @trace(name='discover_from_evidence', capture_args=True)
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
        evidence_dict, _ = normalize_evidence(evidence)
        return evidence_dict

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

    @trace(name='cluster_evidence_with_llm')
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
                metadata={
                    'error': 'No items with text to cluster',
                    'coverage': self._unclustered_coverage(evidence, items_with_text),
                },
            )

        ineligible = len(evidence) - len(items_with_text)
        sampled_out = 0
        if len(items_with_text) > self._max_notes:
            sampled_out = len(items_with_text) - self._max_notes
            items_with_text = deterministic_sample(
                items_with_text, self._max_notes, self._seed
            )

        meta_cfg = self._metadata_config
        cap = self._clustering_excerpt_chars
        notes: List[EvidenceNote] = []
        for eid, item in items_with_text.items():
            context = None
            if meta_cfg.include_in_clustering and item.metadata:
                context = format_metadata_header(item.metadata, meta_cfg)
            text = item.text if cap is None else item.text[:cap]
            notes.append(EvidenceNote(item_id=eid, text=text, context=context))

        input_data = EvidenceClusteringInput(items=notes)
        handler = self._get_evidence_clustering_handler(n_items=len(notes))
        output: ClusteringOutput = await handler.execute(input_data)

        patterns, uncategorized, metadata = self._output_to_patterns(
            output, items_with_text
        )
        metadata['coverage'] = build_coverage(
            submitted_ids=list(items_with_text.keys()),
            assigned_ids={rid for p in patterns for rid in p.record_ids},
            uncategorized_ids=uncategorized,
            filtered_small_cluster_ids=metadata.pop('_filtered_small_cluster_ids'),
            ineligible=ineligible,
            sampled_out=sampled_out,
        )
        self._check_coverage(metadata['coverage'])
        metadata['model'] = handler.model_name
        metadata['provider'] = handler.llm_provider

        return PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=uncategorized,
            total_analyzed=len(evidence),
            method=ClusteringMethod.LLM,
            metadata=metadata,
        )

    @staticmethod
    def _unclustered_coverage(
        evidence: Dict[str, EvidenceItem], items_with_text: Dict[str, EvidenceItem]
    ) -> Dict[str, Any]:
        """Coverage for a result that never reached a model.

        Everything with text is reported as ``uncategorized`` (which is where
        the early-return result puts it), the rest as ``ineligible``. Nothing
        was omitted by a model, so ``ratio`` is 1.0.
        """
        return build_coverage(
            submitted_ids=list(items_with_text),
            assigned_ids=set(),
            uncategorized_ids=list(items_with_text),
            filtered_small_cluster_ids=set(),
            ineligible=len(evidence) - len(items_with_text),
        )

    def _check_coverage(self, coverage: Dict[str, Any]) -> None:
        if self._min_coverage is None:
            return
        ratio = coverage['ratio']
        if ratio >= self._min_coverage:
            return
        coverage['below_min'] = True
        message = (
            f'Clustering placed {ratio:.0%} of {coverage["submitted"]} submitted '
            f'items (min_coverage={self._min_coverage:.0%}); '
            f'{coverage["omitted_by_model"]} returned in neither a pattern nor '
            'uncategorized.'
        )
        if self._raise_on_low_coverage:
            raise LowCoverageError(message, coverage)
        logger.warning(message)

    def _output_to_patterns(
        self,
        output: ClusteringOutput,
        items: Dict[str, EvidenceItem],
    ) -> tuple[List[DiscoveredPattern], List[str], Dict[str, Any]]:
        """Validate raw model output and build patterns.

        Returns ``(patterns, uncategorized, metadata)`` where metadata carries
        ``cluster_repairs`` and a private ``_filtered_small_cluster_ids`` set the
        caller folds into coverage accounting.
        """
        repaired_ids, uncategorized, repairs = repair_cluster_assignments(
            [list(p.record_ids) for p in output.patterns],
            list(output.uncategorized),
            known_ids=set(items.keys()),
            disjoint=self._disjoint_clusters,
        )
        if repairs.total:
            logger.warning('Repaired clustering output: %s', repairs.as_dict())

        patterns: List[DiscoveredPattern] = []
        filtered_small: Set[str] = set()
        for p, record_ids in zip(output.patterns, repaired_ids):
            if len(record_ids) < self._min_category_size:
                filtered_small.update(record_ids)
                continue
            examples = []
            for rid in record_ids[: self.MAX_EXAMPLES_PER_PATTERN]:
                item = items.get(rid)
                if item and item.text:
                    examples.append(item.text[: self.EXAMPLE_PREVIEW_CHARS])
            patterns.append(
                DiscoveredPattern(
                    category=p.category,
                    description=p.description,
                    count=len(record_ids),
                    record_ids=record_ids,
                    examples=examples,
                )
            )
        patterns.sort(key=lambda p: p.count, reverse=True)
        # An id dropped from a small cluster but kept in a surviving one is assigned.
        assigned = {rid for p in patterns for rid in p.record_ids}
        metadata: Dict[str, Any] = {
            'cluster_repairs': repairs.as_dict(),
            '_filtered_small_cluster_ids': filtered_small - assigned,
        }
        return patterns, uncategorized, metadata

    def _build_umap(self, n_docs: int) -> Any:
        """Return a seeded UMAP mirroring BERTopic's defaults, clamped to ``n_docs``.

        Always building the UMAP (rather than deferring to BERTopic's own on large
        corpora) exists for one reason: BERTopic constructs its default UMAP with
        ``random_state=None``, so cluster membership is stochastic run-to-run.
        Downstream recurrence thresholds then kill/birth whole learnings on boundary
        jitter. Seeding with ``self._seed`` (or 42 when unseeded) makes the reduction
        deterministic; every other parameter matches BERTopic's built-in default
        (``n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'``), so
        clustering quality is unchanged.

        Clamping handles small corpora: UMAP's k-NN graph requires
        ``n_samples > n_neighbors``; with the effective ``n_neighbors``
        (``self._bertopic_n_neighbors``, default 15) any corpus of
        ``n_docs <= n_neighbors`` reduces to a zero-row array and ``fit_transform``
        raises ``Found array with 0 sample(s)``. Clamping both ``n_neighbors`` and
        ``n_components`` to the corpus size keeps the reduction well-defined so
        small-but-valid sets still cluster. The boundary tracks the effective
        ``n_neighbors``, so raising it scales the guard with no magic constants.

        Returns None only when ``umap`` is not importable (BERTopic then falls back to
        its own default reduction — unseeded, but at least functional).
        """
        try:
            from umap import UMAP
        except ImportError:
            return None
        n_neighbors = self._bertopic_n_neighbors
        return UMAP(
            n_neighbors=max(2, min(n_neighbors, n_docs - 1)),
            n_components=max(2, min(self.BERTOPIC_DEFAULT_N_COMPONENTS, n_docs - 2)),
            min_dist=0.0,
            metric='cosine',
            random_state=self._seed if self._seed is not None else 42,
        )

    # Back-compat alias — the method previously only produced a UMAP for small corpora.
    _build_small_corpus_umap = _build_umap

    @trace(name='cluster_evidence_with_bertopic')
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
                    'error': f'Too few documents for BERTopic (min {self.BERTOPIC_MIN_DOCUMENTS})',
                    'reason': 'too_few_documents',
                    'coverage': self._unclustered_coverage(evidence, items_with_text),
                },
            )

        representation_model = KeyBERTInspired()
        bertopic_kwargs: Dict[str, Any] = dict(
            embedding_model=self._bertopic_embedding_model,
            representation_model=representation_model,
            nr_topics='auto',
            min_topic_size=self._min_category_size,
            verbose=False,
        )
        umap_model = self._build_umap(len(texts))
        if umap_model is not None:
            bertopic_kwargs['umap_model'] = umap_model
        topic_model = BERTopic(**bertopic_kwargs)

        try:
            topics, probs = topic_model.fit_transform(texts)
            topic_info = topic_model.get_topic_info()
        except Exception as e:
            # Degenerate corpora (few or near-duplicate docs) can still break BERTopic's
            # dimensionality reduction even with a clamped UMAP. Degrade gracefully so the
            # caller (HYBRID) can fall back to LLM clustering instead of crashing the run.
            logger.warning('BERTopic clustering failed (%s); returning no patterns.', e)
            return PatternDiscoveryResult(
                patterns=[],
                uncategorized=record_ids,
                total_analyzed=len(evidence),
                method=ClusteringMethod.BERTOPIC,
                metadata={
                    'error': f'BERTopic clustering failed: {e}',
                    'reason': 'bertopic_failed',
                    'coverage': self._unclustered_coverage(evidence, items_with_text),
                },
            )

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

        coverage = build_coverage(
            submitted_ids=record_ids,
            assigned_ids={rid for p in patterns for rid in p.record_ids},
            uncategorized_ids=uncategorized,
            filtered_small_cluster_ids=set(),
            ineligible=len(evidence) - len(record_ids),
        )

        return PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=uncategorized,
            total_analyzed=len(evidence),
            method=ClusteringMethod.BERTOPIC,
            metadata={'num_topics': len(patterns), 'coverage': coverage},
        )

    def _format_topic_name(self, topic_words: List[tuple]) -> str:
        if not topic_words:
            return 'Unknown Pattern'
        top_words = [word.title() for word, _ in topic_words[: self.TOPIC_NAME_WORDS]]
        return ' / '.join(top_words)

    @trace(name='cluster_evidence_hybrid')
    async def _cluster_evidence_hybrid(
        self, evidence: Dict[str, EvidenceItem]
    ) -> PatternDiscoveryResult:
        # Small corpora are routed straight to LLM clustering — BERTopic's UMAP/HDBSCAN
        # reduction is degenerate below the n_neighbors boundary even when clamped, so
        # LLM grouping yields better patterns than a forced topic model. Opt out with
        # small_corpus_llm_fallback=False to keep the clamped-BERTopic behavior.
        if self._small_corpus_llm_fallback:
            n_text = sum(1 for item in evidence.values() if item.text)
            if n_text <= self._bertopic_n_neighbors:
                logger.info(
                    'Small corpus (%d docs <= n_neighbors=%d); using LLM clustering for '
                    'HYBRID instead of BERTopic.',
                    n_text,
                    self._bertopic_n_neighbors,
                )
                llm_result = await self._cluster_evidence_with_llm(evidence)
                llm_result.method = ClusteringMethod.HYBRID
                llm_result.metadata['small_corpus_llm_fallback'] = n_text
                return llm_result

        result = await self._cluster_evidence_with_bertopic(evidence)

        if not result.patterns:
            # A deliberate too-few-documents early-out stays empty (too little signal to
            # cluster). But a BERTopic failure, or a viable corpus that yielded no topics,
            # falls back to LLM clustering so the evidence is grouped rather than dropped.
            if result.metadata.get('reason') == 'too_few_documents':
                result.method = ClusteringMethod.HYBRID
                return result
            logger.info(
                'BERTopic produced no patterns (%s); falling back to LLM clustering.',
                result.metadata.get('error', 'no topics'),
            )
            llm_result = await self._cluster_evidence_with_llm(evidence)
            llm_result.method = ClusteringMethod.HYBRID
            llm_result.metadata['bertopic_fallback'] = result.metadata.get(
                'error', 'no topics'
            )
            return llm_result

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
