import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union

from axion._core.logging import get_logger
from axion._core.tracing import trace
from axion.caliber.pattern_discovery._utils import (
    ExcerptFn,
    MetadataConfig,
    RecurrenceKeyFn,
    aggregate_cluster_metadata,
    check_recurrence,
    default_excerpt,
    default_recurrence_key,
    default_tag_normalizer,
    deterministic_sample,
    format_metadata_header,
    validate_learning,
)
from axion.caliber.pattern_discovery.discovery import PatternDiscovery
from axion.caliber.pattern_discovery.handlers import (
    ClusterForDistillation,
    DistillationHandler,
    DistillationInput,
    LearningArtifactOutput,
    default_distillation_instruction,
)
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    EvidenceItem,
    LearningArtifact,
    PatternDiscoveryResult,
    PipelineResult,
    Provenance,
)
from axion.caliber.pattern_discovery.plugins import (
    ArtifactSink,
    Deduper,
    NoopSanitizer,
    Sanitizer,
)

logger = get_logger(__name__)


class EvidenceClusterer(Protocol):
    async def cluster(
        self, evidence: Dict[str, EvidenceItem], method: ClusteringMethod
    ) -> PatternDiscoveryResult: ...


class ArtifactWriter(Protocol):
    async def distill(
        self,
        cluster: ClusterForDistillation,
        domain_context: Optional[str],
    ) -> List[LearningArtifactOutput]: ...


class _DefaultClusterer:
    """Wraps PatternDiscovery as an EvidenceClusterer."""

    def __init__(self, discovery: PatternDiscovery) -> None:
        self._discovery = discovery

    async def cluster(
        self,
        evidence: Dict[str, EvidenceItem],
        method: ClusteringMethod,
    ) -> PatternDiscoveryResult:
        return await self._discovery.discover_from_evidence(evidence, method)


class _DefaultWriter:
    """Wraps DistillationHandler as an ArtifactWriter."""

    def __init__(self, handler: DistillationHandler) -> None:
        self._handler = handler

    async def distill(
        self,
        cluster: ClusterForDistillation,
        domain_context: Optional[str],
    ) -> List[LearningArtifactOutput]:
        inp = DistillationInput(cluster=cluster, domain_context=domain_context)
        output = await self._handler.execute(inp)
        return output.learnings


class EvidencePipeline:
    """Orchestrates evidence → clusters → KB-ready learnings."""

    MAX_LEARNINGS_PER_CLUSTER: int = 3
    EXAMPLE_EXCERPT_CHARS: int = 200

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm=None,
        llm_provider: Optional[str] = None,
        clustering_instruction: Optional[str] = None,
        distillation_instruction: Optional[str] = None,
        clusterer: Optional[EvidenceClusterer] = None,
        writer: Optional[ArtifactWriter] = None,
        method: ClusteringMethod = ClusteringMethod.LLM,
        recurrence_threshold: int = 2,
        recurrence_key_fn: Optional[RecurrenceKeyFn] = None,
        max_learnings_per_cluster: int = 3,
        max_items: int = 50,
        min_category_size: int = 2,
        domain_context: Optional[str] = None,
        metadata_config: Optional[MetadataConfig] = None,
        excerpt_fn: Optional[ExcerptFn] = None,
        seed: Optional[int] = None,
        max_concurrent_distillations: int = 5,
        sanitizer: Optional[Sanitizer] = None,
        sink: Optional[ArtifactSink] = None,
        deduper: Optional[Deduper] = None,
        tag_normalizer: Optional[Callable[[List[str]], List[str]]] = None,
        bertopic_embedding_model: Any = 'all-MiniLM-L6-v2',
    ) -> None:
        self._method = method
        self._recurrence_threshold = recurrence_threshold
        self._recurrence_key_fn = recurrence_key_fn or default_recurrence_key
        if max_learnings_per_cluster < 1:
            raise ValueError('max_learnings_per_cluster must be >= 1')
        self._max_learnings_per_cluster = max_learnings_per_cluster
        self._max_items = max_items
        self._domain_context = domain_context
        self._metadata_config = metadata_config or MetadataConfig()
        self._excerpt_fn = excerpt_fn
        self._seed = seed
        self._max_concurrent = max_concurrent_distillations

        self._sanitizer: Sanitizer = sanitizer or NoopSanitizer()
        self._sink = sink
        self._deduper = deduper
        self._tag_normalizer = tag_normalizer or default_tag_normalizer

        # Build default strategies if not provided
        if clusterer is not None:
            self._clusterer: EvidenceClusterer = clusterer
        else:
            discovery = PatternDiscovery(
                model_name=model_name,
                llm=llm,
                llm_provider=llm_provider,
                instruction=clustering_instruction,
                max_notes=max_items,
                min_category_size=min_category_size,
                bertopic_embedding_model=bertopic_embedding_model,
                metadata_config=self._metadata_config,
                excerpt_fn=excerpt_fn,
                seed=seed,
            )
            self._clusterer = _DefaultClusterer(discovery)

        if writer is not None:
            self._writer: ArtifactWriter = writer
        else:
            if distillation_instruction is None:
                distillation_instruction = default_distillation_instruction(
                    self._max_learnings_per_cluster
                )
            handler = DistillationHandler(
                model_name=model_name,
                llm=llm,
                llm_provider=llm_provider,
                instruction=distillation_instruction,
            )
            self._writer = _DefaultWriter(handler)

    @trace(name='EvidencePipeline.run', capture_args=True)
    async def run(
        self,
        evidence: Union[Sequence[EvidenceItem], Dict[str, EvidenceItem]],
        method: Optional[ClusteringMethod] = None,
    ) -> PipelineResult:
        method = method or self._method

        # Normalize input
        evidence_dict = PatternDiscovery._to_evidence_dict(evidence)

        # Sample if too many
        if len(evidence_dict) > self._max_items:
            evidence_dict = deterministic_sample(
                evidence_dict, self._max_items, self._seed
            )

        # Sanitize
        evidence_dict = await self._sanitize(evidence_dict)

        # Cluster
        clustering_result = await self._clusterer.cluster(evidence_dict, method)

        # 3. Pre-filter: drop clusters below recurrence_threshold
        surviving_patterns = [
            p
            for p in clustering_result.patterns
            if len(p.record_ids) >= self._recurrence_threshold
        ]
        filtered_count = len(clustering_result.patterns) - len(surviving_patterns)

        # Distill + Validate (bounded concurrency)
        all_learnings: List[LearningArtifact] = []
        total_repairs = 0
        semaphore = asyncio.Semaphore(self._max_concurrent)

        # Track which cluster produced each learning (for provenance)
        learning_to_cluster: Dict[int, DiscoveredPattern] = {}

        async def _process_cluster(
            pattern: DiscoveredPattern,
        ) -> tuple[List[LearningArtifact], int]:
            async with semaphore:
                return await self._distill_and_validate(
                    pattern, evidence_dict, clustering_result
                )

        tasks = [_process_cluster(p) for p in surviving_patterns]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.warning(
                    f'Distillation failed for cluster '
                    f'"{surviving_patterns[i].category}": {res}'
                )
                continue
            learnings, repairs = res
            total_repairs += repairs
            for la in learnings:
                learning_to_cluster[id(la)] = surviving_patterns[i]
            all_learnings.extend(learnings)

        # Merge: sort by confidence desc
        all_learnings.sort(key=lambda la: la.confidence, reverse=True)

        # Tag normalize
        for la in all_learnings:
            la.tags = self._tag_normalizer(la.tags)

        # Dedupe
        deduplicated_count = 0
        if self._deduper is not None:
            if hasattr(self._deduper, 'reset_per_run') and getattr(
                self._deduper, 'reset_per_run', False
            ):
                if hasattr(self._deduper, 'reset'):
                    self._deduper.reset()

            kept: List[LearningArtifact] = []
            for la in all_learnings:
                if await self._deduper.is_duplicate(la):
                    deduplicated_count += 1
                else:
                    kept.append(la)
            all_learnings = kept

        # Sink
        sink_ids: List[str] = []
        if self._sink is not None:
            timestamp = datetime.now(timezone.utc).isoformat()
            for la in all_learnings:
                cluster_pattern = learning_to_cluster.get(id(la))
                # Resolve a representative source_ref from supporting items
                source_refs = [
                    evidence_dict[sid].source_ref
                    for sid in la.supporting_item_ids
                    if sid in evidence_dict and evidence_dict[sid].source_ref
                ]
                provenance = Provenance(
                    source_ref=source_refs[0] if source_refs else None,
                    clustering_method=method.value if method else None,
                    total_analyzed=clustering_result.total_analyzed,
                    supporting_count=len(la.supporting_item_ids),
                    cluster_category=(
                        cluster_pattern.category if cluster_pattern else None
                    ),
                    timestamp=timestamp,
                    metadata={
                        'all_source_refs': source_refs,
                    },
                )
                sid = await self._sink.write(la, provenance)
                sink_ids.append(sid)

        return PipelineResult(
            clustering_result=clustering_result,
            learnings=all_learnings,
            filtered_count=filtered_count,
            deduplicated_count=deduplicated_count,
            validation_repairs=total_repairs,
            sink_ids=sink_ids,
        )

    @trace(name='EvidencePipeline._sanitize')
    async def _sanitize(
        self, evidence: Dict[str, EvidenceItem]
    ) -> Dict[str, EvidenceItem]:
        result: Dict[str, EvidenceItem] = {}
        for eid, item in evidence.items():
            sanitized_text = await self._sanitizer.sanitize(item.text)
            result[eid] = EvidenceItem(
                id=item.id,
                text=sanitized_text,
                metadata=item.metadata,
                source_ref=item.source_ref,
            )
        return result

    @trace(name='EvidencePipeline._distill_and_validate')
    async def _distill_and_validate(
        self,
        pattern: DiscoveredPattern,
        evidence: Dict[str, EvidenceItem],
        clustering_result: PatternDiscoveryResult,
    ) -> tuple[List[LearningArtifact], int]:
        """Returns (validated_learnings, total_repairs)."""
        # Build ClusterForDistillation
        example_texts: List[str] = []
        meta_cfg = self._metadata_config
        for rid in pattern.record_ids[:5]:
            item = evidence.get(rid)
            if not item:
                continue
            if self._excerpt_fn:
                text = self._excerpt_fn(item)
            else:
                text = default_excerpt(item, self.EXAMPLE_EXCERPT_CHARS)

            if meta_cfg.include_in_distillation and item.metadata:
                header = format_metadata_header(item.metadata, meta_cfg)
                if header:
                    text = header + '\n' + text
            example_texts.append(text)

        metadata_summary = None
        if meta_cfg.include_in_distillation:
            metadata_summary = aggregate_cluster_metadata(
                evidence, pattern.record_ids, meta_cfg
            )

        cluster_input = ClusterForDistillation(
            category=pattern.category,
            description=pattern.description,
            item_ids=pattern.record_ids,
            example_texts=example_texts,
            metadata_summary=metadata_summary,
        )

        raw_learnings = await self._writer.distill(cluster_input, self._domain_context)

        # Truncate (by confidence desc)
        sorted_learnings = sorted(
            raw_learnings, key=lambda l: l.confidence, reverse=True
        )[: self._max_learnings_per_cluster]

        # Validate per-cluster
        cluster_ids = set(pattern.record_ids)
        validated: List[LearningArtifact] = []
        total_repairs = 0
        for raw in sorted_learnings:
            artifact, repairs = validate_learning(raw, cluster_ids)
            total_repairs += repairs
            if artifact is None:
                continue

            # Recurrence check on validated supporting IDs
            if not check_recurrence(
                artifact.supporting_item_ids,
                evidence,
                self._recurrence_threshold,
                self._recurrence_key_fn,
            ):
                continue

            validated.append(artifact)

        return validated, total_repairs
