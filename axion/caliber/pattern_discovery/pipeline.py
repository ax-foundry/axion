import asyncio
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Set,
    Union,
)

from axion._core.logging import get_logger
from axion._core.tracing import init_tracer, trace
from axion.caliber.pattern_discovery._utils import (
    ExcerptFn,
    MetadataConfig,
    RecurrenceKeyFn,
    SamplerFn,
    ScoreFn,
    aggregate_cluster_metadata,
    build_coverage,
    check_recurrence,
    count_distinct_keys,
    default_excerpt,
    default_recurrence_key,
    default_tag_normalizer,
    deterministic_sample,
    format_metadata_header,
    normalize_evidence,
    round_robin_by_key,
    score_by_confidence,
    validate_learning,
)
from axion.caliber.pattern_discovery.discovery import PatternDiscovery
from axion.caliber.pattern_discovery.handlers import (
    ClusterForDistillation,
    DistillationHandler,
    DistillationInput,
    ExampleItem,
    LearningArtifactOutput,
    default_distillation_instruction,
)
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    EvidenceItem,
    FailedCluster,
    LearningArtifact,
    LearningRecord,
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
    MAX_EXAMPLES_PER_CLUSTER: int = 5
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
        tracer: Optional[Any] = None,
        # New params stay at the end so existing positional callers are unaffected.
        cluster_model_name: Optional[str] = None,
        cluster_llm=None,
        cluster_llm_provider: Optional[str] = None,
        disjoint_clusters: bool = False,
        clustering_excerpt_chars: Optional[int] = None,
        min_coverage: Optional[float] = None,
        raise_on_low_coverage: bool = False,
        score_fn: Optional[ScoreFn] = None,
        fail_on_all_errors: bool = False,
        sink_on_error: Literal['raise', 'continue'] = 'raise',
        sampler: Optional[SamplerFn] = None,
    ) -> None:
        self._method = method
        self._recurrence_threshold = recurrence_threshold
        self._recurrence_key_fn = recurrence_key_fn or default_recurrence_key
        if max_learnings_per_cluster < 1:
            raise ValueError('max_learnings_per_cluster must be >= 1')
        self._max_learnings_per_cluster = max_learnings_per_cluster
        self._max_items = max_items
        # How evidence is cut down to max_items. The default is a uniform
        # hash/seeded sample; recurrence_balanced_sampler(key_fn) keeps parents
        # represented instead.
        self._sampler: SamplerFn = sampler or deterministic_sample
        self._domain_context = domain_context
        self._metadata_config = metadata_config or MetadataConfig()
        self._excerpt_fn = excerpt_fn
        self._seed = seed
        self._max_concurrent = max_concurrent_distillations
        self.tracer = init_tracer('base', tracer=tracer)

        # Ranking. With no score_fn, learnings are ordered by confidence and each
        # cluster's candidates are truncated by confidence before validation,
        # exactly as before. With one, candidates are validated first and
        # truncated by score, so the caller's ranking also drives selection.
        self._custom_score_fn = score_fn
        self._score_fn: ScoreFn = score_fn or score_by_confidence
        # Failure policy. Defaults keep today's behavior: a failed distillation is
        # logged and skipped; a failed sink write raises mid-loop.
        self._fail_on_all_errors = fail_on_all_errors
        if sink_on_error not in ('raise', 'continue'):
            raise ValueError("sink_on_error must be 'raise' or 'continue'")
        self._sink_on_error = sink_on_error

        self._sanitizer: Sanitizer = sanitizer or NoopSanitizer()
        self._sink = sink
        self._deduper = deduper
        self._tag_normalizer = tag_normalizer or default_tag_normalizer

        # Build default strategies if not provided.
        # cluster_llm / cluster_model_name / cluster_llm_provider let clustering (and
        # BERTopic label refinement) run on a different model than distillation — e.g.
        # a temperature-0 model for stable cluster assignments while distillation stays
        # on a fixed-temperature reasoning model. Unset -> both share the main llm.
        if clusterer is not None:
            self._clusterer: EvidenceClusterer = clusterer
        else:
            discovery = PatternDiscovery(
                model_name=cluster_model_name or model_name,
                llm=cluster_llm if cluster_llm is not None else llm,
                llm_provider=cluster_llm_provider or llm_provider,
                instruction=clustering_instruction,
                max_notes=max_items,
                min_category_size=min_category_size,
                bertopic_embedding_model=bertopic_embedding_model,
                metadata_config=self._metadata_config,
                excerpt_fn=excerpt_fn,
                seed=seed,
                tracer=self.tracer,
                disjoint_clusters=disjoint_clusters,
                clustering_excerpt_chars=clustering_excerpt_chars,
                min_coverage=min_coverage,
                raise_on_low_coverage=raise_on_low_coverage,
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
                tracer=self.tracer,
            )
            self._writer = _DefaultWriter(handler)

    @trace(name='execute', capture_args=True)
    async def run(
        self,
        evidence: Union[Sequence[EvidenceItem], Dict[str, EvidenceItem]],
        method: Optional[ClusteringMethod] = None,
    ) -> PipelineResult:
        method = method or self._method

        # Normalize input
        evidence_dict, duplicates_dropped = normalize_evidence(evidence)
        input_count = len(evidence_dict)

        # Sample if too many
        sampled_out = 0
        keys_in_input = count_distinct_keys(
            list(evidence_dict), evidence_dict, self._recurrence_key_fn
        )
        if len(evidence_dict) > self._max_items:
            evidence_dict = self._sampler(evidence_dict, self._max_items, self._seed)
            sampled_out = input_count - len(evidence_dict)
        keys_after_sampling = count_distinct_keys(
            list(evidence_dict), evidence_dict, self._recurrence_key_fn
        )

        # Sanitize
        evidence_dict = await self._sanitize(evidence_dict)

        # Cluster
        clustering_result = await self._clusterer.cluster(evidence_dict, method)

        # 3. Pre-filter: drop clusters below recurrence_threshold, counted in
        # distinct recurrence keys (equals item count with the default key).
        surviving_patterns = [
            p
            for p in clustering_result.patterns
            if count_distinct_keys(p.record_ids, evidence_dict, self._recurrence_key_fn)
            >= self._recurrence_threshold
        ]
        filtered_count = len(clustering_result.patterns) - len(surviving_patterns)
        surviving_ids = {rid for p in surviving_patterns for rid in p.record_ids}
        filtered_below_threshold = {
            rid
            for p in clustering_result.patterns
            for rid in p.record_ids
            if rid not in surviving_ids
        }

        # Distill + Validate (bounded concurrency)
        total_repairs = 0
        timestamp = datetime.now(timezone.utc).isoformat()
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _process_cluster(
            pattern: DiscoveredPattern,
        ) -> tuple[List[LearningRecord], int]:
            async with semaphore:
                return await self._distill_and_validate(
                    pattern, evidence_dict, clustering_result, method, timestamp
                )

        tasks = [_process_cluster(p) for p in surviving_patterns]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        records: List[LearningRecord] = []
        failed_clusters: List[FailedCluster] = []
        for pattern, res in zip(surviving_patterns, results):
            if isinstance(res, BaseException):
                logger.warning(
                    f'Distillation failed for cluster "{pattern.category}": {res}'
                )
                failed_clusters.append(
                    FailedCluster(
                        category=pattern.category,
                        error=f'{type(res).__name__}: {res}',
                        item_count=len(pattern.record_ids),
                    )
                )
                continue
            cluster_records, repairs = res
            total_repairs += repairs
            records.extend(cluster_records)

        if (
            self._fail_on_all_errors
            and surviving_patterns
            and len(failed_clusters) == len(surviving_patterns)
        ):
            raise RuntimeError(
                f'Distillation failed for all {len(failed_clusters)} clusters; '
                f'first error: {failed_clusters[0].error}'
            )

        # Order by score desc (confidence unless a score_fn was given). Stable, so
        # equal scores keep distillation order.
        records.sort(key=lambda r: r.score, reverse=True)

        # Tag normalize
        for rec in records:
            rec.learning.tags = self._tag_normalizer(rec.learning.tags)

        # Dedupe (after ordering, so the best-scored twin survives)
        deduplicated_count = 0
        if self._deduper is not None:
            if hasattr(self._deduper, 'reset_per_run') and getattr(
                self._deduper, 'reset_per_run', False
            ):
                if hasattr(self._deduper, 'reset'):
                    self._deduper.reset()

            kept_records: List[LearningRecord] = []
            for rec in records:
                if await self._deduper.is_duplicate(rec.learning):
                    deduplicated_count += 1
                else:
                    kept_records.append(rec)
            records = kept_records

        # Sink
        sink_ids: List[str] = []
        if self._sink is not None:
            for rec in records:
                if self._sink_on_error == 'raise':
                    sid = await self._sink.write(rec.learning, rec.provenance)
                else:
                    try:
                        sid = await self._sink.write(rec.learning, rec.provenance)
                    except Exception as e:
                        rec.sink_error = f'{type(e).__name__}: {e}'
                        logger.warning(
                            f'Sink write failed for "{rec.learning.title}": {e}'
                        )
                        continue
                rec.sink_id = sid
                sink_ids.append(sid)

        coverage = self._roll_up_coverage(
            clustering_result,
            evidence_dict,
            surviving_ids=surviving_ids,
            filtered_below_threshold=len(filtered_below_threshold),
            input_count=input_count,
            duplicates_dropped=duplicates_dropped,
            sampled_out=sampled_out,
        )
        coverage['distinct_keys_input'] = keys_in_input
        coverage['distinct_keys_sampled'] = keys_after_sampling

        return PipelineResult(
            clustering_result=clustering_result,
            learnings=[rec.learning for rec in records],
            filtered_count=filtered_count,
            deduplicated_count=deduplicated_count,
            validation_repairs=total_repairs,
            sink_ids=sink_ids,
            input_count=input_count,
            sampled_out=sampled_out,
            duplicates_dropped=duplicates_dropped,
            coverage=coverage,
            records=records,
            failed_clusters=failed_clusters,
        )

    def _make_record(
        self,
        learning: LearningArtifact,
        pattern: DiscoveredPattern,
        evidence: Dict[str, EvidenceItem],
        clustering_result: PatternDiscoveryResult,
        method: Optional[ClusteringMethod],
        timestamp: str,
    ) -> LearningRecord:
        """Build the complete provenance for a learning and score it once.

        This is the only place provenance is constructed, so a ``score_fn``
        sees the same fields during selection as it does in the final
        ordering and in what the sink receives.
        """
        supporting_items = [
            evidence[sid] for sid in learning.supporting_item_ids if sid in evidence
        ]
        source_refs = [item.source_ref for item in supporting_items if item.source_ref]
        distinct_keys = sorted(
            {self._recurrence_key_fn(item) for item in supporting_items}
        )
        provenance = Provenance(
            source_ref=source_refs[0] if source_refs else None,
            clustering_method=method.value if method else None,
            total_analyzed=clustering_result.total_analyzed,
            supporting_count=len(learning.supporting_item_ids),
            cluster_category=pattern.category,
            timestamp=timestamp,
            metadata={'all_source_refs': source_refs},
            distinct_keys=distinct_keys,
            distinct_key_count=len(distinct_keys),
            cluster_size=len(pattern.record_ids),
            cited_support_count=len(supporting_items),
        )
        score = self._score_fn(learning, provenance, supporting_items)
        return LearningRecord(
            learning=learning,
            provenance=provenance,
            score=float(score),
            cluster=pattern,
        )

    @staticmethod
    def _roll_up_coverage(
        clustering_result: PatternDiscoveryResult,
        evidence: Dict[str, EvidenceItem],
        *,
        surviving_ids: Set[str],
        filtered_below_threshold: int,
        input_count: int,
        duplicates_dropped: int,
        sampled_out: int,
    ) -> Dict[str, Any]:
        """Fold pipeline-level filters into the clusterer's coverage.

        Buckets stay mutually exclusive: ``assigned`` is recomputed from the
        patterns that survived the recurrence pre-filter, and the items of the
        patterns it dropped move to ``filtered_below_threshold``. ``ratio`` is
        left as the clusterer reported it, since the pre-filter is not the
        model's omission. A clusterer that reports no coverage of its own gets
        the buckets derived from what it returned, so every documented key is
        always present.
        """
        coverage = dict(clustering_result.metadata.get('coverage', {}))
        derived = build_coverage(
            submitted_ids=list(evidence),
            assigned_ids={
                rid for p in clustering_result.patterns for rid in p.record_ids
            },
            uncategorized_ids=list(clustering_result.uncategorized),
            filtered_small_cluster_ids=set(),
        )
        for key, value in derived.items():
            coverage.setdefault(key, value)
        coverage['assigned'] = len(surviving_ids)
        coverage['filtered_below_threshold'] = filtered_below_threshold
        coverage['input_count'] = input_count
        coverage['duplicates_dropped'] = duplicates_dropped
        coverage['sampled_out'] = sampled_out + coverage.get('sampled_out', 0)
        return coverage

    def display(self, result: PipelineResult) -> None:
        """Display a pipeline result (summary + patterns + learnings).

        Auto-detects Jupyter notebook vs console environment.
        """
        from axion.caliber.pattern_discovery.display import display_pipeline_result

        display_pipeline_result(result)

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
        method: Optional[ClusteringMethod],
        timestamp: str,
    ) -> tuple[List[LearningRecord], int]:
        """Returns (records for the validated learnings, total_repairs)."""
        # Build ClusterForDistillation. Examples are spread across recurrence keys
        # so one prolific source cannot fill every slot, and each carries its id so
        # the writer can cite text it actually saw.
        example_texts: List[str] = []
        examples: List[ExampleItem] = []
        meta_cfg = self._metadata_config
        example_ids = round_robin_by_key(
            pattern.record_ids,
            evidence,
            self.MAX_EXAMPLES_PER_CLUSTER,
            self._recurrence_key_fn,
        )
        for rid in example_ids:
            item = evidence[rid]
            if self._excerpt_fn:
                text = self._excerpt_fn(item)
            else:
                text = default_excerpt(item, self.EXAMPLE_EXCERPT_CHARS)

            if meta_cfg.include_in_distillation and item.metadata:
                header = format_metadata_header(item.metadata, meta_cfg)
                if header:
                    text = header + '\n' + text
            example_texts.append(text)
            examples.append(ExampleItem(item_id=rid, text=text))

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
            examples=examples,
            distinct_source_count=count_distinct_keys(
                pattern.record_ids, evidence, self._recurrence_key_fn
            ),
        )

        raw_learnings = await self._writer.distill(cluster_input, self._domain_context)

        cluster_ids = set(pattern.record_ids)

        def _validate(raw: Any) -> tuple[Optional[LearningArtifact], int]:
            artifact, repairs = validate_learning(raw, cluster_ids)
            if artifact is None:
                return None, repairs
            # Recurrence check on validated supporting IDs
            if not check_recurrence(
                artifact.supporting_item_ids,
                evidence,
                self._recurrence_threshold,
                self._recurrence_key_fn,
            ):
                return None, repairs
            return artifact, repairs

        total_repairs = 0

        def _record(artifact: LearningArtifact) -> LearningRecord:
            return self._make_record(
                artifact, pattern, evidence, clustering_result, method, timestamp
            )

        if self._custom_score_fn is None:
            # Default path: truncate by confidence, then validate.
            sorted_learnings = sorted(
                raw_learnings, key=lambda learning: learning.confidence, reverse=True
            )[: self._max_learnings_per_cluster]
            validated: List[LearningRecord] = []
            for raw in sorted_learnings:
                artifact, repairs = _validate(raw)
                total_repairs += repairs
                if artifact is not None:
                    validated.append(_record(artifact))
            return validated, total_repairs

        # Scored path: validate everything first so the score sees repaired
        # support, then keep the best-scored candidates. Truncating by confidence
        # first would discard a learning the caller's score prefers. Each
        # candidate is scored against its complete provenance, and that score is
        # the one the record keeps.
        scored: List[LearningRecord] = []
        for raw in raw_learnings:
            artifact, repairs = _validate(raw)
            total_repairs += repairs
            if artifact is not None:
                scored.append(_record(artifact))
        scored.sort(key=lambda rec: rec.score, reverse=True)  # stable
        return scored[: self._max_learnings_per_cluster], total_repairs
