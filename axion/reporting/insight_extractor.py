import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from axion._core.asyncio import run_async_function
from axion._core.logging import get_logger
from axion._core.tracing import trace
from axion.caliber.pattern_discovery._utils import MetadataConfig
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    EvidenceItem,
    LearningArtifact,
    PipelineResult,
)
from axion.caliber.pattern_discovery.pipeline import EvidencePipeline
from axion.reporting.issue_extractor import ExtractedIssue, IssueExtractionResult

logger = get_logger(__name__)


@dataclass
class InsightPattern:
    """
    A cross-metric pattern discovered 
    from evaluation issues.
    """

    category: str
    description: str
    count: int
    issue_ids: List[str]
    metrics_involved: List[str]
    is_cross_metric: bool
    distinct_test_cases: int
    examples: List[str]
    confidence: Optional[float] = None


@dataclass
class InsightResult:
    """Complete result from insight extraction."""

    patterns: List[InsightPattern]
    learnings: List[LearningArtifact]
    total_issues_analyzed: int
    clustering_method: ClusteringMethod
    pipeline_result: PipelineResult


def _issue_to_evidence(issue: ExtractedIssue) -> Optional[EvidenceItem]:
    """
    Convert ExtractedIssue -> EvidenceItem. 
    Returns None if no meaningful text.
    """
    reasoning = issue.reasoning or ''
    query = issue.item_context.get('query', '')

    if not reasoning and not query:
        return None

    # Build text: reasoning-first with failure cue
    text = '\n'.join(
        part
        for part in (
            reasoning or None,
            f'[{issue.metric_name} / {issue.signal_name}: {issue.value}]',
            f'Query: {query}' if query else None,
        )
        if part
    )

    # Stable hash-based ID
    hash_input = (
        f'{issue.test_case_id}{issue.metric_name}'
        f'{issue.signal_group}{issue.signal_name}{issue.value}'
    )
    evidence_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    metadata: Dict[str, Any] = {
        'metric_name': issue.metric_name,
        'signal_name': issue.signal_name,
        'signal_group': issue.signal_group,
        'value': issue.value,
        'score': issue.score,
        'source_path': issue.source_path,
    }

    return EvidenceItem(
        id=evidence_id,
        text=text,
        metadata=metadata,
        source_ref=issue.test_case_id,
    )


_EMPTY_PIPELINE_RESULT = PipelineResult(
    clustering_result=None,  # type: ignore[arg-type]
    learnings=[],
    filtered_count=0,
    deduplicated_count=0,
    validation_repairs=0,
    sink_ids=[],
)


class InsightExtractor:
    """
    Bridges IssueExtractor output with EvidencePipeline 
    for cross-metric pattern discovery.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm=None,
        llm_provider: Optional[str] = None,
        method: ClusteringMethod = ClusteringMethod.LLM,
        recurrence_threshold: int = 2,
        max_items: int = 50,
        min_category_size: int = 2,
        pipeline: Optional[EvidencePipeline] = None,
        **pipeline_kwargs,
    ) -> None:
        if pipeline is not None and pipeline_kwargs:
            raise ValueError(
                'Cannot pass pipeline_kwargs when pipeline is provided'
            )

        self._method = method

        if pipeline is not None:
            self._pipeline = pipeline
        else:
            metadata_config = MetadataConfig(
                include_in_clustering=True,
                allowed_keys={'metric_name', 'signal_name', 'value', 'score'},
            )
            self._pipeline = EvidencePipeline(
                model_name=model_name,
                llm=llm,
                llm_provider=llm_provider,
                method=method,
                recurrence_threshold=recurrence_threshold,
                recurrence_key_fn=lambda item: item.source_ref or item.id,
                max_items=max_items,
                min_category_size=min_category_size,
                metadata_config=metadata_config,
                **pipeline_kwargs,
            )

    @trace(name='InsightExtractor.analyze')
    async def analyze(
        self, extraction_result: IssueExtractionResult
    ) -> InsightResult:
        """
        Analyze extracted issues for cross-metric patterns.

        Args:
            extraction_result: Output from IssueExtractor.extract_from_evaluation()

        Returns:
            InsightResult with discovered patterns and learnings.
        """
        # Short-circuit on empty input
        if not extraction_result.all_issues:
            return self._empty_result()

        # Convert issues to evidence items
        evidence_dict: Dict[str, EvidenceItem] = {}
        issue_lookup: Dict[str, ExtractedIssue] = {}
        for issue in extraction_result.all_issues:
            item = _issue_to_evidence(issue)
            if item is not None:
                evidence_dict[item.id] = item
                issue_lookup[item.id] = issue

        # Short-circuit if all conversions returned None
        if not evidence_dict:
            return self._empty_result()

        # Build domain context and inject into pipeline for distillation
        domain_context = self._build_domain_context(extraction_result)
        old_ctx = self._pipeline._domain_context
        self._pipeline._domain_context = domain_context
        try:
            pipeline_result = await self._pipeline.run(
                evidence_dict, method=self._method
            )
        finally:
            self._pipeline._domain_context = old_ctx

        # Build InsightPattern list from clustering result
        patterns = self._build_patterns(
            pipeline_result, evidence_dict, issue_lookup
        )

        return InsightResult(
            patterns=patterns,
            learnings=pipeline_result.learnings,
            total_issues_analyzed=len(evidence_dict),
            clustering_method=self._method,
            pipeline_result=pipeline_result,
        )

    def analyze_sync(
        self, extraction_result: IssueExtractionResult
    ) -> InsightResult:
        """Synchronous wrapper for analyze()."""
        return run_async_function(self.analyze, extraction_result)

    def _empty_result(self) -> InsightResult:
        return InsightResult(
            patterns=[],
            learnings=[],
            total_issues_analyzed=0,
            clustering_method=self._method,
            pipeline_result=PipelineResult(
                clustering_result=None,  # type: ignore[arg-type]
                learnings=[],
            ),
        )

    def _build_domain_context(
        self, extraction_result: IssueExtractionResult
    ) -> str:
        metrics = sorted(extraction_result.issues_by_metric.keys())
        metric_list = ', '.join(metrics)
        n = extraction_result.total_test_cases
        m = extraction_result.issues_found
        k = len(metrics)

        if extraction_result.evaluation_name:
            return (
                f"Evaluation '{extraction_result.evaluation_name}': "
                f'{n} test cases, {m} issues across {k} metrics ({metric_list})'
            )
        return (
            f'Evaluation: {n} test cases, {m} issues '
            f'across {k} metrics ({metric_list})'
        )

    def _build_patterns(
        self,
        pipeline_result: PipelineResult,
        evidence_dict: Dict[str, EvidenceItem],
        issue_lookup: Dict[str, ExtractedIssue],
    ) -> List[InsightPattern]:
        patterns: List[InsightPattern] = []
        clustering_result = pipeline_result.clustering_result
        if clustering_result is None:
            return patterns

        for dp in clustering_result.patterns:
            # Collect metrics involved
            metrics_in_cluster: set = set()
            source_refs_in_cluster: set = set()
            for rid in dp.record_ids:
                item = evidence_dict.get(rid)
                if item and item.metadata.get('metric_name'):
                    metrics_in_cluster.add(item.metadata['metric_name'])
                if item and item.source_ref:
                    source_refs_in_cluster.add(item.source_ref)

            metrics_list = sorted(metrics_in_cluster)

            patterns.append(
                InsightPattern(
                    category=dp.category,
                    description=dp.description,
                    count=dp.count,
                    issue_ids=dp.record_ids,
                    metrics_involved=metrics_list,
                    is_cross_metric=len(metrics_list) >= 2,
                    distinct_test_cases=len(source_refs_in_cluster),
                    examples=dp.examples,
                    confidence=dp.confidence,
                )
            )

        return patterns
