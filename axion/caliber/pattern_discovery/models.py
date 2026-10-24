from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ClusteringMethod(str, Enum):
    """Available clustering methods."""

    LLM = 'llm'  # LLM-based semantic clustering
    BERTOPIC = 'bertopic'  # BERTopic topic modeling (no LLM needed)
    HYBRID = 'hybrid'  # BERTopic clustering + LLM label refinement


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


@dataclass
class EvidenceItem:
    """A single piece of evidence for clustering.

    Represents any text source (conversation, bug report, eval note, etc.)
    with optional structured metadata and provenance.
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_ref: Optional[str] = None


@dataclass
class Provenance:
    """Structured provenance attached to every learning artifact.

    ``supporting_count`` counts cited items. ``distinct_key_count`` counts the
    independent sources behind them (per ``recurrence_key_fn``); several items
    from one source are one key. ``cluster_size`` is the whole cluster the
    learning was distilled from, which is weaker evidence than ``cited_support_count``
    because the writer only saw a sample of it.
    """

    source_ref: Optional[str] = None
    clustering_method: Optional[str] = None
    total_analyzed: int = 0
    supporting_count: int = 0
    cluster_category: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    distinct_keys: List[str] = field(default_factory=list)
    distinct_key_count: int = 0
    cluster_size: int = 0
    cited_support_count: int = 0


@dataclass
class LearningArtifact:
    """A synthesized insight distilled from a cluster of evidence."""

    title: str
    content: str
    tags: List[str]
    confidence: float
    supporting_item_ids: List[str]
    recommended_actions: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    scope: Optional[str] = None
    when_not_to_apply: Optional[str] = None


@dataclass
class LearningRecord:
    """One learning with everything the pipeline knows about it.

    ``score`` is what the pipeline ordered by (LLM confidence unless a
    ``score_fn`` was given). ``sink_id`` is set when a sink accepted the
    learning; ``sink_error`` when the sink raised and the pipeline was told to
    continue.
    """

    learning: LearningArtifact
    provenance: Provenance
    score: float
    cluster: Optional[DiscoveredPattern] = None
    sink_id: Optional[str] = None
    sink_error: Optional[str] = None


@dataclass
class FailedCluster:
    """A cluster whose distillation raised."""

    category: str
    error: str
    item_count: int = 0


@dataclass
class ClusterWithoutLearnings:
    """A cluster that distilled without error but produced no learning.

    ``candidates`` is how many learnings the writer returned. Zero means the
    model found nothing to say; otherwise every candidate was dropped, and the
    two ``dropped_*`` counts say why: no cited id belonged to the cluster, or
    the cited ids spanned fewer recurrence keys than ``recurrence_threshold``.
    """

    category: str
    item_count: int = 0
    candidates: int = 0
    dropped_invalid: int = 0
    dropped_below_recurrence: int = 0


@dataclass
class PipelineResult:
    """Complete result from the evidence pipeline.

    ``learnings`` and ``sink_ids`` are the flat views; ``records`` carries the
    same learnings in the same order with provenance, score, and sink outcome.
    """

    clustering_result: PatternDiscoveryResult
    learnings: List[LearningArtifact]
    filtered_count: int = 0
    deduplicated_count: int = 0
    validation_repairs: int = 0
    sink_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Accounting: where every input item ended up. ``input_count`` is the size of
    # the evidence handed to ``run`` before any sampling; ``coverage`` rolls the
    # clusterer's per-item buckets up with the pipeline's own filters.
    input_count: int = 0
    sampled_out: int = 0
    duplicates_dropped: int = 0
    coverage: Dict[str, Any] = field(default_factory=dict)
    records: List['LearningRecord'] = field(default_factory=list)
    failed_clusters: List['FailedCluster'] = field(default_factory=list)
    # Distillation accounting: clusters that produced nothing without raising,
    # and the two reasons a returned candidate is discarded after validation.
    clusters_without_learnings: List['ClusterWithoutLearnings'] = field(
        default_factory=list
    )
    learnings_dropped_invalid: int = 0
    learnings_dropped_below_recurrence: int = 0
