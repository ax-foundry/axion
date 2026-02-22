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
    """Structured provenance attached to learning artifacts for sinks."""

    source_ref: Optional[str] = None
    clustering_method: Optional[str] = None
    total_analyzed: int = 0
    supporting_count: int = 0
    cluster_category: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
class PipelineResult:
    """Complete result from the evidence pipeline."""

    clustering_result: PatternDiscoveryResult
    learnings: List[LearningArtifact]
    filtered_count: int = 0
    deduplicated_count: int = 0
    validation_repairs: int = 0
    sink_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
