# Models (existing)
# Compat (AnnotatedItem + converters)
from axion.caliber.pattern_discovery._compat import (
    AnnotatedItem,
    annotated_item_to_evidence,
    annotations_to_evidence,
    normalize_annotations,
)

# Utils (new)
from axion.caliber.pattern_discovery._utils import MetadataConfig

# Discovery (existing class, refactored)
from axion.caliber.pattern_discovery.discovery import PatternDiscovery

# Handlers — Pydantic I/O models (existing)
# Handlers — new Pydantic I/O models
# Handlers — existing handler classes
from axion.caliber.pattern_discovery.handlers import (
    DEFAULT_CLUSTERING_INSTRUCTION,
    AnnotationNote,
    ClusterForDistillation,
    ClusteringInput,
    ClusteringOutput,
    DistillationHandler,
    DistillationInput,
    DistillationOutput,
    EvidenceClusteringHandler,
    EvidenceClusteringInput,
    EvidenceNote,
    LabelInput,
    LabelOutput,
    LabelRefinementHandler,
    LearningArtifactOutput,
    PatternCategory,
    PatternClusteringHandler,
)

# Models (new)
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    EvidenceItem,
    LearningArtifact,
    PatternDiscoveryResult,
    PipelineResult,
    Provenance,
)

# Pipeline (new)
from axion.caliber.pattern_discovery.pipeline import (
    ArtifactWriter,
    EvidenceClusterer,
    EvidencePipeline,
)

# Plugins (new)
from axion.caliber.pattern_discovery.plugins import (
    ArtifactSink,
    Deduper,
    EmbeddingDeduper,
    InMemoryDeduper,
    InMemorySink,
    JsonlSink,
    NoopSanitizer,
    Sanitizer,
)

__all__ = [
    # Models — existing
    'ClusteringMethod',
    'DiscoveredPattern',
    'PatternDiscoveryResult',
    # Models — new
    'EvidenceItem',
    'LearningArtifact',
    'PipelineResult',
    'Provenance',
    # Compat
    'AnnotatedItem',
    'annotated_item_to_evidence',
    'annotations_to_evidence',
    'normalize_annotations',
    # Pydantic I/O — existing
    'AnnotationNote',
    'ClusteringInput',
    'ClusteringOutput',
    'DEFAULT_CLUSTERING_INSTRUCTION',
    'LabelInput',
    'LabelOutput',
    'PatternCategory',
    # Pydantic I/O — new
    'ClusterForDistillation',
    'DistillationHandler',
    'DistillationInput',
    'DistillationOutput',
    'EvidenceClusteringHandler',
    'EvidenceClusteringInput',
    'EvidenceNote',
    'LearningArtifactOutput',
    # Handler classes
    'LabelRefinementHandler',
    'PatternClusteringHandler',
    # Discovery
    'PatternDiscovery',
    # Pipeline
    'ArtifactWriter',
    'EvidenceClusterer',
    'EvidencePipeline',
    # Plugins
    'ArtifactSink',
    'Deduper',
    'EmbeddingDeduper',
    'InMemoryDeduper',
    'InMemorySink',
    'JsonlSink',
    'NoopSanitizer',
    'Sanitizer',
    # Config
    'MetadataConfig',
]
