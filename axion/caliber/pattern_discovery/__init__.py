from axion.caliber.pattern_discovery._compat import (
    AnnotatedItem,
    annotated_item_to_evidence,
    annotations_to_evidence,
    normalize_annotations,
)
from axion.caliber.pattern_discovery._utils import MetadataConfig
from axion.caliber.pattern_discovery.display import (
    display_learnings,
    display_patterns,
    display_pipeline_result,
)
from axion.caliber.pattern_discovery.discovery import PatternDiscovery
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
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    EvidenceItem,
    LearningArtifact,
    PatternDiscoveryResult,
    PipelineResult,
    Provenance,
)
from axion.caliber.pattern_discovery.pipeline import (
    ArtifactWriter,
    EvidenceClusterer,
    EvidencePipeline,
)
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
    'ClusteringMethod',
    'DiscoveredPattern',
    'PatternDiscoveryResult',
    'EvidenceItem',
    'LearningArtifact',
    'PipelineResult',
    'Provenance',
    'AnnotatedItem',
    'annotated_item_to_evidence',
    'annotations_to_evidence',
    'normalize_annotations',
    'AnnotationNote',
    'ClusteringInput',
    'ClusteringOutput',
    'DEFAULT_CLUSTERING_INSTRUCTION',
    'LabelInput',
    'LabelOutput',
    'PatternCategory',
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
    # Display
    'display_learnings',
    'display_patterns',
    'display_pipeline_result',
]
