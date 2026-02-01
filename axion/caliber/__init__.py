"""
CaliberHQ — LLM judge calibration toolkit.

This package provides a workflow for calibrating LLM-as-a-judge evaluators
against human judgments (upload → annotate → evaluate → analyze → optimize).

Docs live in MkDocs:
- `docs/guides/caliberhq.md`
"""

# Session orchestrator
# Analysis
from axion.caliber.analysis import (
    MisalignedCase,
    MisalignmentAnalysis,
    MisalignmentAnalyzer,
    MisalignmentPattern,
)

# Step 2: Annotation
from axion.caliber.annotation import AnnotationManager

# Step 3: Evaluation
from axion.caliber.evaluation import CaliberMetric, EvaluationRunner

# Example Selection
from axion.caliber.example_selector import (
    ExampleSelector,
    SelectionResult,
    SelectionStrategy,
)

# Pattern Discovery
from axion.caliber.pattern_discovery import (
    AnnotatedItem,
    ClusteringMethod,
    DiscoveredPattern,
    PatternDiscovery,
    PatternDiscoveryResult,
)

# Prompt Optimization
from axion.caliber.prompt_optimizer import (
    OptimizedPrompt,
    PromptOptimizer,
    PromptSuggestion,
)

# Renderers
from axion.caliber.renderers import (
    CaliberRenderer,
    ConsoleCaliberRenderer,
    JsonCaliberRenderer,
    NotebookCaliberRenderer,
)

# Models
from axion.caliber.schema import (
    AlignmentMetrics,
    Annotation,
    AnnotationState,
    CalibrationSessionData,
    EvaluationConfig,
    EvaluationRecord,
    EvaluationResult,
    SessionState,
    UploadedRecord,
    UploadResult,
)
from axion.caliber.session import CalibrationSession

# Step 1: Upload
from axion.caliber.upload import UploadHandler

__all__ = [
    # Session
    'CalibrationSession',
    # Models
    'AlignmentMetrics',
    'Annotation',
    'AnnotationState',
    'CalibrationSessionData',
    'EvaluationConfig',
    'EvaluationRecord',
    'EvaluationResult',
    'SessionState',
    'UploadedRecord',
    'UploadResult',
    # Upload
    'UploadHandler',
    # Annotation
    'AnnotationManager',
    # Evaluation
    'CaliberMetric',
    'EvaluationRunner',
    # Analysis
    'MisalignedCase',
    'MisalignmentAnalysis',
    'MisalignmentAnalyzer',
    'MisalignmentPattern',
    # Prompt Optimization
    'OptimizedPrompt',
    'PromptOptimizer',
    'PromptSuggestion',
    # Pattern Discovery
    'AnnotatedItem',
    'ClusteringMethod',
    'DiscoveredPattern',
    'PatternDiscovery',
    'PatternDiscoveryResult',
    # Example Selection
    'ExampleSelector',
    'SelectionResult',
    'SelectionStrategy',
    # Renderers
    'CaliberRenderer',
    'ConsoleCaliberRenderer',
    'JsonCaliberRenderer',
    'NotebookCaliberRenderer',
]
