"""
CaliberHQ - LLM Judge Calibration Toolkit.

Calibrate your LLM-as-judge evaluators to align with human judgment through a
structured 6-step workflow.

================================================================================
QUICK START
================================================================================

    from axion.caliber import CalibrationSession

    session = CalibrationSession()

    # 1. Upload your data
    session.upload_csv("evaluation_data.csv")

    # 2. Add human annotations
    session.annotate("record_1", score=1, notes="Good response")
    session.annotate("record_2", score=0, notes="Factually incorrect")

    # 3. Run LLM evaluation
    result = await session.evaluate(
        criteria="Score 1 if accurate and helpful, 0 otherwise",
        model_name="gpt-4o"
    )

    # 4. View alignment metrics
    print(f"Accuracy: {result.metrics.accuracy:.1%}")
    print(f"Cohen's Kappa: {result.metrics.cohen_kappa:.3f}")

================================================================================
THE 6-STEP WORKFLOW
================================================================================

Step 1: UPLOAD - Load your evaluation data
    >>> session.upload_csv("data.csv")
    >>> # or
    >>> session.upload_records([{"id": "1", "query": "...", "actual_output": "..."}])

Step 2: ANNOTATE - Add human judgments (Accept=1, Reject=0)
    >>> for record in session.records:
    ...     session.annotate(record.id, score=1, notes="Why this is good/bad")

Step 3: EVALUATE - Run LLM judge and compute alignment metrics
    >>> result = await session.evaluate(criteria="Your evaluation criteria here")
    >>> print(f"Accuracy: {result.metrics.accuracy:.1%}")

Step 4: DISCOVER PATTERNS - Find themes in your annotations
    >>> patterns = await session.discover_patterns()
    >>> for p in patterns.patterns:
    ...     print(f"{p.category}: {p.count} records")

Step 5: ANALYZE MISALIGNMENTS - Understand where LLM and humans disagree
    >>> analysis = await session.analyze_misalignments()
    >>> print(f"False positives: {analysis.false_positives}")
    >>> print(f"False negatives: {analysis.false_negatives}")

Step 6: OPTIMIZE - Improve your evaluation criteria
    >>> optimized = await session.optimize_prompt()
    >>> print(optimized.optimized_criteria)

================================================================================
TWO USAGE PATTERNS
================================================================================

Pattern 1: Session-Based (Recommended for most use cases)
    Use CalibrationSession for state management and serialization.
    Great for web APIs, scripts, and simple notebooks.

    >>> from axion.caliber import CalibrationSession
    >>> session = CalibrationSession()
    >>> session.upload_csv("data.csv")
    >>> session.annotate("r1", score=1, notes="Good")
    >>> result = await session.evaluate(criteria="...")

Pattern 2: Direct Components (For advanced customization)
    Use individual components for fine-grained control.
    Great for complex notebooks and custom pipelines.

    >>> from axion.caliber import UploadHandler, AnnotationManager, EvaluationRunner
    >>> records = UploadHandler().from_csv("data.csv")
    >>> manager = AnnotationManager(records.records)
    >>> manager.annotate("r1", score=1, notes="Good")
    >>> result = await EvaluationRunner(config).run(records.records, manager.get_annotations_dict())

================================================================================
KEY COMPONENTS
================================================================================

Core Session:
    CalibrationSession      Central orchestrator for the full workflow

Data Models:
    UploadedRecord          A single record to evaluate
    Annotation              Human judgment (score + notes)
    EvaluationResult        Complete evaluation output with metrics
    AlignmentMetrics        Accuracy, precision, recall, F1, Cohen's kappa

Step Components:
    UploadHandler           Load data from CSV, DataFrame, or dicts
    AnnotationManager       Track annotation state and progress
    EvaluationRunner        Run LLM-as-judge evaluation

Analysis Tools:
    PatternDiscovery        Discover themes in annotations (LLM/BERTopic)
    MisalignmentAnalyzer    Analyze false positives/negatives
    PromptOptimizer         Generate improved evaluation criteria
    ExampleSelector         Select balanced examples for few-shot prompts

Renderers:
    ConsoleCaliberRenderer  Terminal output
    NotebookCaliberRenderer Jupyter notebook with styled HTML
    JsonCaliberRenderer     JSON output for web APIs

================================================================================
DEMO
================================================================================

Run the demo script to see the full workflow in action:

    # Basic demo (no API key needed)
    python examples/caliber_demo.py

    # Full end-to-end with LLM calls
    OPENAI_API_KEY=your-key python examples/caliber_demo.py --full

================================================================================
"""

# Session orchestrator
from axion.caliber.session import CalibrationSession

# Models
from axion.caliber.models import (
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

# Step 1: Upload
from axion.caliber.upload import UploadHandler

# Step 2: Annotation
from axion.caliber.annotation import AnnotationManager

# Step 3: Evaluation
from axion.caliber.evaluation import CaliberMetric, EvaluationRunner

# Analysis
from axion.caliber.analysis import (
    MisalignedCase,
    MisalignmentAnalysis,
    MisalignmentAnalyzer,
    MisalignmentPattern,
)

# Prompt Optimization
from axion.caliber.prompt_optimizer import (
    OptimizedPrompt,
    PromptOptimizer,
    PromptSuggestion,
)

# Pattern Discovery
from axion.caliber.pattern_discovery import (
    AnnotatedItem,
    ClusteringMethod,
    DiscoveredPattern,
    PatternDiscovery,
    PatternDiscoveryResult,
)

# Example Selection
from axion.caliber.example_selector import (
    ExampleSelector,
    SelectionResult,
    SelectionStrategy,
)

# Renderers
from axion.caliber.renderers import (
    CaliberRenderer,
    ConsoleCaliberRenderer,
    JsonCaliberRenderer,
    NotebookCaliberRenderer,
)

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
