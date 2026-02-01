from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from axion._core.schema import StrictBaseModel


class UploadedRecord(StrictBaseModel):
    """Single record from uploaded data."""

    id: str = Field(description='Unique identifier for the record')
    query: str = Field(description='The input query or prompt')
    actual_output: str = Field(description='The LLM response being evaluated')
    expected_output: Optional[str] = Field(
        default=None, description='Optional reference/expected output'
    )
    llm_score: Optional[int] = Field(
        default=None, description='Pre-existing LLM judge score (0 or 1)'
    )
    llm_reasoning: Optional[str] = Field(
        default=None, description='Pre-existing LLM reasoning/explanation'
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description='Additional metadata'
    )


class UploadResult(StrictBaseModel):
    """Result of upload step."""

    records: List[UploadedRecord] = Field(description='Uploaded records')
    total_count: int = Field(description='Total number of records')
    has_llm_scores: bool = Field(
        description='Whether records have pre-existing LLM scores'
    )
    validation_warnings: List[str] = Field(
        default_factory=list, description='Validation warnings encountered'
    )


class Annotation(StrictBaseModel):
    """Human annotation for a single record."""

    record_id: str = Field(description='ID of the annotated record')
    score: int = Field(ge=0, le=1, description='Human score: 0=reject, 1=accept')
    notes: Optional[str] = Field(default=None, description='Optional annotation notes')
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description='When annotation was made',
    )


class AnnotationState(StrictBaseModel):
    """Current state of annotations."""

    annotations: Dict[str, Annotation] = Field(
        default_factory=dict, description='Map of record_id to Annotation'
    )
    current_index: int = Field(default=0, description='Current annotation index')
    total_records: int = Field(description='Total number of records to annotate')

    @property
    def completed_count(self) -> int:
        """Number of completed annotations."""
        return len(self.annotations)

    @property
    def progress(self) -> float:
        """Progress as a fraction (0.0 to 1.0)."""
        return (
            self.completed_count / self.total_records if self.total_records > 0 else 0.0
        )


class EvaluationConfig(StrictBaseModel):
    """Configuration for LLM evaluation."""

    model_name: str = Field(default='gpt-4o', description='LLM model name')
    llm_provider: Optional[str] = Field(
        default=None, description='LLM provider (openai, anthropic, etc.)'
    )
    criteria: str = Field(description='The LLM-as-judge criteria/instruction')
    system_prompt: Optional[str] = Field(
        default=None, description='Optional system prompt for the evaluator'
    )


class EvaluationRecord(StrictBaseModel):
    """Single evaluation result."""

    record_id: str = Field(description='ID of the evaluated record')
    human_score: int = Field(description='Human annotation score')
    llm_score: int = Field(description='LLM judge score')
    llm_reasoning: Optional[str] = Field(
        default=None, description='LLM reasoning for its judgment'
    )
    aligned: bool = Field(description='Whether human and LLM scores match')
    score_difference: int = Field(description='Absolute difference between scores')


class AlignmentMetrics(StrictBaseModel):
    """Comprehensive alignment metrics."""

    accuracy: float = Field(description='Overall accuracy/alignment score')
    precision: float = Field(description='Precision score')
    recall: float = Field(description='Recall/sensitivity score')
    f1_score: float = Field(description='F1 score')
    cohen_kappa: float = Field(description="Cohen's kappa agreement coefficient")
    specificity: float = Field(description='Specificity/true negative rate')
    true_positives: int = Field(description='Count of true positives')
    true_negatives: int = Field(description='Count of true negatives')
    false_positives: int = Field(
        description='Count of false positives (LLM too lenient)'
    )
    false_negatives: int = Field(
        description='Count of false negatives (LLM too strict)'
    )


class EvaluationResult(StrictBaseModel):
    """Complete evaluation result."""

    records: List[EvaluationRecord] = Field(description='Individual evaluation results')
    metrics: AlignmentMetrics = Field(description='Computed alignment metrics')
    confusion_matrix: Dict[str, Dict[str, int]] = Field(
        description='Confusion matrix as nested dict'
    )
    config: EvaluationConfig = Field(description='Evaluation configuration used')


class SessionState(str, Enum):
    """Session workflow states."""

    UPLOAD = 'upload'
    ANNOTATE = 'annotate'
    EVALUATE = 'evaluate'
    COMPLETE = 'complete'


class CalibrationSessionData(StrictBaseModel):
    """Serializable session state for persistence."""

    session_id: str = Field(description='Unique session identifier')
    state: SessionState = Field(description='Current workflow state')
    upload_result: Optional[UploadResult] = Field(
        default=None, description='Upload step result'
    )
    annotation_state: Optional[AnnotationState] = Field(
        default=None, description='Annotation step state'
    )
    evaluation_result: Optional[EvaluationResult] = Field(
        default=None, description='Evaluation step result'
    )
    created_at: datetime = Field(description='Session creation time')
    updated_at: datetime = Field(description='Last update time')

