from datetime import datetime, timezone

import pytest

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
)


class TestUploadedRecord:
    """Tests for UploadedRecord model."""

    def test_create_minimal(self):
        """Test creating record with minimal fields."""
        record = UploadedRecord(
            id='test-1',
            query='What is Python?',
            actual_output='Python is a programming language.',
        )
        assert record.id == 'test-1'
        assert record.query == 'What is Python?'
        assert record.actual_output == 'Python is a programming language.'
        assert record.expected_output is None
        assert record.llm_score is None
        assert record.metadata == {}

    def test_create_full(self):
        """Test creating record with all fields."""
        record = UploadedRecord(
            id='test-2',
            query='What is Python?',
            actual_output='Python is a programming language.',
            expected_output='Python is a high-level programming language.',
            llm_score=1,
            llm_reasoning='Accurate response',
            metadata={'source': 'test'},
        )
        assert record.llm_score == 1
        assert record.llm_reasoning == 'Accurate response'
        assert record.metadata == {'source': 'test'}


class TestAnnotation:
    """Tests for Annotation model."""

    def test_create_annotation(self):
        """Test creating annotation."""
        annotation = Annotation(record_id='test-1', score=1, notes='Good response')
        assert annotation.record_id == 'test-1'
        assert annotation.score == 1
        assert annotation.notes == 'Good response'
        assert isinstance(annotation.timestamp, datetime)

    def test_score_validation(self):
        """Test score must be 0 or 1."""
        with pytest.raises(ValueError):
            Annotation(record_id='test-1', score=2)

        with pytest.raises(ValueError):
            Annotation(record_id='test-1', score=-1)


class TestAnnotationState:
    """Tests for AnnotationState model."""

    def test_progress_calculation(self):
        """Test progress property calculation."""
        state = AnnotationState(
            annotations={
                'r1': Annotation(record_id='r1', score=1),
                'r2': Annotation(record_id='r2', score=0),
            },
            total_records=4,
        )
        assert state.completed_count == 2
        assert state.progress == 0.5

    def test_empty_progress(self):
        """Test progress with no annotations."""
        state = AnnotationState(annotations={}, total_records=0)
        assert state.progress == 0.0


class TestAlignmentMetrics:
    """Tests for AlignmentMetrics model."""

    def test_create_metrics(self):
        """Test creating metrics."""
        metrics = AlignmentMetrics(
            accuracy=0.85,
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            cohen_kappa=0.7,
            specificity=0.9,
            true_positives=18,
            true_negatives=17,
            false_positives=2,
            false_negatives=3,
        )
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.9


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_create_result(self):
        """Test creating evaluation result."""
        metrics = AlignmentMetrics(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            cohen_kappa=1.0,
            specificity=1.0,
            true_positives=5,
            true_negatives=5,
            false_positives=0,
            false_negatives=0,
        )
        config = EvaluationConfig(criteria='Test criteria')
        records = [
            EvaluationRecord(
                record_id='r1',
                human_score=1,
                llm_score=1,
                aligned=True,
                score_difference=0,
            )
        ]

        result = EvaluationResult(
            records=records,
            metrics=metrics,
            confusion_matrix={
                'LLM=0': {'Human=0': 5, 'Human=1': 0},
                'LLM=1': {'Human=0': 0, 'Human=1': 5},
            },
            config=config,
        )

        assert len(result.records) == 1
        assert result.metrics.accuracy == 1.0


class TestSessionState:
    """Tests for SessionState enum."""

    def test_session_states(self):
        """Test session state values."""
        assert SessionState.UPLOAD.value == 'upload'
        assert SessionState.ANNOTATE.value == 'annotate'
        assert SessionState.EVALUATE.value == 'evaluate'
        assert SessionState.COMPLETE.value == 'complete'


class TestCalibrationSessionData:
    """Tests for CalibrationSessionData model."""

    def test_create_session_data(self):
        """Test creating session data."""
        now = datetime.now(timezone.utc)
        data = CalibrationSessionData(
            session_id='test-session',
            state=SessionState.UPLOAD,
            created_at=now,
            updated_at=now,
        )
        assert data.session_id == 'test-session'
        assert data.state == SessionState.UPLOAD
        assert data.upload_result is None
