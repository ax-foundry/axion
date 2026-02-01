import pytest

from axion.caliber.schema import SessionState
from axion.caliber.session import CalibrationSession


class TestCalibrationSession:
    """Tests for CalibrationSession."""

    @pytest.fixture
    def session(self):
        """Create a fresh session."""
        return CalibrationSession()

    @pytest.fixture
    def sample_records(self):
        """Sample records for upload."""
        return [
            {'id': 'r1', 'query': 'Q1', 'actual_output': 'A1'},
            {'id': 'r2', 'query': 'Q2', 'actual_output': 'A2'},
            {'id': 'r3', 'query': 'Q3', 'actual_output': 'A3'},
        ]

    def test_init(self, session):
        """Test session initialization."""
        assert session.session_id is not None
        assert session.state == SessionState.UPLOAD
        assert len(session.records) == 0

    def test_upload_records(self, session, sample_records):
        """Test uploading records."""
        result = session.upload_records(sample_records)
        assert result.total_count == 3
        assert len(session.records) == 3
        assert session.state == SessionState.ANNOTATE

    def test_annotate(self, session, sample_records):
        """Test annotating a record."""
        session.upload_records(sample_records)

        annotation = session.annotate('r1', score=1, notes='Good')
        assert annotation.record_id == 'r1'
        assert annotation.score == 1

    def test_annotate_without_upload(self, session):
        """Test that annotating without upload fails."""
        with pytest.raises(RuntimeError):
            session.annotate('r1', score=1)

    def test_get_annotation_state(self, session, sample_records):
        """Test getting annotation state."""
        session.upload_records(sample_records)
        session.annotate('r1', score=1)

        state = session.get_annotation_state()
        assert state.completed_count == 1
        assert state.total_records == 3

    def test_get_record_for_annotation(self, session, sample_records):
        """Test getting record by index."""
        session.upload_records(sample_records)

        record = session.get_record_for_annotation(0)
        assert record.id == 'r1'

        assert session.get_record_for_annotation(99) is None

    def test_get_next_unannotated(self, session, sample_records):
        """Test getting next unannotated record."""
        session.upload_records(sample_records)
        session.annotate('r1', score=1)

        next_record = session.get_next_unannotated()
        assert next_record.id == 'r2'

    def test_is_annotation_complete(self, session, sample_records):
        """Test annotation completion check."""
        session.upload_records(sample_records)

        assert not session.is_annotation_complete()

        session.annotate('r1', score=1)
        session.annotate('r2', score=0)
        session.annotate('r3', score=1)

        assert session.is_annotation_complete()

    def test_serialization(self, session, sample_records):
        """Test to_dict and from_dict."""
        session.upload_records(sample_records)
        session.annotate('r1', score=1, notes='Test note')
        session.annotate('r2', score=0)

        data = session.to_dict()
        assert 'session_id' in data
        assert 'state' in data
        assert 'upload_result' in data
        assert 'annotation_state' in data

        restored = CalibrationSession.from_dict(data)
        assert restored.session_id == session.session_id
        assert restored.state == session.state
        assert len(restored.records) == 3

        # Check annotations were restored
        annotation_state = restored.get_annotation_state()
        assert annotation_state.completed_count == 2

    def test_custom_session_id(self):
        """Test creating session with custom ID."""
        session = CalibrationSession(session_id='custom-id')
        assert session.session_id == 'custom-id'

    def test_select_examples(self, session, sample_records):
        """Test example selection."""
        session.upload_records(sample_records)
        session.annotate('r1', score=1)
        session.annotate('r2', score=0)
        session.annotate('r3', score=1)

        result = session.select_examples(count=2, seed=42)
        assert len(result.examples) == 2
