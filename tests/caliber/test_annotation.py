"""Tests for caliber annotation manager."""

import pytest

from axion.caliber.annotation import AnnotationManager
from axion.caliber.models import UploadedRecord


class TestAnnotationManager:
    """Tests for AnnotationManager."""

    @pytest.fixture
    def sample_records(self):
        """Create sample records."""
        return [
            UploadedRecord(id='r1', query='Q1', actual_output='A1'),
            UploadedRecord(id='r2', query='Q2', actual_output='A2'),
            UploadedRecord(id='r3', query='Q3', actual_output='A3'),
        ]

    @pytest.fixture
    def manager(self, sample_records):
        """Create annotation manager."""
        return AnnotationManager(sample_records)

    def test_init(self, manager, sample_records):
        """Test initialization."""
        assert manager.total_records == 3
        assert manager.completed_count == 0
        assert manager.progress == 0.0

    def test_annotate(self, manager):
        """Test adding annotation."""
        annotation = manager.annotate('r1', score=1, notes='Good')
        assert annotation.record_id == 'r1'
        assert annotation.score == 1
        assert annotation.notes == 'Good'
        assert manager.completed_count == 1

    def test_annotate_invalid_record(self, manager):
        """Test annotating non-existent record."""
        with pytest.raises(ValueError):
            manager.annotate('invalid', score=1)

    def test_annotate_invalid_score(self, manager):
        """Test annotating with invalid score."""
        with pytest.raises(ValueError):
            manager.annotate('r1', score=5)

    def test_get_annotation(self, manager):
        """Test getting annotation."""
        manager.annotate('r1', score=1)
        annotation = manager.get_annotation('r1')
        assert annotation is not None
        assert annotation.score == 1

        assert manager.get_annotation('r2') is None

    def test_get_state(self, manager):
        """Test getting state."""
        manager.annotate('r1', score=1)
        state = manager.get_state()
        assert state.completed_count == 1
        assert state.total_records == 3
        assert 'r1' in state.annotations

    def test_get_record(self, manager):
        """Test getting record by index."""
        record = manager.get_record(0)
        assert record is not None
        assert record.id == 'r1'

        assert manager.get_record(99) is None

    def test_get_record_by_id(self, manager):
        """Test getting record by ID."""
        record = manager.get_record_by_id('r2')
        assert record is not None
        assert record.query == 'Q2'

        assert manager.get_record_by_id('invalid') is None

    def test_get_current_record(self, manager):
        """Test getting current record."""
        current = manager.get_current_record()
        assert current.id == 'r1'

    def test_get_next_unannotated(self, manager):
        """Test getting next unannotated record."""
        manager.annotate('r1', score=1)
        next_record = manager.get_next_unannotated()
        assert next_record.id == 'r2'

    def test_is_complete(self, manager):
        """Test completion check."""
        assert not manager.is_complete()

        manager.annotate('r1', score=1)
        manager.annotate('r2', score=0)
        manager.annotate('r3', score=1)

        assert manager.is_complete()

    def test_get_annotations_dict(self, manager):
        """Test getting annotations as simple dict."""
        manager.annotate('r1', score=1)
        manager.annotate('r2', score=0)

        annotations_dict = manager.get_annotations_dict()
        assert annotations_dict == {'r1': 1, 'r2': 0}

    def test_serialization(self, manager):
        """Test to_dict and from_dict."""
        manager.annotate('r1', score=1, notes='Test')
        manager.annotate('r2', score=0)

        data = manager.to_dict()
        assert 'annotations' in data
        assert 'current_index' in data

        # Create new records list for restoration
        records = [
            UploadedRecord(id='r1', query='Q1', actual_output='A1'),
            UploadedRecord(id='r2', query='Q2', actual_output='A2'),
            UploadedRecord(id='r3', query='Q3', actual_output='A3'),
        ]
        restored = AnnotationManager.from_dict(data, records)

        assert restored.completed_count == 2
        assert restored.get_annotation('r1').score == 1
        assert restored.get_annotation('r1').notes == 'Test'

    def test_progress(self, manager):
        """Test progress calculation."""
        assert manager.progress == 0.0

        manager.annotate('r1', score=1)
        assert manager.progress == pytest.approx(1 / 3)

        manager.annotate('r2', score=0)
        assert manager.progress == pytest.approx(2 / 3)

        manager.annotate('r3', score=1)
        assert manager.progress == 1.0

    def test_auto_advance_index(self, manager):
        """Test that current index advances after annotating current record."""
        # Annotate first record
        manager.annotate('r1', score=1)
        # Current should now be r2
        assert manager.get_current_record().id == 'r2'
