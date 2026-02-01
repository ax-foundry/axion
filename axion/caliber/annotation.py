"""
Annotation management for CaliberHQ workflow (Step 2).

Manages human annotation state and provides navigation through records.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from axion.caliber.models import Annotation, AnnotationState, UploadedRecord


class AnnotationManager:
    """
    Manages annotation state for the CaliberHQ workflow.

    Tracks which records have been annotated, provides navigation,
    and maintains annotation history.

    Example:
        >>> records = [UploadedRecord(id="1", query="...", actual_output="...")]
        >>> manager = AnnotationManager(records)
        >>>
        >>> # Annotate a record
        >>> manager.annotate("1", score=1, notes="Good response")
        >>>
        >>> # Check progress
        >>> state = manager.get_state()
        >>> print(f"Progress: {state.progress:.0%}")
    """

    def __init__(self, records: List[UploadedRecord]):
        """
        Initialize the annotation manager.

        Args:
            records: List of records to annotate
        """
        self._records: Dict[str, UploadedRecord] = {r.id: r for r in records}
        self._record_ids: List[str] = [r.id for r in records]
        self._annotations: Dict[str, Annotation] = {}
        self._current_index: int = 0

    @property
    def total_records(self) -> int:
        """Total number of records to annotate."""
        return len(self._records)

    @property
    def completed_count(self) -> int:
        """Number of completed annotations."""
        return len(self._annotations)

    @property
    def progress(self) -> float:
        """Progress as a fraction (0.0 to 1.0)."""
        return (
            self.completed_count / self.total_records if self.total_records > 0 else 0.0
        )

    def annotate(
        self,
        record_id: str,
        score: int,
        notes: Optional[str] = None,
    ) -> Annotation:
        """
        Add or update an annotation for a record.

        Args:
            record_id: ID of the record to annotate
            score: Human score (0 for reject, 1 for accept)
            notes: Optional annotation notes

        Returns:
            The created/updated Annotation

        Raises:
            ValueError: If record_id is not found or score is invalid
        """
        if record_id not in self._records:
            raise ValueError(f"Record '{record_id}' not found")

        if score not in (0, 1):
            raise ValueError(f'Score must be 0 or 1, got {score}')

        annotation = Annotation(
            record_id=record_id,
            score=score,
            notes=notes,
            timestamp=datetime.now(timezone.utc),
        )
        self._annotations[record_id] = annotation

        # Auto-advance current index if this was the current record
        if (
            self._current_index < len(self._record_ids)
            and self._record_ids[self._current_index] == record_id
        ):
            self._advance_to_next_unannotated()

        return annotation

    def get_annotation(self, record_id: str) -> Optional[Annotation]:
        """
        Get annotation for a specific record.

        Args:
            record_id: ID of the record

        Returns:
            Annotation if exists, None otherwise
        """
        return self._annotations.get(record_id)

    def get_state(self) -> AnnotationState:
        """
        Get the current annotation state.

        Returns:
            AnnotationState with current progress and annotations
        """
        return AnnotationState(
            annotations=self._annotations.copy(),
            current_index=self._current_index,
            total_records=self.total_records,
        )

    def get_record(self, index: int) -> Optional[UploadedRecord]:
        """
        Get a record by index.

        Args:
            index: Index of the record (0-based)

        Returns:
            UploadedRecord if index is valid, None otherwise
        """
        if 0 <= index < len(self._record_ids):
            record_id = self._record_ids[index]
            return self._records.get(record_id)
        return None

    def get_record_by_id(self, record_id: str) -> Optional[UploadedRecord]:
        """
        Get a record by its ID.

        Args:
            record_id: ID of the record

        Returns:
            UploadedRecord if found, None otherwise
        """
        return self._records.get(record_id)

    def get_current_record(self) -> Optional[UploadedRecord]:
        """
        Get the current record for annotation.

        Returns:
            Current UploadedRecord or None if all annotated
        """
        return self.get_record(self._current_index)

    def get_next_unannotated(self) -> Optional[UploadedRecord]:
        """
        Get the next unannotated record.

        Returns:
            Next unannotated record or None if all annotated
        """
        for record_id in self._record_ids:
            if record_id not in self._annotations:
                return self._records[record_id]
        return None

    def set_current_index(self, index: int) -> None:
        """
        Set the current annotation index.

        Args:
            index: New current index

        Raises:
            ValueError: If index is out of range
        """
        if not 0 <= index < len(self._record_ids):
            raise ValueError(f'Index {index} out of range [0, {len(self._record_ids)})')
        self._current_index = index

    def is_complete(self) -> bool:
        """
        Check if all records have been annotated.

        Returns:
            True if all records are annotated
        """
        return self.completed_count == self.total_records

    def get_annotations_dict(self) -> Dict[str, int]:
        """
        Get annotations as a simple dict mapping record_id to score.

        Returns:
            Dict mapping record_id to score
        """
        return {record_id: ann.score for record_id, ann in self._annotations.items()}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the manager state to a dictionary.

        Returns:
            Dict representation of the annotation state
        """
        return {
            'annotations': {
                rid: ann.model_dump() for rid, ann in self._annotations.items()
            },
            'current_index': self._current_index,
            'total_records': self.total_records,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], records: List[UploadedRecord]
    ) -> 'AnnotationManager':
        """
        Restore an annotation manager from a dictionary.

        Args:
            data: Serialized state from to_dict()
            records: Original list of records

        Returns:
            Restored AnnotationManager instance
        """
        manager = cls(records)

        # Restore annotations
        for rid, ann_data in data.get('annotations', {}).items():
            manager._annotations[rid] = Annotation(**ann_data)

        # Restore index
        manager._current_index = data.get('current_index', 0)

        return manager

    def _advance_to_next_unannotated(self) -> None:
        """Advance current_index to the next unannotated record."""
        for i in range(self._current_index + 1, len(self._record_ids)):
            if self._record_ids[i] not in self._annotations:
                self._current_index = i
                return
        # All remaining are annotated, stay at end
        self._current_index = len(self._record_ids)
