from dataclasses import dataclass
from typing import Dict, Optional, Union

from axion.caliber.pattern_discovery.models import EvidenceItem


@dataclass
class AnnotatedItem:
    """A single annotated item with optional notes."""

    record_id: str
    score: int  # Binary: 0 or 1
    notes: Optional[str] = None
    timestamp: Optional[str] = None
    query: Optional[str] = None
    actual_output: Optional[str] = None


def normalize_annotations(
    annotations: Union[Dict[str, AnnotatedItem], Dict[str, Dict]],
) -> Dict[str, AnnotatedItem]:
    """Convert dict annotations to AnnotatedItem instances."""
    result: Dict[str, AnnotatedItem] = {}
    for record_id, item in annotations.items():
        if isinstance(item, AnnotatedItem):
            result[record_id] = item
        elif isinstance(item, dict):
            result[record_id] = AnnotatedItem(
                record_id=item.get('record_id', record_id),
                score=item.get('score', 0),
                notes=item.get('notes'),
                timestamp=item.get('timestamp'),
                query=item.get('query'),
                actual_output=item.get('actual_output'),
            )
        else:
            raise TypeError(f'Invalid annotation type: {type(item)}')
    return result


def annotated_item_to_evidence(item: AnnotatedItem) -> EvidenceItem:
    """Convert a single AnnotatedItem to an EvidenceItem (1:1)."""
    metadata = {
        key: value
        for key, value in {
            'score': item.score,
            'timestamp': item.timestamp,
            'query': item.query,
            'actual_output': item.actual_output,
        }.items()
        if value is not None
    }

    return EvidenceItem(
        id=item.record_id,
        text=item.notes or '',
        metadata=metadata,
    )


def annotations_to_evidence(
    items: Dict[str, AnnotatedItem],
) -> Dict[str, EvidenceItem]:
    """Bulk convert AnnotatedItems to EvidenceItems, filtering out items with no notes."""
    return {
        rid: annotated_item_to_evidence(item)
        for rid, item in items.items()
        if item.notes
    }
