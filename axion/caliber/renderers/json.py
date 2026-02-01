"""
JSON renderer for CaliberHQ workflow.

Provides minimal rendering for API/web environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from axion.caliber.renderers.base import CaliberRenderer
from axion.caliber.schema import (
    Annotation,
    AnnotationState,
    EvaluationResult,
    UploadedRecord,
)

if TYPE_CHECKING:
    from axion.caliber.analysis import MisalignmentAnalysis


class JsonCaliberRenderer(CaliberRenderer):
    """
    JSON-first renderer for CaliberHQ.

    Provides no-op rendering for API environments where data is consumed
    as JSON rather than displayed.
    """

    def render_record(
        self,
        record: UploadedRecord,
        annotation: Optional[Annotation] = None,
    ) -> None:
        """No-op: JSON renderer does not display records."""
        pass

    def render_annotation_progress(self, state: AnnotationState) -> None:
        """No-op: JSON renderer does not display progress."""
        pass

    def render_evaluation_result(self, result: EvaluationResult) -> None:
        """No-op: JSON renderer does not display results."""
        pass

    def render_misalignment_analysis(self, analysis: 'MisalignmentAnalysis') -> None:
        """No-op: JSON renderer does not display analysis."""
        pass
