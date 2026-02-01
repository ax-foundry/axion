"""
Abstract base renderer for CaliberHQ workflow.

Defines the interface for UI rendering implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from axion.caliber.schema import (
    Annotation,
    AnnotationState,
    EvaluationResult,
    UploadedRecord,
)

if TYPE_CHECKING:
    from axion.caliber.analysis import MisalignmentAnalysis


class CaliberRenderer(ABC):
    """
    Abstract renderer interface for CaliberHQ.

    Implementations provide UI rendering for different environments:
    - NotebookCaliberRenderer: Jupyter notebooks with rich styling
    - ConsoleCaliberRenderer: Terminal/CLI output
    - JsonCaliberRenderer: JSON-only (for web APIs)
    """

    @abstractmethod
    def render_record(
        self,
        record: UploadedRecord,
        annotation: Optional[Annotation] = None,
    ) -> None:
        """
        Render a single record for annotation.

        Args:
            record: The record to display
            annotation: Existing annotation if any
        """

    @abstractmethod
    def render_annotation_progress(self, state: AnnotationState) -> None:
        """
        Render annotation progress.

        Args:
            state: Current annotation state
        """

    @abstractmethod
    def render_evaluation_result(self, result: EvaluationResult) -> None:
        """
        Render evaluation results.

        Args:
            result: Evaluation result to display
        """

    @abstractmethod
    def render_misalignment_analysis(self, analysis: 'MisalignmentAnalysis') -> None:
        """
        Render misalignment analysis.

        Args:
            analysis: Analysis result to display
        """
