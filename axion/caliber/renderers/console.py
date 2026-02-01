"""
Console renderer for CaliberHQ workflow.

Provides text-based rendering for terminal/CLI environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from axion.caliber.models import (
    Annotation,
    AnnotationState,
    EvaluationResult,
    UploadedRecord,
)
from axion.caliber.renderers.base import CaliberRenderer

if TYPE_CHECKING:
    from axion.caliber.analysis import MisalignmentAnalysis


class ConsoleCaliberRenderer(CaliberRenderer):
    """
    Console/terminal renderer for CaliberHQ.

    Provides text-based output suitable for CLI environments.
    """

    def render_record(
        self,
        record: UploadedRecord,
        annotation: Optional[Annotation] = None,
    ) -> None:
        """Render a single record for annotation."""
        print(f'\n📝 Record: {record.id}')
        print('-' * 50)
        print(f'  Query: {record.query}')
        print(f'  Actual Output: {record.actual_output}')
        if record.expected_output:
            print(f'  Expected Output: {record.expected_output}')

        if annotation:
            score_icon = '✅' if annotation.score == 1 else '❌'
            print(f'\n  Current Annotation: {score_icon} Score: {annotation.score}')
            if annotation.notes:
                print(f'  Notes: {annotation.notes}')

    def render_annotation_progress(self, state: AnnotationState) -> None:
        """Render annotation progress."""
        completed = state.completed_count
        total = state.total_records
        progress_pct = state.progress * 100

        bar_width = 30
        filled = int(bar_width * state.progress)
        bar = '█' * filled + '░' * (bar_width - filled)

        print('\n🚀 Annotation Progress')
        print(f'  [{bar}] {progress_pct:.1f}%')
        print(f'  {completed} / {total} records annotated')

    def render_evaluation_result(self, result: EvaluationResult) -> None:
        """Render evaluation results."""
        print('\n' + '=' * 50)
        print('📊 Calibration Evaluation Results')
        print('=' * 50)

        # Summary metrics
        metrics = result.metrics
        print('\n📈 Summary Metrics:')
        print(f'  🎯 Accuracy (Alignment): {metrics.accuracy:.1%}')
        print(f'  ✅ Precision: {metrics.precision:.1%}')
        print(f'  📊 Recall: {metrics.recall:.1%}')
        print(f'  📈 F1 Score: {metrics.f1_score:.1%}')
        print(f"  🤝 Cohen's Kappa: {metrics.cohen_kappa:.3f}")
        print(f'  🎚️ Specificity: {metrics.specificity:.1%}')

        # Confusion matrix
        cm = result.confusion_matrix
        print('\n📋 Confusion Matrix:')
        print('              Human=0  Human=1')
        print(
            f'  LLM=0        {cm["LLM=0"]["Human=0"]:5d}    {cm["LLM=0"]["Human=1"]:5d}'
        )
        print(
            f'  LLM=1        {cm["LLM=1"]["Human=0"]:5d}    {cm["LLM=1"]["Human=1"]:5d}'
        )

        # Count stats
        print('\n📊 Counts:')
        print(f'  True Positives: {metrics.true_positives}')
        print(f'  True Negatives: {metrics.true_negatives}')
        print(f'  False Positives: {metrics.false_positives}')
        print(f'  False Negatives: {metrics.false_negatives}')

        # Summary by alignment
        aligned = sum(1 for r in result.records if r.aligned)
        misaligned = len(result.records) - aligned
        print(f'\n  ✅ Aligned: {aligned}')
        print(f'  ❌ Misaligned: {misaligned}')

    def render_misalignment_analysis(self, analysis: 'MisalignmentAnalysis') -> None:
        """Render misalignment analysis."""
        print('\n' + '=' * 50)
        print('🔍 Misalignment Analysis')
        print('=' * 50)

        print('\n📊 Summary:')
        print(f'  Total Misaligned: {analysis.total_misaligned}')
        print(f'  False Positives (LLM too lenient): {analysis.false_positives}')
        print(f'  False Negatives (LLM too strict): {analysis.false_negatives}')

        print(f'\n📝 Analysis: {analysis.summary}')

        if analysis.patterns:
            print('\n🔍 Discovered Patterns:')
            for i, pattern in enumerate(analysis.patterns, 1):
                print(f'  {i}. [{pattern.pattern_type}]')
                print(f'     {pattern.description}')
                print(f'     Count: {pattern.count}')

        if analysis.recommendations:
            print('\n💡 Recommendations:')
            for i, rec in enumerate(analysis.recommendations, 1):
                print(f'  {i}. {rec}')
