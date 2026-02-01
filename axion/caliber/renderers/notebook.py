"""
Notebook renderer for CaliberHQ workflow.

Provides rich rendering for Jupyter notebook environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

from axion.caliber.renderers.base import CaliberRenderer
from axion.caliber.schema import (
    Annotation,
    AnnotationState,
    EvaluationResult,
    UploadedRecord,
)

if TYPE_CHECKING:
    from axion.caliber.analysis import MisalignmentAnalysis


class NotebookCaliberRenderer(CaliberRenderer):
    """
    Notebook-specific renderer for CaliberHQ.

    Provides rich HTML and styled pandas output for Jupyter notebooks.
    """

    def render_record(
        self,
        record: UploadedRecord,
        annotation: Optional[Annotation] = None,
    ) -> None:
        """Render a single record for annotation."""
        from IPython.display import HTML, display

        status = ''
        if annotation:
            score_icon = '✅' if annotation.score == 1 else '❌'
            status = f'<p><strong>Current Annotation:</strong> {score_icon} Score: {annotation.score}</p>'
            if annotation.notes:
                status += f'<p><strong>Notes:</strong> {annotation.notes}</p>'

        html = f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
            <h4>Record: {record.id}</h4>
            <p><strong>Query:</strong> {record.query}</p>
            <p><strong>Actual Output:</strong> {record.actual_output}</p>
            {f'<p><strong>Expected Output:</strong> {record.expected_output}</p>' if record.expected_output else ''}
            {status}
        </div>
        """
        display(HTML(html))

    def render_annotation_progress(self, state: AnnotationState) -> None:
        """Render annotation progress."""
        from IPython.display import HTML, display

        progress_pct = state.progress * 100
        completed = state.completed_count
        total = state.total_records

        html = f"""
        <div style="margin: 10px 0;">
            <h4>Annotation Progress</h4>
            <div style="background: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #4CAF50, #8BC34A);
                            width: {progress_pct:.1f}%; height: 100%;"></div>
            </div>
            <p>{completed} / {total} records annotated ({progress_pct:.1f}%)</p>
        </div>
        """
        display(HTML(html))

    def render_evaluation_result(self, result: EvaluationResult) -> None:
        """Render evaluation results with styled tables."""
        from IPython.display import display

        # Summary metrics
        metrics = result.metrics
        summary_data = [
            ['🎯 Accuracy (Alignment)', f'{metrics.accuracy:.1%}'],
            ['✅ Precision', f'{metrics.precision:.1%}'],
            ['📊 Recall', f'{metrics.recall:.1%}'],
            ['📈 F1 Score', f'{metrics.f1_score:.1%}'],
            ["🤝 Cohen's Kappa", f'{metrics.cohen_kappa:.3f}'],
            ['🎚️ Specificity', f'{metrics.specificity:.1%}'],
        ]
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        summary_styled = self._style_summary(summary_df)

        print('\n' + '=' * 50)
        print('📊 Calibration Evaluation Results')
        print('=' * 50)

        display(summary_styled)

        # Confusion matrix
        cm = result.confusion_matrix
        cm_df = pd.DataFrame(
            [
                [cm['LLM=0']['Human=0'], cm['LLM=0']['Human=1']],
                [cm['LLM=1']['Human=0'], cm['LLM=1']['Human=1']],
            ],
            columns=['Human=0', 'Human=1'],
            index=['LLM=0', 'LLM=1'],
        )

        print('\nConfusion Matrix:')
        display(cm_df.style.set_caption('LLM vs Human Scores'))

        # Detailed results
        records_data = [
            {
                'ID': r.record_id,
                'Aligned': '✅' if r.aligned else '❌',
                'Human': r.human_score,
                'LLM': r.llm_score,
                'Diff': r.score_difference,
            }
            for r in result.records
        ]
        results_df = pd.DataFrame(records_data)
        results_styled = self._style_results(results_df)

        print('\nDetailed Results:')
        display(results_styled)

    def render_misalignment_analysis(self, analysis: 'MisalignmentAnalysis') -> None:
        """Render misalignment analysis."""
        from IPython.display import HTML, display

        # Summary
        html = f"""
        <div style="border: 1px solid #f0ad4e; padding: 15px; margin: 10px 0;
                    border-radius: 5px; background: #fcf8e3;">
            <h4>Misalignment Analysis</h4>
            <p><strong>Total Misaligned:</strong> {analysis.total_misaligned}</p>
            <p><strong>False Positives (LLM too lenient):</strong> {analysis.false_positives}</p>
            <p><strong>False Negatives (LLM too strict):</strong> {analysis.false_negatives}</p>
        </div>
        """
        display(HTML(html))

        # Summary text
        print(f'\n📝 Summary: {analysis.summary}')

        # Patterns
        if analysis.patterns:
            print('\n🔍 Discovered Patterns:')
            for i, pattern in enumerate(analysis.patterns, 1):
                print(
                    f'  {i}. [{pattern.pattern_type}] {pattern.description} '
                    f'(Count: {pattern.count})'
                )

        # Recommendations
        if analysis.recommendations:
            print('\n💡 Recommendations:')
            for i, rec in enumerate(analysis.recommendations, 1):
                print(f'  {i}. {rec}')

    def _style_summary(self, df: pd.DataFrame):
        """Apply styling to summary DataFrame."""
        return (
            df.style.hide(axis='index')
            .set_table_styles(
                [
                    {
                        'selector': 'table',
                        'props': [('width', '50%'), ('margin', '20px 0')],
                    },
                    {
                        'selector': 'td',
                        'props': [('padding', '10px'), ('font-size', '14px')],
                    },
                ]
            )
            .set_caption('📈 Calibration Summary')
            .set_properties(**{'text-align': 'left', 'font-weight': 'bold'})
        )

    def _style_results(self, df: pd.DataFrame):
        """Apply styling to results DataFrame."""

        def highlight_alignment(row):
            styles = pd.Series('', index=row.index)
            is_aligned = row['Aligned'] == '✅'

            if is_aligned:
                styles['Human'] = (
                    'background: linear-gradient(135deg, #10b981, #059669); '
                    'color: white; font-weight: bold;'
                )
                styles['LLM'] = styles['Human']
            else:
                styles['Human'] = (
                    'background: linear-gradient(135deg, #ef4444, #dc2626); '
                    'color: white; font-weight: bold;'
                )
                styles['LLM'] = styles['Human']

            return styles

        return (
            df.style.apply(highlight_alignment, axis=1)
            .set_table_styles(
                [
                    {
                        'selector': 'thead th',
                        'props': [
                            ('background', '#667eea'),
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('padding', '12px'),
                        ],
                    },
                    {
                        'selector': 'tbody tr:hover',
                        'props': [('background-color', '#f5f5f5')],
                    },
                    {
                        'selector': 'tbody td',
                        'props': [('padding', '10px'), ('border', '1px solid #ddd')],
                    },
                ]
            )
            .set_properties(**{'text-align': 'left'})
        )
