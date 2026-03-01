"""Display functions for pattern discovery results.

Standalone functions with auto-detection of notebook vs console environment.
Follows the existing ``display_*`` pattern (see ``axion/display.py``).
"""

from __future__ import annotations

from typing import List

from axion.caliber.pattern_discovery.models import (
    LearningArtifact,
    PatternDiscoveryResult,
    PipelineResult,
)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _is_notebook() -> bool:
    """Return *True* when running inside a Jupyter/IPython notebook."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _display_patterns_console(result: PatternDiscoveryResult, *, header: bool = True) -> None:
    if header:
        print('\n' + '=' * 55)
        print('🔬 Pattern Discovery Results')
        print('=' * 55)

        print(f'\n  Method: {result.method.value}')
        print(f'  Total analyzed: {result.total_analyzed}')
        print(f'  Patterns found: {len(result.patterns)}')
        print(f'  Uncategorized: {len(result.uncategorized)}')

    if not result.patterns:
        print('\n  No patterns discovered.')
        return

    print('\n' + '-' * 55)
    for i, p in enumerate(result.patterns, 1):
        conf = f' ({p.confidence:.0%} confidence)' if p.confidence is not None else ''
        print(f'\n  📌 {i}. {p.category}{conf}')
        print(f'     {p.description}')
        print(f'     Count: {p.count}')
        if p.examples:
            print('     Examples:')
            for ex in p.examples[:3]:
                print(f'       - {ex}')


def _display_learnings_console(learnings: List[LearningArtifact]) -> None:
    print('\n' + '=' * 55)
    print('💡 Learning Artifacts')
    print('=' * 55)

    if not learnings:
        print('\n  No learnings generated.')
        return

    for i, la in enumerate(learnings, 1):
        print(f'\n  {"─" * 50}')
        print(f'  📘 {i}. {la.title}  (confidence {la.confidence:.0%})')
        print(f'     {la.content}')

        if la.tags:
            print(f'     Tags: {", ".join(la.tags)}')
        if la.scope:
            print(f'     Scope: {la.scope}')
        if la.recommended_actions:
            print('     Recommended actions:')
            for action in la.recommended_actions:
                print(f'       • {action}')
        if la.counterexamples:
            print('     Counterexamples:')
            for ce in la.counterexamples:
                print(f'       ✗ {ce}')
        if la.when_not_to_apply:
            print(f'     When NOT to apply: {la.when_not_to_apply}')
        print(f'     Supporting items: {len(la.supporting_item_ids)}')


def _display_pipeline_console(result: PipelineResult) -> None:
    print('\n' + '=' * 55)
    print('🚀 Evidence Pipeline Results')
    print('=' * 55)

    print(f'\n  Filtered: {result.filtered_count}')
    print(f'  Deduplicated: {result.deduplicated_count}')
    print(f'  Validation repairs: {result.validation_repairs}')
    if result.sink_ids:
        print(f'  Artifacts written: {len(result.sink_ids)}')

    _display_patterns_console(result.clustering_result)
    _display_learnings_console(result.learnings)


# ---------------------------------------------------------------------------
# Notebook helpers
# ---------------------------------------------------------------------------

def _build_summary_card_html(
    title: str,
    rows: List[tuple[str, str]],
    *,
    border_color: str = '#667eea',
    bg_color: str = '#f8f9ff',
) -> str:
    row_html = ''.join(
        f'<tr><td style="padding:6px 12px;font-weight:bold;">{k}</td>'
        f'<td style="padding:6px 12px;">{v}</td></tr>'
        for k, v in rows
    )
    return (
        f'<div style="border-left:4px solid {border_color};background:{bg_color};'
        f'padding:14px 18px;margin:10px 0;border-radius:6px;">'
        f'<h4 style="margin:0 0 8px;">{title}</h4>'
        f'<table style="border-collapse:collapse;">{row_html}</table></div>'
    )


def _display_patterns_notebook(result: PatternDiscoveryResult, *, header: bool = True) -> None:
    from IPython.display import HTML, display
    import pandas as pd

    if header:
        summary = _build_summary_card_html(
            '🔬 Pattern Discovery',
            [
                ('Method', result.method.value),
                ('Total analyzed', str(result.total_analyzed)),
                ('Patterns found', str(len(result.patterns))),
                ('Uncategorized', str(len(result.uncategorized))),
            ],
        )
        display(HTML(summary))

    if not result.patterns:
        display(HTML('<p style="color:#888;"><em>No patterns discovered.</em></p>'))
        return

    rows = []
    for p in result.patterns:
        rows.append({
            'Category': p.category,
            'Description': p.description,
            'Count': p.count,
            'Confidence': f'{p.confidence:.0%}' if p.confidence is not None else '—',
            'Examples': '; '.join(p.examples[:3]) if p.examples else '—',
        })

    df = pd.DataFrame(rows)
    styled = (
        df.style.hide(axis='index')
        .set_table_styles([
            {
                'selector': 'thead th',
                'props': [
                    ('background', '#667eea'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('padding', '10px'),
                ],
            },
            {
                'selector': 'tbody td',
                'props': [('padding', '8px'), ('border', '1px solid #ddd')],
            },
            {
                'selector': 'tbody tr:hover',
                'props': [('background-color', '#f5f5f5')],
            },
        ])
        .set_properties(**{'text-align': 'left'})
        .set_caption('Discovered Patterns')
    )
    display(styled)


def _display_learnings_notebook(learnings: List[LearningArtifact]) -> None:
    from IPython.display import HTML, display

    if not learnings:
        display(HTML('<p style="color:#888;"><em>No learnings generated.</em></p>'))
        return

    cards: list[str] = []
    for la in learnings:
        tags_html = ''.join(
            f'<span style="display:inline-block;background:#e0e7ff;color:#4338ca;'
            f'padding:2px 8px;border-radius:12px;font-size:12px;margin-right:4px;">'
            f'{t}</span>'
            for t in la.tags
        )
        actions_html = ''
        if la.recommended_actions:
            items = ''.join(f'<li>{a}</li>' for a in la.recommended_actions)
            actions_html = f'<p style="margin:4px 0 2px;"><strong>Recommended actions:</strong></p><ul style="margin:0;">{items}</ul>'

        counter_html = ''
        if la.counterexamples:
            items = ''.join(f'<li>{c}</li>' for c in la.counterexamples)
            counter_html = f'<p style="margin:4px 0 2px;"><strong>Counterexamples:</strong></p><ul style="margin:0;">{items}</ul>'

        scope_html = f'<p><strong>Scope:</strong> {la.scope}</p>' if la.scope else ''
        not_apply_html = (
            f'<p><strong>When NOT to apply:</strong> {la.when_not_to_apply}</p>'
            if la.when_not_to_apply
            else ''
        )

        card = (
            f'<div style="border:1px solid #ddd;padding:14px 18px;margin:8px 0;border-radius:6px;">'
            f'<h4 style="margin:0 0 6px;">📘 {la.title}'
            f'<span style="float:right;font-size:13px;color:#666;">'
            f'{la.confidence:.0%} confidence</span></h4>'
            f'<p>{la.content}</p>'
            f'<div>{tags_html}</div>'
            f'{scope_html}{actions_html}{counter_html}{not_apply_html}'
            f'<p style="font-size:12px;color:#888;">Supporting items: {len(la.supporting_item_ids)}</p>'
            f'</div>'
        )
        cards.append(card)

    display(HTML(
        '<div style="margin:10px 0;">'
        '<h4>💡 Learning Artifacts</h4>'
        + ''.join(cards)
        + '</div>'
    ))


def _display_pipeline_notebook(result: PipelineResult) -> None:
    from IPython.display import HTML, display

    rows = [
        ('Filtered', str(result.filtered_count)),
        ('Deduplicated', str(result.deduplicated_count)),
        ('Validation repairs', str(result.validation_repairs)),
    ]
    if result.sink_ids:
        rows.append(('Artifacts written', str(len(result.sink_ids))))

    summary = _build_summary_card_html('🚀 Evidence Pipeline', rows)
    display(HTML(summary))

    _display_patterns_notebook(result.clustering_result)
    _display_learnings_notebook(result.learnings)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def display_pipeline_result(result: PipelineResult) -> None:
    """Display a full pipeline result (summary + patterns + learnings).

    Auto-detects Jupyter notebook vs console environment.
    """
    if _is_notebook():
        _display_pipeline_notebook(result)
    else:
        _display_pipeline_console(result)


def display_patterns(result: PatternDiscoveryResult) -> None:
    """Display pattern discovery results.

    Auto-detects Jupyter notebook vs console environment.
    """
    if _is_notebook():
        _display_patterns_notebook(result)
    else:
        _display_patterns_console(result)


def display_learnings(learnings: List[LearningArtifact]) -> None:
    """Display a list of learning artifacts.

    Auto-detects Jupyter notebook vs console environment.
    """
    if _is_notebook():
        _display_learnings_notebook(learnings)
    else:
        _display_learnings_console(learnings)
