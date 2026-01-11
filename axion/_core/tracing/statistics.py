"""Statistics display utilities for tracing.

This module provides functions for formatting and displaying statistics
from metadata instances. These were extracted from the handler classes
to provide a simpler, more focused utility.
"""

from typing import Any, Callable, Dict, List, Optional

from axion._core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    'display_base_statistics',
    'display_llm_statistics',
    'display_evaluation_statistics',
    'display_knowledge_statistics',
    'display_statistics',
]

# Constants for truncation
MAX_ARG_LENGTH = 500
MAX_RESULT_LENGTH = 2000
MAX_ERROR_LENGTH = 200


def _fmt(num: Optional[float], precision: int = 0, suffix: str = '') -> str:
    """Format a number with optional precision and suffix."""
    if num is None:
        return 'N/A'
    return f'{num:,.{precision}f}{suffix}'


def display_base_statistics(
    stats: Dict[str, Any],
    log_table: Callable[[List[Dict[str, Any]], Optional[str]], None],
    info: Callable[[str], None],
) -> None:
    """Display basic execution statistics."""
    if not stats:
        info('No statistics available')
        return

    table = [
        {
            'metric': 'Execution ID',
            'value': stats['execution_id'][:8] + '...',
            'details': f"Status: {stats['status']}",
        },
        {
            'metric': 'Execution Time',
            'value': _fmt(stats.get('latency'), 3, 's') if stats.get('latency') else 'N/A',
            'details': f"Traces: {stats.get('traces_count', 0)}",
        },
    ]

    log_table(table, 'Execution Statistics')


def display_llm_statistics(
    stats: Dict[str, Any],
    log_table: Callable[[List[Dict[str, Any]], Optional[str]], None],
    info: Callable[[str], None],
) -> None:
    """Display LLM execution statistics."""
    if not stats or stats.get('total_calls', 0) == 0:
        info('No LLM statistics available')
        return

    table = [
        {
            'metric': 'Total Calls',
            'value': _fmt(stats['total_calls']),
            'details': f"Success: {stats['successful_calls']}, Failed: {stats['failed_calls']}",
        },
        {
            'metric': 'Success Rate',
            'value': f"{stats['success_rate']:.1%}",
            'details': f"({stats['successful_calls']}/{stats['total_calls']})",
        },
        {
            'metric': 'Token Usage',
            'value': _fmt(stats['total_tokens']),
            'details': f"Prompt: {_fmt(stats['total_prompt_tokens'])}, Completion: {_fmt(stats['total_completion_tokens'])}",
        },
        {
            'metric': 'Average Latency',
            'value': _fmt(stats['average_latency'], 3, 's'),
            'details': f"Total: {_fmt(stats['total_latency'], 3)}s",
        },
        {
            'metric': 'Avg Tokens/Call',
            'value': _fmt(stats['average_tokens_per_call']),
            'details': 'Tokens per successful call',
        },
    ]

    if stats.get('total_cost_estimate', 0) > 0:
        table.append({
            'metric': 'Estimated Cost',
            'value': f"${stats['total_cost_estimate']:.4f}",
            'details': f"${stats['total_cost_estimate'] / stats['total_calls']:.4f} per call",
        })

    log_table(table, 'LLM Execution Statistics')


def display_evaluation_statistics(
    stats: Dict[str, Any],
    log_table: Callable[[List[Dict[str, Any]], Optional[str]], None],
    info: Callable[[str], None],
) -> None:
    """Display evaluation execution statistics."""
    if not stats or stats.get('total_evaluations', 0) == 0:
        info('No evaluation statistics available')
        return

    table = [
        {
            'metric': 'Total Evaluations',
            'value': _fmt(stats['total_evaluations']),
            'details': f"Success: {stats['successful_evaluations']}, Failed: {stats['failed_evaluations']}",
        },
        {
            'metric': 'Success Rate',
            'value': f"{stats['success_rate']:.1%}",
            'details': f"({stats['successful_evaluations']}/{stats['total_evaluations']})",
        },
        {
            'metric': 'Total Datapoints',
            'value': _fmt(stats['total_datapoints']),
            'details': f"Across {len(stats.get('unique_datasets', []))} datasets",
        },
        {
            'metric': 'Avg Evaluation Time',
            'value': _fmt(stats['average_evaluation_latency'], 3, 's'),
            'details': f"Total: {_fmt(stats['total_evaluation_time'], 3)}s",
        },
        {
            'metric': 'Avg Time/Datapoint',
            'value': _fmt(stats['average_datapoint_latency'], 3, 's'),
            'details': 'Per individual datapoint',
        },
    ]

    if stats.get('total_cost_estimate', 0) > 0:
        table.append({
            'metric': 'Estimated Cost',
            'value': f"${stats['total_cost_estimate']:.4f}",
            'details': f"${stats['total_cost_estimate'] / stats['total_evaluations']:.4f} per evaluation",
        })

    if stats.get('total_tokens_used', 0) > 0:
        table.append({
            'metric': 'Tokens Used',
            'value': _fmt(stats['total_tokens_used']),
            'details': f"{_fmt(stats['total_tokens_used'] / stats['total_evaluations'])} per evaluation",
        })

    # Check for metric-specific statistics
    metric_stats = [(k, v) for k, v in stats.items() if k.startswith(('avg_', 'min_', 'max_'))]
    if metric_stats:
        table.append({
            'metric': 'Evaluation Metrics',
            'value': f"{len([k for k in stats.keys() if k.startswith('avg_')])} metrics",
            'details': 'See detailed breakdown below',
        })

    log_table(table, 'Evaluation Execution Statistics')

    # Display detailed metrics if available
    if metric_stats:
        metric_table = []
        metric_names = set()
        for key, _ in metric_stats:
            if key.startswith('avg_'):
                metric_name = key[4:]  # Remove 'avg_' prefix
                metric_names.add(metric_name)

        for metric_name in sorted(metric_names):
            avg_key = f'avg_{metric_name}'
            min_key = f'min_{metric_name}'
            max_key = f'max_{metric_name}'

            if all(key in stats for key in [avg_key, min_key, max_key]):
                metric_table.append({
                    'metric': metric_name,
                    'average': _fmt(stats[avg_key], 3),
                    'range': f'{_fmt(stats[min_key], 3)} - {_fmt(stats[max_key], 3)}',
                })

        if metric_table:
            log_table(metric_table, 'Detailed Evaluation Metrics')


def display_knowledge_statistics(
    stats: Dict[str, Any],
    log_table: Callable[[List[Dict[str, Any]], Optional[str]], None],
    info: Callable[[str], None],
) -> None:
    """Display Knowledge/RAG execution statistics."""
    if not stats:
        info('No knowledge statistics available')
        return

    table = [
        {
            'metric': 'Total Retrievals',
            'value': _fmt(stats.get('total_retrievals', 0)),
            'details': f"Documents: {stats.get('total_documents_retrieved', 0)}",
        },
        {
            'metric': 'Total LLM Calls',
            'value': _fmt(stats.get('total_llm_calls', 0)),
            'details': f"Tokens: {_fmt(stats.get('total_tokens', 0))}",
        },
    ]

    if 'average_retrieval_latency' in stats:
        table.append({
            'metric': 'Avg Retrieval Time',
            'value': _fmt(stats['average_retrieval_latency'], 3, 's'),
            'details': f"Files: {stats.get('unique_files', 0)}",
        })

    if 'average_relevance_score' in stats:
        table.append({
            'metric': 'Avg Relevance Score',
            'value': _fmt(stats['average_relevance_score'], 3),
            'details': f"Range: {stats['min_relevance_score']:.3f} - {stats['max_relevance_score']:.3f}",
        })

    log_table(table, 'Knowledge Execution Statistics')


def display_statistics(
    metadata_type: str,
    stats: Dict[str, Any],
    log_table: Callable[[List[Dict[str, Any]], Optional[str]], None],
    info: Callable[[str], None],
) -> None:
    """Display statistics based on metadata type.

    Args:
        metadata_type: Type of metadata ('base', 'llm', 'evaluation', 'knowledge')
        stats: Statistics dictionary from metadata.get_statistics()
        log_table: Function to log table data
        info: Function to log info messages
    """
    display_funcs = {
        'llm': display_llm_statistics,
        'evaluation': display_evaluation_statistics,
        'knowledge': display_knowledge_statistics,
    }

    display_func = display_funcs.get(metadata_type, display_base_statistics)
    display_func(stats, log_table, info)
