from typing import Any, Dict, List

from axion._core.metadata.schema import (
    EvaluationDatapoint,
    EvaluationMetric,
    LLMExecutionMetadata,
)
from axion._core.tracing.handlers import BaseTraceHandler, TracerProtocol


class LLMTraceHandler(BaseTraceHandler):
    """Handler for LLM-specific functionality."""

    def __init__(self, tracer: TracerProtocol):
        super().__init__(tracer)
        if not isinstance(tracer.metadata, LLMExecutionMetadata):
            raise ValueError('LLM trace handler requires LLMExecutionMetadata instance')

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        prompt_tokens: int = None,
        completion_tokens: int = None,
        latency: float = None,
        cost_estimate: float = None,
        error: str = None,
        **attributes,
    ) -> None:
        """Log an LLM call with automatic tracing."""
        try:
            prompt_tokens = prompt_tokens or len(prompt.split())
            completion_tokens = completion_tokens or (
                len(response.split()) if response else 0
            )
        except AttributeError:
            prompt_tokens = 0
            completion_tokens = 0

        total_tokens = prompt_tokens + completion_tokens

        # Get current span context
        span_id, trace_id = self.get_span_context()

        llm_data = {
            'model': model,
            'prompt_length': len(prompt),
            'response_length': len(response or ''),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'latency': latency,
            'cost_estimate': cost_estimate,
            'error': error,
            'span_id': span_id,
            'trace_id': trace_id,
            **attributes,
        }

        # Update metadata
        self.tracer.metadata.add_llm_call(
            model=model,
            provider=attributes.get('provider', 'unknown'),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=latency or 0.0,
            prompt_length=len(prompt),
            response_length=len(response or ''),
            cost_estimate=cost_estimate,
            error=error,
            span_id=span_id,
            trace_id=trace_id,
        )

        # Log performance and add trace
        self.tracer.log_performance('llm_call', latency or 0, **llm_data)
        self.tracer.add_trace(
            'llm_call',
            f"LLM call {'failed' if error else 'completed'}: {model}",
            llm_data,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get LLM execution statistics."""
        metadata = self.tracer.metadata
        if metadata.number_of_calls == 0:
            return {}

        successful_calls = metadata.number_of_calls - metadata.failed_calls
        return {
            'total_calls': metadata.number_of_calls,
            'successful_calls': successful_calls,
            'failed_calls': metadata.failed_calls,
            'success_rate': successful_calls / metadata.number_of_calls,
            'total_tokens': metadata.total_tokens,
            'total_prompt_tokens': metadata.total_prompt_tokens,
            'total_completion_tokens': metadata.total_completion_tokens,
            'average_tokens_per_call': metadata.total_tokens / metadata.number_of_calls,
            'total_latency': metadata.total_llm_latency,
            'average_latency': metadata.average_latency,
            'total_cost_estimate': metadata.total_cost_estimate,
            'unique_models': list(metadata.unique_models),
            'unique_providers': list(metadata.unique_providers),
            'retry_count': metadata.retry_count,
        }

    def display_statistics(self) -> None:
        """Display LLM statistics in a formatted table."""
        stats = self.get_statistics()
        if not stats:
            self.tracer.info('No LLM statistics available')
            return

        def fmt(num, precision=0, suffix=''):
            return f'{num:,.{precision}f}{suffix}'

        table = [
            {
                'metric': 'Total Calls',
                'value': fmt(stats['total_calls']),
                'details': f"Success: {stats['successful_calls']}, Failed: {stats['failed_calls']}",
            },
            {
                'metric': 'Success Rate',
                'value': f"{stats['success_rate']:.1%}",
                'details': f"({stats['successful_calls']}/{stats['total_calls']})",
            },
            {
                'metric': 'Token Usage',
                'value': fmt(stats['total_tokens']),
                'details': f"Prompt: {fmt(stats['total_prompt_tokens'])}, Completion: {fmt(stats['total_completion_tokens'])}",
            },
            {
                'metric': 'Average Latency',
                'value': fmt(stats['average_latency'], 3, 's'),
                'details': f"Total: {fmt(stats['total_latency'], 3)}s",
            },
            {
                'metric': 'Avg Tokens/Call',
                'value': fmt(stats['average_tokens_per_call']),
                'details': 'Tokens per successful call',
            },
        ]

        if stats['total_cost_estimate'] > 0:
            table.append(
                {
                    'metric': 'Estimated Cost',
                    'value': f"${stats['total_cost_estimate']:.4f}",
                    'details': f"${stats['total_cost_estimate'] / stats['total_calls']:.4f} per call",
                }
            )

        self.tracer.log_table(table, title='LLM Execution Statistics')


class EvaluationTraceHandler(BaseTraceHandler):
    """Handler for evaluation-specific functionality."""

    def log_evaluation(
        self,
        evaluation_id: str,
        evaluator_name: str,
        evaluator_type: str,
        dataset_size: int,
        latency: float,
        overall_metrics: List[EvaluationMetric] = None,
        datapoint_results: List[EvaluationDatapoint] = None,
        dataset_name: str = None,
        cost_estimate: float = None,
        tokens_used: int = None,
        error: str = None,
        evaluator_config: Dict[str, Any] = None,
        **attributes,
    ) -> None:
        """Log an evaluation run with automatic tracing."""

        # Get current span context
        span_id, trace_id = self.get_span_context()

        # Calculate summary statistics
        successful_datapoints = len(
            [dp for dp in (datapoint_results or []) if dp.error is None]
        )
        failed_datapoints = dataset_size - successful_datapoints
        success_rate = successful_datapoints / dataset_size if dataset_size > 0 else 0
        avg_latency_per_datapoint = latency / dataset_size if dataset_size > 0 else 0

        # Aggregate metric values for logging
        metric_summary = {}
        if overall_metrics:
            for metric in overall_metrics:
                if isinstance(metric.value, (int, float)):
                    metric_summary[f'metric_{metric.name}'] = metric.value
                    if metric.passed is not None:
                        metric_summary[f'metric_{metric.name}_passed'] = metric.passed

        evaluation_data = {
            'evaluation_id': evaluation_id,
            'evaluator_name': evaluator_name,
            'evaluator_type': evaluator_type,
            'dataset_name': dataset_name,
            'dataset_size': dataset_size,
            'successful_datapoints': successful_datapoints,
            'failed_datapoints': failed_datapoints,
            'success_rate': success_rate,
            'latency': latency,
            'average_latency_per_datapoint': avg_latency_per_datapoint,
            'cost_estimate': cost_estimate,
            'tokens_used': tokens_used,
            'error': error,
            'span_id': span_id,
            'trace_id': trace_id,
            **metric_summary,
            **attributes,
        }

        # Update metadata
        self.tracer.metadata.add_evaluation_call(
            evaluation_id=evaluation_id,
            evaluator_name=evaluator_name,
            evaluator_type=evaluator_type,
            dataset_size=dataset_size,
            latency=latency,
            overall_metrics=overall_metrics,
            datapoint_results=datapoint_results,
            dataset_name=dataset_name,
            cost_estimate=cost_estimate,
            tokens_used=tokens_used,
            error=error,
            span_id=span_id,
            trace_id=trace_id,
            evaluator_config=evaluator_config,
        )

        # Log performance and add trace
        self.tracer.log_performance('evaluation', latency, **evaluation_data)
        self.tracer.add_trace(
            'evaluation',
            f"Evaluation {'failed' if error else 'completed'}: {evaluator_name} on {dataset_size} datapoints",
            evaluation_data,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation execution statistics."""
        metadata = self.tracer.metadata
        if metadata.number_of_evaluations == 0:
            return {}

        successful_evaluations = (
            metadata.number_of_evaluations - metadata.failed_evaluations
        )

        # Calculate aggregated metrics
        all_metrics = {}
        metric_counts = {}

        for eval_call in metadata.evaluation_calls:
            for metric in eval_call.overall_metrics:
                if isinstance(metric.value, (int, float)):
                    if metric.name not in all_metrics:
                        all_metrics[metric.name] = []
                    all_metrics[metric.name].append(metric.value)
                    metric_counts[metric.name] = metric_counts.get(metric.name, 0) + 1

        # Calculate averages
        average_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                average_metrics[f'avg_{metric_name}'] = sum(values) / len(values)
                average_metrics[f'min_{metric_name}'] = min(values)
                average_metrics[f'max_{metric_name}'] = max(values)

        return {
            'total_evaluations': metadata.number_of_evaluations,
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': metadata.failed_evaluations,
            'success_rate': successful_evaluations / metadata.number_of_evaluations,
            'total_datapoints': metadata.total_datapoints_evaluated,
            'total_evaluation_time': metadata.total_evaluation_time,
            'average_evaluation_latency': metadata.average_evaluation_latency,
            'average_datapoint_latency': metadata.average_datapoint_latency,
            'total_cost_estimate': metadata.total_cost_estimate,
            'total_tokens_used': metadata.total_tokens_used,
            'unique_evaluators': list(metadata.unique_evaluators),
            'unique_datasets': list(metadata.unique_datasets),
            **average_metrics,
        }

    def display_statistics(self) -> None:
        """Display evaluation statistics in a formatted table."""
        stats = self.get_statistics()
        if not stats:
            self.tracer.info('No evaluation statistics available')
            return

        def fmt(num, precision=0, suffix=''):
            return f'{num:,.{precision}f}{suffix}'

        table = [
            {
                'metric': 'Total Evaluations',
                'value': fmt(stats['total_evaluations']),
                'details': f"Success: {stats['successful_evaluations']}, Failed: {stats['failed_evaluations']}",
            },
            {
                'metric': 'Success Rate',
                'value': f"{stats['success_rate']:.1%}",
                'details': f"({stats['successful_evaluations']}/{stats['total_evaluations']})",
            },
            {
                'metric': 'Total Datapoints',
                'value': fmt(stats['total_datapoints']),
                'details': f"Across {len(stats['unique_datasets'])} datasets",
            },
            {
                'metric': 'Avg Evaluation Time',
                'value': fmt(stats['average_evaluation_latency'], 3, 's'),
                'details': f"Total: {fmt(stats['total_evaluation_time'], 3)}s",
            },
            {
                'metric': 'Avg Time/Datapoint',
                'value': fmt(stats['average_datapoint_latency'], 3, 's'),
                'details': 'Per individual datapoint',
            },
        ]

        if stats['total_cost_estimate'] > 0:
            table.append(
                {
                    'metric': 'Estimated Cost',
                    'value': f"${stats['total_cost_estimate']:.4f}",
                    'details': f"${stats['total_cost_estimate'] / stats['total_evaluations']:.4f} per evaluation",
                }
            )

        if stats['total_tokens_used'] > 0:
            table.append(
                {
                    'metric': 'Tokens Used',
                    'value': fmt(stats['total_tokens_used']),
                    'details': f"{fmt(stats['total_tokens_used'] / stats['total_evaluations'])} per evaluation",
                }
            )

        # Add metric-specific statistics
        metric_stats = [
            (k, v) for k, v in stats.items() if k.startswith(('avg_', 'min_', 'max_'))
        ]
        if metric_stats:
            table.append(
                {
                    'metric': 'Evaluation Metrics',
                    'value': f"{len([k for k in stats.keys() if k.startswith('avg_')])} metrics",
                    'details': 'See detailed breakdown below',
                }
            )

        self.tracer.log_table(table, title='Evaluation Execution Statistics')

        # Display detailed metrics if available
        if metric_stats:
            metric_table = []
            metric_names = set()
            for key, value in metric_stats:
                if key.startswith('avg_'):
                    metric_name = key[4:]  # Remove 'avg_' prefix
                    metric_names.add(metric_name)

            for metric_name in sorted(metric_names):
                avg_key = f'avg_{metric_name}'
                min_key = f'min_{metric_name}'
                max_key = f'max_{metric_name}'

                if all(key in stats for key in [avg_key, min_key, max_key]):
                    metric_table.append(
                        {
                            'metric': metric_name,
                            'average': fmt(stats[avg_key], 3),
                            'range': f'{fmt(stats[min_key], 3)} - {fmt(stats[max_key], 3)}',
                        }
                    )

            if metric_table:
                self.tracer.log_table(metric_table, title='Detailed Evaluation Metrics')
