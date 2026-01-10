from typing import Any, Dict, List

from axion._core.tracing.handlers import BaseTraceHandler, TracerProtocol


class KnowledgeTraceHandler(BaseTraceHandler):
    """Handler for Knowledge/RAG-specific functionality."""

    def __init__(self, tracer: TracerProtocol):
        super().__init__(tracer)
        if not hasattr(tracer.metadata, 'add_retrieval_call'):
            raise ValueError(
                'Knowledge trace handler requires KnowledgeExecutionMetadata instance'
            )

    def log_retrieval_call(
        self,
        context: List[Dict[str, Any]],
        latency: float,
        query: str = None,
        **attributes,
    ) -> None:
        """Log a retrieval call with automatic tracing."""
        span_id, trace_id = self.get_span_context()

        retrieval_data = {
            'query': query,
            'num_documents': len(context),
            'latency': latency,
            'span_id': span_id,
            'trace_id': trace_id,
            **attributes,
        }

        # Update metadata
        self.tracer.metadata.add_retrieval_call(context, latency)

        self.tracer.log_performance('retrieval_call', latency, **retrieval_data)
        self.tracer.add_trace(
            'retrieval_call', f'Retrieved {len(context)} documents', retrieval_data
        )

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        prompt_tokens: int = None,
        completion_tokens: int = None,
        latency: float = None,
        **attributes,
    ) -> None:
        """Log an LLM call for knowledge operations (simplified)."""
        prompt_tokens = prompt_tokens or len(prompt.split())
        completion_tokens = completion_tokens or (
            len(response.split()) if response else 0
        )

        span_id, trace_id = self.get_span_context()

        llm_data = {
            'model': model,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'latency': latency,
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
        )

        self.tracer.log_performance('llm_call', latency or 0, **llm_data)
        self.tracer.add_trace('llm_call', f'LLM call completed: {model}', llm_data)

    def get_statistics(self) -> Dict[str, Any]:
        """Get Knowledge/RAG execution statistics."""
        metadata = self.tracer.metadata
        stats = {
            'total_retrievals': len(metadata.retrieved_calls),
            'total_llm_calls': metadata.number_of_calls,
            'total_tokens': metadata.total_tokens,
        }

        if metadata.retrieved_calls:
            retrieval_latencies = [call.latency for call in metadata.retrieved_calls]
            stats.update(
                {
                    'average_retrieval_latency': sum(retrieval_latencies)
                    / len(retrieval_latencies),
                    'total_documents_retrieved': len(metadata.retrieved_calls),
                    'unique_files': len(
                        set(call.file_name for call in metadata.retrieved_calls)
                    ),
                }
            )

            scores = [
                call.score
                for call in metadata.retrieved_calls
                if call.score is not None
            ]
            if scores:
                stats.update(
                    {
                        'average_relevance_score': sum(scores) / len(scores),
                        'max_relevance_score': max(scores),
                        'min_relevance_score': min(scores),
                    }
                )

        return stats

    def display_statistics(self) -> None:
        """Display Knowledge/RAG statistics in a formatted table."""
        stats = self.get_statistics()
        if not stats:
            self.tracer.info('No knowledge statistics available')
            return

        def fmt(num, precision=0, suffix=''):
            return f'{num:,.{precision}f}{suffix}'

        table = [
            {
                'metric': 'Total Retrievals',
                'value': fmt(stats.get('total_retrievals', 0)),
                'details': f"Documents: {stats.get('total_documents_retrieved', 0)}",
            },
            {
                'metric': 'Total LLM Calls',
                'value': fmt(stats.get('total_llm_calls', 0)),
                'details': f"Tokens: {fmt(stats.get('total_tokens', 0))}",
            },
        ]

        if 'average_retrieval_latency' in stats:
            table.append(
                {
                    'metric': 'Avg Retrieval Time',
                    'value': fmt(stats['average_retrieval_latency'], 3, 's'),
                    'details': f"Files: {stats.get('unique_files', 0)}",
                }
            )

        if 'average_relevance_score' in stats:
            table.append(
                {
                    'metric': 'Avg Relevance Score',
                    'value': fmt(stats['average_relevance_score'], 3),
                    'details': f"Range: {stats['min_relevance_score']:.3f} - {stats['max_relevance_score']:.3f}",
                }
            )

        self.tracer.log_table(table, title='Knowledge Execution Statistics')
