"""
Trace loaders for observability platforms.

This module provides loaders for fetching traces from various observability
platforms (Langfuse, Opik, Logfire) and converting them to Axion Dataset
format for evaluation.

Example:
    from axion.tracing import LangfuseTraceLoader
    from axion import evaluation_runner
    from axion.metrics import Faithfulness

    # Load traces from Langfuse
    loader = LangfuseTraceLoader()
    dataset = loader.to_dataset(name='rag-eval', limit=100)

    # Run evaluation
    result = evaluation_runner(
        evaluation_inputs=dataset,
        scoring_metrics=[Faithfulness()]
    )

    # Push scores back to Langfuse
    stats = loader.push_scores_to_langfuse(result)
    print(f"Uploaded {stats['uploaded']} scores")
"""

from axion._core.tracing.loaders.base import BaseTraceLoader, FetchedTraceData
from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader
from axion._core.tracing.loaders.logfire import LogfireTraceLoader
from axion._core.tracing.loaders.opik import OpikTraceLoader

__all__ = [
    # Base classes
    'FetchedTraceData',
    'BaseTraceLoader',
    # Provider implementations
    'LangfuseTraceLoader',
    'OpikTraceLoader',
    'LogfireTraceLoader',
]
