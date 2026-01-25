from typing import Optional

# Environment
from axion._core.environment import settings

# Core data structures
from axion.dataset import Dataset, DatasetItem

# Metric registry
from axion.metrics import MetricRegistry, metric_registry

# Evaluation runner
from axion.runners import EvaluationConfig, EvaluationRunner, evaluation_runner
from axion.schema import ErrorConfig, EvaluationResult, MetricScore, TestResult
from axion.handlers import BaseHandler, LLMHandler


def init(
    tracing: Optional[str] = None,
    log_level: Optional[str] = None,
    log_rich: Optional[bool] = None,
) -> None:
    """
    Initialize axion with optional overrides.

    Configures both tracing and logging. Call once at startup if you want
    to override auto-detected settings. If not called, both systems
    auto-configure on first use.

    Args:
        tracing: Tracing provider ('noop', 'logfire', 'otel', 'langfuse', 'opik').
            If None, auto-detects from environment variables.
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
            If None, uses LOG_LEVEL env var or 'INFO'.
        log_rich: Enable rich formatting. If None, uses LOG_RICH env var.

    Example:
        >>> import axion
        >>> axion.init(tracing='langfuse', log_level='DEBUG')
    """
    from axion.logging import configure_logging
    from axion.tracing import configure_tracing

    configure_tracing(provider=tracing)
    configure_logging(level=log_level, use_rich=log_rich)


__all__ = [
    # Initialization
    'init',
    # Data structures
    'Dataset',
    'DatasetItem',
    'MetricScore',
    'EvaluationResult',
    'TestResult',
    'ErrorConfig',
    # Registry
    'metric_registry',
    'MetricRegistry',
    # Runner
    'evaluation_runner',
    'EvaluationRunner',
    'EvaluationConfig',
    # Handlers
    'BaseHandler',
    'LLMHandler',
    # Environment
    'settings',
]
