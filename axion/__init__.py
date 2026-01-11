# Suppress Pydantic v2 warnings about validate_default in Field()
# This warning occurs when Field() is used with validate_default in certain contexts
import warnings

warnings.filterwarnings(
    'ignore',
    message=".*validate_default.*",
    module='pydantic.*',
)

# Also suppress by specific warning category if available
try:
    from pydantic import PydanticDeprecatedSince20
    warnings.filterwarnings('ignore', category=PydanticDeprecatedSince20)
except ImportError:
    pass

# Core data structures
from axion.dataset import Dataset, DatasetItem
from axion.schema import MetricScore, EvaluationResult, TestResult, ErrorConfig

# Metric registry
from axion.metrics import metric_registry, MetricRegistry

# Evaluation runner
from axion.runners import evaluation_runner, EvaluationRunner, EvaluationConfig

__all__ = [
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
]
