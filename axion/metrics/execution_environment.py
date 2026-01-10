import logging
from typing import Any, Dict

RESTRICTED_PYTHON_AVAILABLE = True
try:
    from RestrictedPython import safe_globals
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    safe_globals = {}

logger = logging.getLogger(__name__)

ALLOWED_BUILTINS = {
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'enumerate': enumerate,
    'float': float,
    'frozenset': frozenset,
    'getattr': getattr,
    'hasattr': hasattr,
    'int': int,
    'isinstance': isinstance,
    'len': len,
    'list': list,
    'max': max,
    'min': min,
    'range': range,
    'repr': repr,
    'reversed': reversed,
    'round': round,
    'set': set,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'zip': zip,
}

ALLOWED_MODULES = {
    're': __import__('re'),
    'math': __import__('math'),
    'statistics': __import__('statistics'),
    'itertools': __import__('itertools'),
    'collections': __import__('collections'),
    'json': __import__('json'),
}


def create_execution_environment() -> Dict[str, Any]:
    """
    Creates a safe execution environment for user-provided heuristic code.
    """
    from axion.dataset import DatasetItem
    from axion.metrics.schema import MetricEvaluationResult

    base_environment = {
        '__name__': '__main__',
        'DatasetItem': DatasetItem,
        'MetricEvaluationResult': MetricEvaluationResult,
        **ALLOWED_MODULES,
    }

    if RESTRICTED_PYTHON_AVAILABLE:
        env = safe_globals.copy()
        env.update(base_environment)
        env.update(ALLOWED_BUILTINS)
        return env
    else:
        # The fallback path (less secure).
        logger.warning(
            'RestrictedPython not found. Using a less secure fallback environment. '
            'Ensure heuristic code is from a trusted source.'
        )
        env = base_environment.copy()
        env['__builtins__'] = ALLOWED_BUILTINS
        return env
