import importlib
import json
from ast import literal_eval
from datetime import datetime, timezone
from typing import Any, Callable, List, TypeVar, Union

T = TypeVar('T')


def current_datetime():
    return datetime.now(timezone.utc).isoformat()


def convert_to_list(list_string: str) -> Union[List[str], str]:
    """
    Convert string representation of a list to an actual list,
    or return original if it fails.
    """
    try:
        return literal_eval(list_string)
    except (ValueError, SyntaxError):
        return list_string


def format_value(value: Any) -> Any:
    """
    Format and convert values to appropriate Python types.

    Handles:
    - String representations of lists/dicts
    - JSON strings
    - Number strings
    - Boolean strings
    - None/null strings

    Args:
        value: Input value to format

    Returns:
        Appropriately converted value
    """
    # Return non-string values as-is
    if not isinstance(value, str):
        return value

    # Handle empty strings
    if not value.strip():
        return value

    # Handle None/null
    if value.lower() in ('none', 'null'):
        return None

    # Handle boolean strings
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False

    # Try to parse as JSON first (handles both lists and dicts)
    if value.strip().startswith(('[', '{')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # If JSON fails, try convert_to_list (which uses literal_eval)
            result = convert_to_list(value)
            # If convert_to_list succeeded, return the result
            if result != value:
                return result

    # Try to convert to number
    try:
        # Check for integer
        if '.' not in value:
            return int(value)
        # Check for float
        return float(value)
    except ValueError:
        pass

    return value


def lazy_import(path: str) -> Callable:
    """
    Dynamically import a class or function from a string path.

    Args:
        path (str): Full import path to the target object (e.g., 'module.submodule.ClassName').

    Returns:
        Callable: The class or function object referenced by the import path.
    """
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
