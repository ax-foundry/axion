from typing import Any, List, Optional

import pandas as pd
from pydantic import BaseModel

from axion.dataset import Dataset


def input_to_dataset(evaluation_inputs: Any, name: Optional[str] = None) -> Dataset:
    """
    Convert various input formats into a standardized `Dataset` object.

    This helper function accepts multiple input types commonly used for
    evaluation and normalizes them into a `Dataset` instance, ensuring a
    consistent interface for downstream processing.

    Args:
        evaluation_inputs: The input data to be converted. Can be one of:
            - `Dataset`: Returned directly without modification.
            - `list`: A list of dataset items to wrap in a new `Dataset`.
            - `pd.DataFrame`: Converted into a `Dataset` via `Dataset.read_dataframe()`.
        name: Optional name assigned to the resulting `Dataset`.

    Returns:
        Dataset: A standardized dataset object derived from the provided input.
    """
    if isinstance(evaluation_inputs, Dataset):
        return evaluation_inputs
    if isinstance(evaluation_inputs, list):
        return Dataset(items=evaluation_inputs, name=name)
    if isinstance(evaluation_inputs, pd.DataFrame):
        return Dataset.read_dataframe(dataframe=evaluation_inputs, name=name)
    raise TypeError(f'Unsupported input type: {type(evaluation_inputs)}')


def models_to_dataframe(models: List[BaseModel], **dump_kwargs) -> pd.DataFrame:
    """
    Converts a list of Pydantic models to a pandas DataFrame.

    Args:
        models (List[BaseModel]): A list of Pydantic model instances.
        **dump_kwargs: Optional keyword arguments to pass to `model_dump()`
                       (e.g., exclude_unset=True, exclude_none=True).

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a model and each column to a field.
    """
    return pd.DataFrame(model.model_dump(**dump_kwargs) for model in models)
