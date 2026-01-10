from typing import List, Union

import pandas as pd
from axion.dataset import DatasetItem
from axion._core.types import FieldNames


class EvaluationValidation:
    """
    A utility class for reusable evaluation validation methods.
    """

    @staticmethod
    def ensure_required_fields_present(item: DatasetItem, context: str = '') -> None:
        """
        Validates that at least one of the required fields is present in the given DatasetItem.

        Args:
            item (DatasetItem): The evaluation item to check.
            context (str, optional): Context string for error messages (e.g., validation step name).

        Raises:
            ValueError: If none of the required fields are present or non-None.
        """
        missing = all(
            getattr(item, field, None) is None
            for field in FieldNames.get_required_evaluation_input_fields()
        )

        if missing:
            context_msg = f' for {context}' if context else ''
            required_fields = ' or '.join(
                f'`{f}`' for f in FieldNames.get_required_evaluation_input_fields()
            )
            raise ValueError(f'{required_fields} is required{context_msg}')

    @staticmethod
    def validate_evaluation_task_inputs(
        evaluation_inputs: Union[List[DatasetItem], pd.DataFrame],
    ) -> None:
        """
        Validates that an evaluation config is valid for execution.

        Args:
            evaluation_inputs: List of Object that may contain `actual_output`.

        Raises:
            ValueError: If task is missing and evaluation inputs lack `actual_output`.
        """

        if isinstance(evaluation_inputs, pd.DataFrame):
            from axion.dataset import Dataset

            evaluation_inputs = Dataset.read_dataframe(evaluation_inputs).items
        first_item = evaluation_inputs[0]
        EvaluationValidation.ensure_required_fields_present(
            first_item, context='EvaluationConfig'
        )
