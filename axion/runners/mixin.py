from abc import ABC
from typing import List, Union, cast

import pandas as pd

from axion._core.metadata.schema import ToolMetadata
from axion.dataset import Dataset, DatasetItem


class RunnerMixin(ABC):
    """
    Mixin class that provides utility methods to normalize evaluation inputs.
    """

    name: str
    description: str = 'Orchestration Engine'
    owner: str = 'AI Engineering'
    version: str = '1.0.0'

    def get_tool_metadata(self):
        return ToolMetadata(
            name=self.name,
            description=self.description,
            owner=self.owner,
            version=self.version,
        )

    def to_queries(
        self,
        evaluation_inputs: Union[Dataset, List[DatasetItem], pd.DataFrame, List[str]],
        dataset_name: str = 'Untitled',
    ) -> List[str]:
        """
        Converts evaluation inputs into a list of queries.

        Args:
            evaluation_inputs: Various formats of evaluation data.
            dataset_name: Optional name if a DataFrame is passed and converted.

        Returns:
            A list of query strings.
        """
        if isinstance(evaluation_inputs, list) and isinstance(
            evaluation_inputs[0], str
        ):
            return cast(List[str], evaluation_inputs)

        dataset = self.to_evaluation_input(evaluation_inputs, dataset_name)
        if isinstance(dataset, (Dataset, list)):
            return [item.query for item in dataset]

        return dataset  # type: ignore[return-value]

    @staticmethod
    def to_evaluation_input(
        evaluation_inputs: Union[Dataset, List[DatasetItem], pd.DataFrame],
        dataset_name: str = None,
    ) -> Union[Dataset, List[DatasetItem]]:
        """
        Converts a DataFrame to a Dataset; returns existing datasets unchanged.

        Args:
            evaluation_inputs: Either a Dataset, a list of DatasetItems, or a DataFrame.
            dataset_name: Optional name to assign to the dataset if created from a DataFrame.

        Returns:
            A Dataset or list of DatasetItems.
        """
        if isinstance(evaluation_inputs, pd.DataFrame):
            return Dataset.read_dataframe(
                name=dataset_name, dataframe=evaluation_inputs
            )

        return evaluation_inputs
