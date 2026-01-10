import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import s3fs
from axion._core.logging import get_logger

logger = get_logger(__name__)


class StorageManager:
    """
    A flexible manager for saving and loading structured data (like QA pairs or statements)
    to and from CSV files using pandas. Supports automatic JSON and boolean column conversion.
    Handles local filesystem and Amazon S3.
    """

    def __init__(
        self,
        create_dirs: bool = True,
        schema: Optional[Dict[str, List[str]]] = None,
        json_columns: Optional[Set[str]] = None,
        boolean_columns: Optional[Set[str]] = None,
    ):
        """
        Initialize the StorageManager.

        Args:
            create_dirs: Whether to create directories if they don't exist.
            schema: A dictionary mapping data types to their expected columns.
            json_columns: Columns that should be serialized/deserialized as JSON.
            boolean_columns: Columns that should be interpreted as booleans.
        """
        self.create_dirs = create_dirs
        self.schema = schema or {
            'qa': [
                'question',
                'answer',
                'question_type',
                'difficulty',
                'statement_indices',
                'is_valid',
                'validation_feedback',
            ],
            'statements': ['index', 'statement'],
        }
        self.json_columns = json_columns or {'statement_indices'}
        self.boolean_columns = boolean_columns or {'is_valid'}
        self._s3 = None
        logger.debug('Initialized StorageManager')

    def _is_s3_path(self, path: Union[str, Path]) -> bool:
        """
        Check if a path is an S3 path.

        Args:
            path: The path to check.

        Returns:
            True if it's an S3 path, False otherwise.
        """
        return str(path).startswith('s3://')

    def _get_s3_fs(self):
        """
        Lazily initialize and return the S3 filesystem.

        Returns:
            An initialized S3FileSystem object.
        """
        if self._s3 is None:
            self._s3 = s3fs.S3FileSystem(anon=False)
        return self._s3

    def ensure_data_type(self, data_type: str) -> None:
        """Ensure valid data type."""
        keys = list(self.schema.keys())
        if data_type not in keys:
            raise ValueError(f'data_type must be either {keys}. Got: {data_type}')

    def _ensure_path(
        self, file_path: Union[str, Path], extension: str = '.csv'
    ) -> Union[Path, str]:
        """
        Resolve and validate a file path, appending the extension if needed.

        Args:
            file_path: The file path to resolve.
            extension: File extension to enforce (default is '.csv').

        Returns:
            A fully qualified Path object or S3 path string.
        """
        if self._is_s3_path(file_path):
            path_str = str(file_path)
            if not path_str.endswith(extension):
                path_str += extension
            return path_str

        path = Path(file_path).with_suffix(extension)
        if not path.is_absolute():
            path = Path(os.getcwd()) / path

        if not path.parent.exists():
            if self.create_dirs:
                logger.debug(f'Creating directory: {path.parent}')
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f'Directory does not exist: {path.parent}')

        return path

    def _prepare_data_for_save(self, data: List[Dict], data_type: str) -> pd.DataFrame:
        """
        Convert list of dictionaries into a DataFrame, applying JSON serialization.

        Args:
            data: The list of dictionaries to save.
            data_type: The type of data (must match the schema keys).

        Returns:
            A pandas DataFrame with processed data.
        """
        if not data:
            return pd.DataFrame(columns=self.schema[data_type])

        processed = [
            {
                **{
                    k: (
                        json.dumps(v)
                        if k in self.json_columns and isinstance(v, (list, dict))
                        else v
                    )
                    for k, v in item.items()
                }
            }
            for item in data
        ]

        df = pd.DataFrame(processed)
        for col in self.schema[data_type]:
            if col not in df:
                df[col] = ''

        return df[self.schema[data_type]]

    def _parse_loaded_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse data loaded from CSV to restore JSON and boolean columns.

        Args:
            records: List of records from a CSV.

        Returns:
            Parsed list of dictionaries.
        """
        for item in records:
            for col in self.json_columns:
                if isinstance(item.get(col), str):
                    try:
                        item[col] = json.loads(item[col])
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in '{col}': {item[col]}")
            for col in self.boolean_columns:
                if isinstance(item.get(col), str):
                    item[col] = item[col].lower() == 'true'
            for key, value in item.items():
                if pd.isna(value):
                    item[key] = ''
        return records

    def save_data(
        self,
        data: Union[List[Dict], List[str]],
        file_path: Union[str, Path],
        data_type: str,
    ) -> Union[Path, str]:
        """
        Save a list of data (dictionaries or strings) to a CSV file.

        Args:
            data: The data to save.
            file_path: Destination file path.
            data_type: A key identifying the schema to use.

        Returns:
            The full path to the saved file.
        """
        path = self._ensure_path(file_path)
        self.ensure_data_type(data_type)

        if isinstance(data[0], str):
            data = [
                {'index': i, self.schema[data_type][-1]: val}
                for i, val in enumerate(data)
            ]

        df = self._prepare_data_for_save(data, data_type)

        if self._is_s3_path(path):
            with self._get_s3_fs().open(path, 'w') as f:
                df.to_csv(f, index=False)
            logger.info(f'Saved {len(data)} records to S3: {path}')
        else:
            df.to_csv(path, index=False)
            logger.info(f'Saved {len(data)} records to {path}')

        return path

    def load_data(
        self, file_path: Union[str, Path], data_type: str
    ) -> Union[List[Dict], List[str]]:
        """
        Load data from a CSV file and return a structured list.

        Args:
            file_path: Complete path to the CSV file to load.
            data_type: A key identifying the schema to use.

        Returns:
            A list of dictionaries or strings, depending on the data type.
        """
        # path = self.ensure_path(file_path)
        path = file_path  # TODO
        self.ensure_data_type(data_type)

        keys = list(self.schema.keys())
        if data_type not in keys:
            raise ValueError(f'data_type must be either {keys}. Got: {data_type}')

        if self._is_s3_path(path):
            fs = self._get_s3_fs()
            if not fs.exists(path):
                raise FileNotFoundError(f'S3 file not found: {path}')
            with fs.open(path, 'r') as f:
                df = pd.read_csv(f)
        else:
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f'File not found: {path_obj}')
            df = pd.read_csv(path_obj)

        if data_type == 'statements':
            if 'index' in df.columns:
                df = df.sort_values('index')
            last_col = self.schema[data_type][-1]
            statements = df[last_col].tolist()
            logger.info(
                f"Loaded {len(statements)} statements from {'S3: ' if self._is_s3_path(path) else ''}{path}"
            )
            return statements

        records = df.to_dict(orient='records')
        parsed = self._parse_loaded_data(records)
        logger.info(
            f"Loaded {len(parsed)} {data_type} records from {'S3: ' if self._is_s3_path(path) else ''}{path}"
        )
        return parsed
