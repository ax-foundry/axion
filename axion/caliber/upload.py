"""
Upload handling for CaliberHQ workflow (Step 1).

Handles data loading and validation from various sources.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from axion._core.logging import get_logger
from axion._core.uuid import uuid7
from axion.caliber.models import UploadedRecord, UploadResult

logger = get_logger(__name__)


class UploadHandler:
    """
    Handles data upload and validation for CaliberHQ.

    Supports loading data from CSV files, DataFrames, and dictionaries.

    Example:
        >>> handler = UploadHandler()
        >>> result = handler.from_csv("data.csv")
        >>> print(f"Loaded {result.total_count} records")
    """

    REQUIRED_COLUMNS: Set[str] = {'query', 'actual_output'}
    OPTIONAL_COLUMNS: Set[str] = {
        'id',
        'expected_output',
        'llm_score',
        'llm_reasoning',
        'judgment',
    }

    # Column aliases for flexible input
    COLUMN_ALIASES: Dict[str, List[str]] = {
        'query': ['query', 'input', 'question', 'prompt'],
        'actual_output': ['actual_output', 'output', 'response', 'answer'],
        'expected_output': ['expected_output', 'reference', 'ground_truth', 'expected'],
        'llm_score': ['llm_score', 'score', 'llm_judgment'],
        'llm_reasoning': [
            'llm_reasoning',
            'reasoning',
            'explanation',
            'llm_explanation',
        ],
        'id': ['id', 'record_id', 'item_id'],
    }

    def __init__(self, strict_validation: bool = False):
        """
        Initialize the upload handler.

        Args:
            strict_validation: If True, raise errors for validation issues.
                              If False (default), collect warnings instead.
        """
        self._strict_validation = strict_validation

    def from_csv(
        self,
        path: Union[str, Path],
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """
        Load data from a CSV file.

        Args:
            path: Path to the CSV file
            column_mapping: Optional mapping to rename columns

        Returns:
            UploadResult with loaded records
        """
        df = pd.read_csv(path)
        if column_mapping:
            df = df.rename(columns=column_mapping)
        return self.from_dataframe(df)

    def from_dataframe(self, df: pd.DataFrame) -> UploadResult:
        """
        Load data from a pandas DataFrame.

        Args:
            df: DataFrame with records

        Returns:
            UploadResult with loaded records
        """
        # Apply column aliases
        df = self._apply_column_aliases(df)

        # Validate required columns
        warnings = self._validate_columns(df)

        records: List[UploadedRecord] = []
        for idx, row in df.iterrows():
            record = self._row_to_record(row, idx)
            if record:
                records.append(record)

        # Check for LLM scores
        has_llm_scores = any(r.llm_score is not None for r in records)

        return UploadResult(
            records=records,
            total_count=len(records),
            has_llm_scores=has_llm_scores,
            validation_warnings=warnings,
        )

    def from_records(self, records: List[Dict[str, Any]]) -> UploadResult:
        """
        Load data from a list of dictionaries.

        Args:
            records: List of record dictionaries

        Returns:
            UploadResult with loaded records
        """
        df = pd.DataFrame(records)
        return self.from_dataframe(df)

    def validate(self, records: List[UploadedRecord]) -> List[str]:
        """
        Validate a list of records.

        Args:
            records: Records to validate

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check for empty records
        if not records:
            warnings.append('No records provided')
            return warnings

        # Check for missing required fields
        for idx, record in enumerate(records):
            if not record.query:
                warnings.append(f'Record {idx}: missing query')
            if not record.actual_output:
                warnings.append(f'Record {idx}: missing actual_output')

        # Check for duplicate IDs
        ids = [r.id for r in records]
        if len(ids) != len(set(ids)):
            warnings.append('Duplicate record IDs detected')

        return warnings

    def _apply_column_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column aliases to standardize column names."""
        df = df.copy()
        columns_lower = {c.lower(): c for c in df.columns}

        for target, aliases in self.COLUMN_ALIASES.items():
            if target in df.columns:
                continue  # Already has the target column
            for alias in aliases:
                alias_lower = alias.lower()
                if alias_lower in columns_lower:
                    original_col = columns_lower[alias_lower]
                    df = df.rename(columns={original_col: target})
                    break

        return df

    def _validate_columns(self, df: pd.DataFrame) -> List[str]:
        """Validate that required columns are present."""
        warnings = []
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                msg = f"Required column '{col}' not found"
                if self._strict_validation:
                    raise ValueError(msg)
                warnings.append(msg)
        return warnings

    def _row_to_record(self, row: pd.Series, idx: int) -> Optional[UploadedRecord]:
        """Convert a DataFrame row to an UploadedRecord."""
        try:
            # Get or generate ID
            record_id = str(row.get('id', '') or uuid7())

            # Get required fields
            query = self._get_value(row, 'query', '')
            actual_output = self._get_value(row, 'actual_output', '')

            if not query or not actual_output:
                logger.warning(f'Row {idx}: missing required fields, skipping')
                return None

            # Get optional fields
            expected_output = self._get_value(row, 'expected_output', None)
            llm_score = self._get_int_value(row, 'llm_score', None)
            llm_reasoning = self._get_value(row, 'llm_reasoning', None)

            # Collect additional metadata
            metadata = {}
            known_columns = self.REQUIRED_COLUMNS | self.OPTIONAL_COLUMNS | {'id'}
            for col in row.index:
                if col not in known_columns:
                    val = row[col]
                    if pd.notna(val):
                        metadata[col] = val

            return UploadedRecord(
                id=record_id,
                query=query,
                actual_output=actual_output,
                expected_output=expected_output,
                llm_score=llm_score,
                llm_reasoning=llm_reasoning,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f'Row {idx}: failed to parse - {e}')
            return None

    @staticmethod
    def _get_value(row: pd.Series, key: str, default: Any) -> Any:
        """Get a value from a row, handling NaN."""
        val = row.get(key, default)
        if pd.isna(val):
            return default
        return str(val) if val is not None else default

    @staticmethod
    def _get_int_value(
        row: pd.Series, key: str, default: Optional[int]
    ) -> Optional[int]:
        """Get an integer value from a row, handling NaN and type conversion."""
        val = row.get(key, default)
        if pd.isna(val):
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default
