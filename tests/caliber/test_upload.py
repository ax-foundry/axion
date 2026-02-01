"""Tests for caliber upload handler."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from axion.caliber.upload import UploadHandler


class TestUploadHandler:
    """Tests for UploadHandler."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                'id': ['r1', 'r2', 'r3'],
                'query': ['Q1', 'Q2', 'Q3'],
                'actual_output': ['A1', 'A2', 'A3'],
                'expected_output': ['E1', 'E2', 'E3'],
            }
        )

    @pytest.fixture
    def handler(self):
        """Create upload handler."""
        return UploadHandler()

    def test_from_dataframe(self, handler, sample_df):
        """Test loading from DataFrame."""
        result = handler.from_dataframe(sample_df)
        assert result.total_count == 3
        assert len(result.records) == 3
        assert result.records[0].id == 'r1'
        assert result.records[0].query == 'Q1'
        assert not result.has_llm_scores

    def test_from_dataframe_with_llm_scores(self, handler):
        """Test loading DataFrame with LLM scores."""
        df = pd.DataFrame(
            {
                'id': ['r1', 'r2'],
                'query': ['Q1', 'Q2'],
                'actual_output': ['A1', 'A2'],
                'llm_score': [1, 0],
            }
        )
        result = handler.from_dataframe(df)
        assert result.has_llm_scores
        assert result.records[0].llm_score == 1
        assert result.records[1].llm_score == 0

    def test_from_records(self, handler):
        """Test loading from list of dicts."""
        records = [
            {'id': 'r1', 'query': 'Q1', 'actual_output': 'A1'},
            {'id': 'r2', 'query': 'Q2', 'actual_output': 'A2'},
        ]
        result = handler.from_records(records)
        assert result.total_count == 2
        assert result.records[0].id == 'r1'

    def test_from_csv(self, handler, sample_df):
        """Test loading from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df.to_csv(f, index=False)
            temp_path = f.name

        try:
            result = handler.from_csv(temp_path)
            assert result.total_count == 3
        finally:
            Path(temp_path).unlink()

    def test_column_aliases(self, handler):
        """Test column alias handling."""
        df = pd.DataFrame(
            {
                'id': ['r1'],
                'input': ['Q1'],  # alias for query
                'response': ['A1'],  # alias for actual_output
            }
        )
        result = handler.from_dataframe(df)
        assert result.total_count == 1
        assert result.records[0].query == 'Q1'
        assert result.records[0].actual_output == 'A1'

    def test_missing_required_columns(self, handler):
        """Test warning for missing required columns."""
        df = pd.DataFrame({'id': ['r1'], 'query': ['Q1']})  # missing actual_output
        result = handler.from_dataframe(df)
        assert len(result.validation_warnings) > 0

    def test_metadata_collection(self, handler):
        """Test that extra columns become metadata."""
        df = pd.DataFrame(
            {
                'id': ['r1'],
                'query': ['Q1'],
                'actual_output': ['A1'],
                'custom_field': ['custom_value'],
            }
        )
        result = handler.from_dataframe(df)
        assert result.records[0].metadata.get('custom_field') == 'custom_value'

    def test_auto_generate_id(self, handler):
        """Test ID auto-generation when missing."""
        df = pd.DataFrame({'query': ['Q1'], 'actual_output': ['A1']})
        result = handler.from_dataframe(df)
        assert result.records[0].id is not None
        assert len(result.records[0].id) > 0

    def test_validate_records(self, handler, sample_df):
        """Test record validation."""
        result = handler.from_dataframe(sample_df)
        warnings = handler.validate(result.records)
        assert len(warnings) == 0  # Valid records should have no warnings
