from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from axion.dataset import Dataset, DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult
from axion.runners import MetricRunner
from axion.schema import TestResult
from axion._core.metrics_utils import (
    cohen_kappa_score,
    confusion_matrix_binary,
    f1_score,
    precision_score,
    recall_score,
)


class AlignMetric(BaseMetric):
    """
    Dynamically configured metric from the Align Eval.
    """

    def __init__(
        self,
        instruction: str,
        model_name: str = None,
        llm_provider: str = None,
        examples: List[Dict] = None,
        required_fields: List[str] = None,
        **kwargs,
    ):
        """
        Initializes the metric with configuration from Align Eval.

        Args:
            instruction (str): The LLM-as-a-judge prompt.
            model_name (str, optional): The name of the LLM to use.
            llm_provider (str, optional): The provider of the LLM.
            examples (List[Dict], optional): Few-shot examples from the UI.
            required_fields (List[str], optional): Required input fields for the metric.
        """
        self.instruction = instruction
        super().__init__(
            model_name=model_name,
            llm_provider=llm_provider,
            required_fields=required_fields,
            **kwargs,
        )

        if examples:
            self.examples = [
                (
                    DatasetItem(**example['input']),
                    MetricEvaluationResult(**example['output']),
                )
                for example in examples
            ]
        else:
            self.examples = []


class BaseAlignEval(ABC):
    """Base class for alignment evaluation, containing core, non-UI logic."""

    def __init__(self, dataset: Dataset, metric: BaseMetric):
        if not isinstance(dataset, Dataset):
            raise TypeError('`dataset` must be an instance of the Dataset class.')
        if not isinstance(metric, BaseMetric):
            raise TypeError('`metric` must be an instance of a BaseMetric class.')

        self.dataset = dataset
        self.metric = metric
        self.llm_evaluations: List[Dict[str, Any]] = []
        self.results_df: pd.DataFrame = pd.DataFrame()
        self.alignment_score: float = 0.0

    async def _run_evals_async(self) -> None:
        """
        Runs the LLM-as-a-judge metric over the dataset and stores the results.
        """
        runner = MetricRunner(metrics=[self.metric], max_concurrent=5)
        test_results: List[TestResult] = await runner.execute_batch(self.dataset.items)

        self.llm_evaluations = []
        for test_result in test_results:
            item_id = test_result.test_case.id
            if test_result.score_results:
                metric_result = test_result.score_results[0]
                self.llm_evaluations.append(
                    {
                        'id': item_id,
                        'llm_score': metric_result.score,
                        'llm_explanation': metric_result.explanation,
                    }
                )
            else:
                # Log warning if subclass has logging capability
                self._log_warning(
                    f"Metric '{self.metric.name}' failed to produce a score for item '{item_id}'."
                )
                self.llm_evaluations.append(
                    {
                        'id': item_id,
                        'llm_score': np.nan,
                        'llm_explanation': 'Metric execution failed or produced no score.',
                    }
                )

    def _prepare_results_df(self) -> None:
        """
        Builds the results table, calculates the alignment score,
        and stores them as instance attributes.
        """
        human_scores_data = []
        for item in self.dataset.items:
            if item.judgment is None or str(item.judgment).strip() not in ['0', '1']:
                raise ValueError(
                    f"DatasetItem with id '{getattr(item, 'id', 'N/A')}' is missing a judgment. "
                    "Please ensure all items have judgment with values '0' or '1'."
                )
            human_scores_data.append({'id': item.id, 'human_score': int(item.judgment)})

        human_df = pd.DataFrame(human_scores_data)
        llm_df = pd.DataFrame(self.llm_evaluations)

        # Create dataset dataframe from items
        dataset_data = [
            {
                'id': item.id,
                'query': item.query,
                'expected_output': item.expected_output,
                'actual_output': item.actual_output,
                'judgment': item.judgment,
            }
            for item in self.dataset.items
        ]
        dataset_df = pd.DataFrame(dataset_data)

        # Merge all dataframes
        merged_df = pd.merge(dataset_df, human_df, on='id')
        results_df = pd.merge(merged_df, llm_df, on='id')

        # Process and calculate alignment metrics
        results_copy = results_df.copy()
        results_copy['human_score'] = pd.to_numeric(results_copy['human_score'])
        results_copy['llm_score'] = pd.to_numeric(results_copy['llm_score'])

        # Calculate alignment score and additional metrics
        valid_comparisons = results_copy.dropna(subset=['llm_score'])

        if len(valid_comparisons) > 0:
            # Basic alignment score (accuracy)
            correct_predictions = (
                valid_comparisons['human_score'] == valid_comparisons['llm_score']
            ).sum()
            total_valid = len(valid_comparisons)
            self.alignment_score = correct_predictions / total_valid

            # Calculate comprehensive binary classification metrics
            self._calculate_binary_metrics(valid_comparisons)
        else:
            self.alignment_score = 0.0
            self._initialize_empty_metrics()

        # Add alignment and difference columns
        results_copy['aligned'] = (
            results_copy['human_score'] == results_copy['llm_score']
        )
        results_copy['score_difference'] = abs(
            results_copy['human_score'] - results_copy['llm_score']
        )

        # Reorder and rename columns for final output
        self.results_df = results_copy.rename(
            columns={'judgment': 'human_judgment_source'}
        )
        print('self.results_df', self.results_df)
        self.results_df = self.results_df[
            [
                'id',
                'aligned',
                'human_score',
                'llm_score',
                'score_difference',
                'query',
                'actual_output',
                'expected_output',
                'llm_explanation',
                'human_judgment_source',
            ]
        ]

    def _calculate_binary_metrics(self, valid_comparisons: pd.DataFrame) -> None:
        """
        Calculate comprehensive binary classification metrics.

        Args:
            valid_comparisons: DataFrame with valid human and LLM scores
        """
        y_true = valid_comparisons['human_score'].values
        y_pred = valid_comparisons['llm_score'].values

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)

        # Store confusion matrix counts
        self.true_negatives = int(tn)
        self.false_positives = int(fp)
        self.false_negatives = int(fn)
        self.true_positives = int(tp)

        # Calculate metrics with zero_division handling
        self.precision = precision_score(y_true, y_pred, zero_division=0.0)
        self.recall = recall_score(y_true, y_pred, zero_division=0.0)
        self.f1_score = f1_score(y_true, y_pred, zero_division=0.0)

        # Calculate Cohen's kappa
        self.cohen_kappa = cohen_kappa_score(y_true, y_pred)

        # Calculate specificity (true negative rate)
        self.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def _initialize_empty_metrics(self) -> None:
        """Initialize metrics when no valid comparisons are available."""
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.cohen_kappa = 0.0
        self.specificity = 0.0

    def get_alignment_summary(self) -> Dict[str, Any]:
        """
        Returns a comprehensive summary of alignment results including all binary classification metrics.

        Returns:
            Dict containing alignment metrics, confusion matrix, and classification statistics.
        """
        if self.results_df.empty:
            raise RuntimeError('No results available. Please run execute() first.')

        total_items = len(self.results_df)
        aligned_items = self.results_df['aligned'].sum()
        misaligned_items = total_items - aligned_items

        return {
            # Basic alignment metrics
            'alignment_score': self.alignment_score,
            'total_items': total_items,
            'aligned_items': int(aligned_items),
            'misaligned_items': int(misaligned_items),
            'alignment_percentage': self.alignment_score * 100,
            # Binary classification metrics
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'cohen_kappa': self.cohen_kappa,
            'specificity': self.specificity,
            # Confusion matrix counts
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            # Additional derived metrics
            'sensitivity': self.recall,  # Alias for recall
            'positive_predictive_value': self.precision,  # Alias for precision
            'negative_predictive_value': (
                self.true_negatives / (self.true_negatives + self.false_negatives)
                if (self.true_negatives + self.false_negatives) > 0
                else 0.0
            ),
        }

    def get_confusion_matrix_df(self) -> pd.DataFrame:
        """
        Returns the confusion matrix as a formatted DataFrame.

        Returns:
            DataFrame with confusion matrix in standard format
        """
        if self.results_df.empty:
            raise RuntimeError('No results available. Please run execute() first.')

        confusion_data = [
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives],
        ]

        return pd.DataFrame(
            confusion_data, columns=['Human=0', 'Human=1'], index=['LLM=0', 'LLM=1']
        )

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret Cohen's kappa value."""
        if kappa < 0:
            return 'Poor agreement'
        elif kappa < 0.20:
            return 'Slight agreement'
        elif kappa < 0.40:
            return 'Fair agreement'
        elif kappa < 0.60:
            return 'Moderate agreement'
        elif kappa < 0.80:
            return 'Substantial agreement'
        else:
            return 'Almost perfect agreement'

    def get_metrics_df(self) -> pd.DataFrame:
        """
        Returns all calculated metrics as a formatted DataFrame.

        Returns:
            DataFrame with metric names and values
        """
        if self.results_df.empty:
            raise RuntimeError('No results available. Please run execute() first.')

        metrics_data = [
            [
                'Accuracy (Alignment)',
                f'{self.alignment_score:.3f}',
                f'{self.alignment_score * 100:.1f}%',
            ],
            ['Precision', f'{self.precision:.3f}', f'{self.precision * 100:.1f}%'],
            ['Recall (Sensitivity)', f'{self.recall:.3f}', f'{self.recall * 100:.1f}%'],
            [
                'Specificity',
                f'{self.specificity:.3f}',
                f'{self.specificity * 100:.1f}%',
            ],
            ['F1 Score', f'{self.f1_score:.3f}', f'{self.f1_score * 100:.1f}%'],
            [
                "Cohen's Kappa",
                f'{self.cohen_kappa:.3f}',
                self._interpret_kappa(self.cohen_kappa),
            ],
        ]

        return pd.DataFrame(
            metrics_data, columns=['Metric', 'Value', 'Percentage/Interpretation']
        )

    def get_misaligned_items(self) -> pd.DataFrame:
        """
        Returns only the items where human and LLM scores don't align.

        Returns:
            DataFrame containing only misaligned items.
        """
        if self.results_df.empty:
            raise RuntimeError('No results available. Please run execute() first.')

        return self.results_df[~self.results_df['aligned']].copy()

    def get_classification_report(self) -> str:
        """
        Returns a formatted classification report string.

        Returns:
            String containing detailed classification metrics
        """
        if self.results_df.empty:
            raise RuntimeError('No results available. Please run execute() first.')

        report = f"""
Classification Report
====================

Confusion Matrix:
                 Predicted
                 0      1
Actual    0    {self.true_negatives:3d}    {self.false_positives:3d}
          1    {self.false_negatives:3d}    {self.true_positives:3d}

Metrics:
--------
Accuracy:     {self.alignment_score:.3f} ({self.alignment_score * 100:.1f}%)
Precision:    {self.precision:.3f} ({self.precision * 100:.1f}%)
Recall:       {self.recall:.3f} ({self.recall * 100:.1f}%)
Specificity:  {self.specificity:.3f} ({self.specificity * 100:.1f}%)
F1 Score:     {self.f1_score:.3f} ({self.f1_score * 100:.1f}%)
Cohen's κ:    {self.cohen_kappa:.3f} ({self._interpret_kappa(self.cohen_kappa)})

Counts:
-------
True Positives:  {self.true_positives}
True Negatives:  {self.true_negatives}
False Positives: {self.false_positives}
False Negatives: {self.false_negatives}
Total Items:     {len(self.results_df)}
"""
        return report

    def _log_warning(self, message: str) -> None:
        """
        Log a warning message. Override in subclasses for custom logging.
        """
        pass

    @abstractmethod
    def execute(self) -> pd.DataFrame:
        """
        Execute the alignment evaluation workflow.
        """
        pass
