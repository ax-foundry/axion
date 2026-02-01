"""
Evaluation handling for CaliberHQ workflow (Step 3).

Runs LLM-as-judge evaluation and computes alignment metrics.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from axion._core.logging import get_logger
from axion._core.metrics_utils import (
    cohen_kappa_score,
    confusion_matrix_binary,
    f1_score,
    precision_score,
    recall_score,
)
from axion.caliber.models import (
    AlignmentMetrics,
    Annotation,
    EvaluationConfig,
    EvaluationRecord,
    EvaluationResult,
    UploadedRecord,
)
from axion.dataset import Dataset, DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult
from axion.runners import MetricRunner
from axion.schema import TestResult

logger = get_logger(__name__)


class CaliberMetric(BaseMetric):
    """
    Dynamically configured metric for CaliberHQ evaluation.

    This metric uses the provided criteria/instruction for LLM-as-judge evaluation.
    """

    def __init__(
        self,
        instruction: str,
        model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        examples: Optional[List[Dict]] = None,
        required_fields: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the CaliberMetric.

        Args:
            instruction: The LLM-as-a-judge prompt/criteria
            model_name: The name of the LLM to use
            llm_provider: The provider of the LLM
            examples: Few-shot examples from the UI
            required_fields: Required input fields for the metric
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


class EvaluationRunner:
    """
    Runs LLM-as-judge evaluation for the CaliberHQ workflow.

    Executes the LLM judge on records and computes alignment metrics
    by comparing LLM scores with human annotations.

    Example:
        >>> config = EvaluationConfig(criteria="Score 1 if accurate, 0 otherwise")
        >>> runner = EvaluationRunner(config)
        >>>
        >>> result = await runner.run(records, annotations)
        >>> print(f"Accuracy: {result.metrics.accuracy:.1%}")
    """

    def __init__(
        self,
        config: EvaluationConfig,
        max_concurrent: int = 5,
    ):
        """
        Initialize the evaluation runner.

        Args:
            config: Evaluation configuration with criteria and model settings
            max_concurrent: Maximum concurrent evaluations
        """
        self._config = config
        self._max_concurrent = max_concurrent
        self._metric: Optional[CaliberMetric] = None

    async def run(
        self,
        records: List[UploadedRecord],
        annotations: Dict[str, Annotation],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationResult:
        """
        Run LLM-as-judge evaluation.

        Args:
            records: Records to evaluate
            annotations: Human annotations (record_id -> Annotation)
            on_progress: Optional callback for progress updates (current, total)

        Returns:
            EvaluationResult with metrics and individual results
        """
        # Create metric from config
        self._metric = CaliberMetric(
            instruction=self._config.criteria,
            model_name=self._config.model_name,
            llm_provider=self._config.llm_provider,
        )

        # Convert records to DatasetItems
        dataset_items = self._records_to_dataset_items(records, annotations)
        dataset = Dataset(name='caliber_evaluation', items=dataset_items)

        # Run LLM evaluations
        llm_evaluations = await self._run_evaluations(dataset, on_progress)

        # Build evaluation records
        eval_records = self._build_evaluation_records(
            records, annotations, llm_evaluations
        )

        # Compute metrics
        metrics = self._compute_metrics(eval_records)

        # Build confusion matrix
        confusion_matrix = self._build_confusion_matrix(eval_records)

        return EvaluationResult(
            records=eval_records,
            metrics=metrics,
            confusion_matrix=confusion_matrix,
            config=self._config,
        )

    def run_sync(
        self,
        records: List[UploadedRecord],
        annotations: Dict[str, Annotation],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationResult:
        """
        Synchronous wrapper for run().

        Args:
            records: Records to evaluate
            annotations: Human annotations
            on_progress: Optional progress callback

        Returns:
            EvaluationResult with metrics and individual results
        """
        from axion._core.asyncio import run_async_function

        async def _run():
            return await self.run(records, annotations, on_progress)

        return run_async_function(_run)

    async def _run_evaluations(
        self,
        dataset: Dataset,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run LLM evaluations on the dataset."""
        runner = MetricRunner(
            metrics=[self._metric], max_concurrent=self._max_concurrent
        )
        test_results: List[TestResult] = await runner.execute_batch(dataset.items)

        evaluations = {}
        total = len(test_results)

        for index, test_result in enumerate(test_results, start=1):
            item_id = test_result.test_case.id

            if test_result.score_results:
                metric_result = test_result.score_results[0]
                llm_score, score_note = self._coerce_binary_score(
                    metric_result.score, item_id
                )
                explanation = metric_result.explanation
                if score_note:
                    explanation = (
                        f'{explanation} ({score_note})' if explanation else score_note
                    )
                evaluations[item_id] = {
                    'llm_score': llm_score,
                    'llm_reasoning': explanation,
                }
            else:
                logger.warning(f"Metric failed to produce a score for item '{item_id}'")
                evaluations[item_id] = {
                    'llm_score': 0,
                    'llm_reasoning': 'Metric execution failed or produced no score.',
                }

            if on_progress:
                on_progress(index, total)

        return evaluations

    def _records_to_dataset_items(
        self,
        records: List[UploadedRecord],
        annotations: Dict[str, Annotation],
    ) -> List[DatasetItem]:
        """Convert UploadedRecords to DatasetItems with judgments."""
        items = []
        for record in records:
            annotation = annotations.get(record.id)
            if not annotation:
                logger.warning(f"No annotation for record '{record.id}', skipping")
                continue

            item = DatasetItem(
                id=record.id,
                query=record.query,
                actual_output=record.actual_output,
                expected_output=record.expected_output,
                judgment=str(annotation.score),
            )
            items.append(item)

        return items

    def _build_evaluation_records(
        self,
        records: List[UploadedRecord],
        annotations: Dict[str, Annotation],
        llm_evaluations: Dict[str, Dict[str, Any]],
    ) -> List[EvaluationRecord]:
        """Build EvaluationRecord list from results."""
        eval_records = []

        for record in records:
            annotation = annotations.get(record.id)
            llm_eval = llm_evaluations.get(record.id)

            if not annotation or not llm_eval:
                continue

            human_score = annotation.score
            llm_score = llm_eval['llm_score']
            aligned = human_score == llm_score

            eval_records.append(
                EvaluationRecord(
                    record_id=record.id,
                    human_score=human_score,
                    llm_score=llm_score,
                    llm_reasoning=llm_eval.get('llm_reasoning'),
                    aligned=aligned,
                    score_difference=abs(human_score - llm_score),
                )
            )

        return eval_records

    def _compute_metrics(self, records: List[EvaluationRecord]) -> AlignmentMetrics:
        """Compute alignment metrics from evaluation records."""
        if not records:
            return AlignmentMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                cohen_kappa=0.0,
                specificity=0.0,
                true_positives=0,
                true_negatives=0,
                false_positives=0,
                false_negatives=0,
            )

        y_true = [r.human_score for r in records]
        y_pred = [r.llm_score for r in records]

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)

        # Basic metrics
        accuracy = sum(r.aligned for r in records) / len(records)
        precision = precision_score(y_true, y_pred, zero_division=0.0)
        recall = recall_score(y_true, y_pred, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
        kappa = cohen_kappa_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return AlignmentMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            cohen_kappa=kappa,
            specificity=specificity,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
        )

    def _build_confusion_matrix(
        self, records: List[EvaluationRecord]
    ) -> Dict[str, Dict[str, int]]:
        """Build confusion matrix as nested dict."""
        if not records:
            return {
                'LLM=0': {'Human=0': 0, 'Human=1': 0},
                'LLM=1': {'Human=0': 0, 'Human=1': 0},
            }

        y_true = [r.human_score for r in records]
        y_pred = [r.llm_score for r in records]
        tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)

        return {
            'LLM=0': {'Human=0': int(tn), 'Human=1': int(fn)},
            'LLM=1': {'Human=0': int(fp), 'Human=1': int(tp)},
        }

    def _coerce_binary_score(
        self, score: Any, item_id: str
    ) -> tuple[int, Optional[str]]:
        """Coerce a score to binary (0 or 1)."""
        if score in (0, 1):
            return int(score), None
        if isinstance(score, str) and score.strip() in ['0', '1']:
            return int(score.strip()), None
        if score is None or (isinstance(score, float) and np.isnan(score)):
            note = 'Missing score coerced to 0.'
        else:
            note = f"Invalid score '{score}' coerced to 0."
        logger.warning(f"Invalid score for item '{item_id}': {score}")
        return 0, note
