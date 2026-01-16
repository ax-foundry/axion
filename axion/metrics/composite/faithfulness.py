from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from axion._core.schema import RichBaseModel, RichEnum
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.cache import AnalysisCache
from axion.metrics.internals.engine import RAGAnalyzer
from axion.metrics.internals.schema import EvaluationMode, JudgedClaim
from axion.metrics.schema import SignalDescriptor


class FaithfulnessResult(RichBaseModel):
    """Structured result for the Faithfulness metric."""

    overall_score: float
    total_claims: int
    verdict_counts: Dict[str, int]
    judged_claims: List[JudgedClaim]
    analysis_metadata: Optional[dict] = None


class MetricVerdict(str, RichEnum):
    """Internal enum for classifying faithfulness verdicts within the metric."""

    FULLY_SUPPORTED = 'FULLY_SUPPORTED'
    PARTIALLY_SUPPORTED = 'PARTIALLY_SUPPORTED'
    NO_EVIDENCE = 'NO_EVIDENCE'
    CONTRADICTORY = 'CONTRADICTORY'


# Default scoring weights for different verdicts.
DEFAULT_VERDICT_SCORES: Dict[MetricVerdict, float] = {
    MetricVerdict.FULLY_SUPPORTED: 1.0,
    MetricVerdict.PARTIALLY_SUPPORTED: 0.5,
    MetricVerdict.NO_EVIDENCE: 0.0,
    MetricVerdict.CONTRADICTORY: -1.0,
}


@metric(
    name='Faithfulness',
    key='faithfulness',
    description='Evaluates if the answer is factually consistent with the retrieved context.',
    required_fields=['query', 'actual_output', 'retrieved_content'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['knowledge', 'single_turn'],
)
class Faithfulness(BaseMetric):
    """
    Measures how faithful the generated answer is to the retrieved context.
    Default Scoring: Fully Supported (+1.0), Partially Supported (+0.5), No Evidence (0.0), Contradictory (-1.0)
    """

    shares_internal_cache = True
    required_components = ['claim_faithfulness']

    def __init__(
        self,
        mode: EvaluationMode = EvaluationMode.GRANULAR,
        strict_mode: bool = False,
        verdict_scores: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Initializes the Faithfulness metric.
        Args:
            mode: The evaluation mode for the underlying RAG analyzer.
            strict_mode (bool): If True, sets 'NO_EVIDENCE' to -1.0, penalizing
                                uncited claims (hallucinations) as heavily as contradictions.
                                This is overridden by 'verdict_scores' if provided.
            verdict_scores: A dictionary to override the default scoring weights (e.g.,
                            {"CONTRADICTORY": -2.0, "PARTIALLY_SUPPORTED": 0.75}).
                            If provided, this takes precedence over 'strict_mode'.
            **kwargs: Additional keyword arguments passed to the RAGAnalyzer.
        """
        super().__init__(**kwargs)
        self.engine = RAGAnalyzer(mode=mode, **kwargs)
        self.strict_mode = strict_mode

        # Start with the defaults
        self.verdict_scores = DEFAULT_VERDICT_SCORES.copy()

        # Apply strict_mode *only if* verdict_scores is not provided
        if self.strict_mode and not verdict_scores:
            self.verdict_scores[MetricVerdict.NO_EVIDENCE] = -1.0

        # 3. Apply user's custom overrides (if they exist)
        if verdict_scores:
            for key, value in verdict_scores.items():
                try:
                    enum_key = MetricVerdict[key.upper()]
                    self.verdict_scores[enum_key] = value
                except KeyError:
                    raise ValueError(
                        f"'{key}' is not a valid FaithfulnessVerdict member. "
                        f'Valid keys are: {[m.name for m in MetricVerdict]}'
                    )

    def _get_verdict_score(self, claim: JudgedClaim) -> Tuple[str, float]:
        """Get verdict score for claim."""
        verdict_str = getattr(claim.faithfulness_verdict, 'value', None)
        verdict_str = verdict_str.upper().replace(' ', '_')
        verdict = MetricVerdict.from_str(verdict_str, default=MetricVerdict.NO_EVIDENCE)
        return verdict, self.verdict_scores.get(verdict, 0.0)

    @trace(name='Faithfulness', capture_args=True, capture_response=True)
    async def execute(
        self, item: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricEvaluationResult:
        analysis = await self.engine.execute(item, cache, self.required_components)
        self._validate_required_metric_fields(item)
        if analysis.has_error():
            return MetricEvaluationResult(
                score=np.nan, explanation=f'Analysis failed: {analysis.error}'
            )

        claims = analysis.judged_claims
        if not claims:
            return MetricEvaluationResult(
                score=1.0, explanation='No claims extracted; assuming faithful.'
            )

        score_sum = 0.0
        verdict_counts = {v: 0 for v in MetricVerdict}

        for claim in claims:
            verdict, score = self._get_verdict_score(claim)
            verdict_counts[verdict] += 1
            score_sum += score

        # Compute final score, clamped between 0 and 1
        score = max(0.0, min(1.0, score_sum / len(claims) if claims else 0.0))

        result_data = FaithfulnessResult(
            overall_score=score,
            total_claims=len(claims),
            verdict_counts={v.value.lower(): c for v, c in verdict_counts.items()},
            judged_claims=claims,
        )

        if analysis.metadata:
            self.cost_estimate = getattr(analysis.metadata, 'cost_estimate', 0)

        # Changed to instance method to access self.verdict_scores
        summary = self._generate_summary(verdict_counts, len(claims), score)

        return MetricEvaluationResult(
            score=score,
            explanation=summary,
            signals=result_data,
            metadata={
                'analysis_metadata': analysis.metadata.model_dump()
                if analysis.metadata
                else {}
            },
        )

    def _generate_summary(
        self, counts: Dict[MetricVerdict, int], total: int, score: float
    ) -> str:
        """Generates a summary that is aware of the scoring configuration."""
        contradictory_count = counts.get(MetricVerdict.CONTRADICTORY, 0)
        no_evidence_count = counts.get(MetricVerdict.NO_EVIDENCE, 0)

        # Check if NO_EVIDENCE is penalized
        no_evidence_is_penalized = (
            self.verdict_scores.get(MetricVerdict.NO_EVIDENCE, 0.0) < 0.0
        )

        if contradictory_count > 0:
            return (
                f'CRITICAL: Found {contradictory_count} contradictory claim(s) out of {total}. '
                f'Score penalized to {score:.2f}.'
            )

        if no_evidence_count > 0 and no_evidence_is_penalized:
            return (
                f'CRITICAL: Found {no_evidence_count} uncited claim(s) (no evidence) out of {total}. '
                f'Strict mode is enabled, penalizing the score to {score:.2f}.'
            )

        fully_supported_count = counts.get(MetricVerdict.FULLY_SUPPORTED, 0)

        if score >= 0.9:
            return f'Excellent: {fully_supported_count}/{total} claims fully supported by context.'
        elif score >= 0.7:
            return (
                f'Good: Most claims are supported, '
                f'but {no_evidence_count} have no evidence.'
            )
        else:
            return (
                f'Poor: {no_evidence_count}/{total} claims had no evidence in the context, '
                f'leading to a score of {score:.2f}.'
            )

    def _get_score_calculation_str(
        self, result: FaithfulnessResult, total_claims: int
    ) -> str:
        parts = []
        for verdict, score in self.verdict_scores.items():
            count = result.verdict_counts.get(verdict.value.lower(), 0)
            parts.append(f'({count} * {score:.1f})')

        formula = ' + '.join(parts)
        return f'max(0.0, ({formula}) / {total_claims})'

    def get_signals(self, result: FaithfulnessResult) -> List[SignalDescriptor]:
        total_claims = result.total_claims or 1

        # This mapping will now dynamically reflect strict_mode
        verdict_score_mapping = {
            verdict.name: score for verdict, score in self.verdict_scores.items()
        }

        signals = [
            SignalDescriptor(
                name='final_score',
                extractor=lambda r: r.overall_score,
                headline_display=True,
            ),
            SignalDescriptor(
                name='score_calculation',
                extractor=lambda r: self._get_score_calculation_str(r, total_claims),
            ),
            SignalDescriptor(name='total_claims', extractor=lambda r: r.total_claims),
            *[
                SignalDescriptor(
                    name=f'{verdict.value.lower()}_claims',
                    extractor=lambda r, v=verdict.value.lower(): r.verdict_counts.get(
                        v, 0
                    ),
                )
                for verdict in MetricVerdict
            ],
        ]

        for i, claim in enumerate(result.judged_claims):
            group_name = f'claim_{i}: "{claim.claim_text[:80]}..."'

            signals.extend(
                [
                    SignalDescriptor(
                        name='faithfulness_verdict',
                        group=group_name,
                        extractor=lambda r, idx=i: self._get_verdict_score(
                            r.judged_claims[idx]
                        )[0].value,
                        description='Verdict on how well this claim is supported by the context.',
                        headline_display=True,
                        score_mapping=verdict_score_mapping,
                    ),
                    SignalDescriptor(
                        name='verdict_score',
                        group=group_name,
                        extractor=lambda r, idx=i: self._get_verdict_score(
                            r.judged_claims[idx]
                        )[1],
                        description='Numerical score for this claim based on the verdict.',
                    ),
                    SignalDescriptor(
                        name='claim_text',
                        group=group_name,
                        extractor=lambda r, idx=i: r.judged_claims[idx].claim_text,
                    ),
                    SignalDescriptor(
                        name='reasoning',
                        group=group_name,
                        extractor=lambda r, idx=i: getattr(
                            r.judged_claims[idx].faithfulness_verdict, 'reason', 'N/A'
                        ),
                    ),
                ]
            )
        return signals

    def get_diagnostic_data(
        self,
        result: FaithfulnessResult,
        mode: Literal['failures', 'successes', 'all'] = 'failures',
    ) -> List[Dict[str, Any]]:
        """Extract claims for analysis or prompt learning.

        Args:
            result: The FaithfulnessResult from metric execution.
            mode: What to extract:
                - 'failures': Only claims not FULLY_SUPPORTED (default)
                - 'successes': Only FULLY_SUPPORTED claims
                - 'all': All claims with 'passed' field

        Returns:
            List of dicts with claim details.
            When mode='all', includes 'passed' boolean.
        """
        data = []
        for claim in result.judged_claims:
            verdict = claim.faithfulness_verdict
            verdict_value = getattr(verdict, 'value', None)

            # FULLY_SUPPORTED = success, everything else = failure
            passed = verdict_value and verdict_value.upper() == 'FULLY_SUPPORTED'

            # Filter based on mode
            if mode == 'failures' and passed:
                continue
            if mode == 'successes' and not passed:
                continue

            item = {
                'claim': claim.claim_text,
                'verdict': verdict_value,
                'reason': getattr(verdict, 'reason', 'N/A'),
            }

            if mode == 'all':
                item['passed'] = passed

            data.append(item)
        return data
