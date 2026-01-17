from typing import List

import numpy as np
from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.schema import SignalDescriptor

logger = get_logger(__name__)


class StatementGeneratorInput(RichBaseModel):
    """Input for the statement generator."""

    question: str = Field(..., description='The question being asked.')
    answer: str = Field(..., description='The answer to decompose.')
    model_config = {'extra': 'forbid'}


class StatementGeneratorOutput(RichBaseModel):
    """Output for the statement generator."""

    statements: List[str] = Field(
        ..., description='The list of atomic statements extracted.'
    )
    model_config = {'extra': 'forbid'}


class StatementGenerator(BaseMetric[StatementGeneratorInput, StatementGeneratorOutput]):
    """
    Decomposes an answer into atomic, standalone statements for factual verification.
    """

    instruction = """
    Analyze the 'Answer' in the context of the 'Question'.
    Extract ONLY the verifiable facts and claims regarding the subject matter.

    Rules:
    1. **Exclude Conversational Filler**: Ignore pleasantries (e.g., "I am happy to help"), persona statements (e.g., "I am excited"), and meta-commentary (e.g., "Here is the list").
    2. **Exclude Repetitions**: If a fact is stated twice, include it only once.
    3. **Atomic & Self-Contained**: Each statement must be a single fact with resolved pronouns (e.g., replace "it" with the noun).
    4. **Subject Matter Only**: Focus strictly on the core information requested by the user (products, features, definitions).
    """

    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput

    examples = [
        (
            StatementGeneratorInput(
                question='What does the infield fly rule do?',
                answer="I'm excited to help! The infield fly rule protects baserunners. It also prevents unfair double plays.",
            ),
            StatementGeneratorOutput(
                statements=[
                    'The infield fly rule protects baserunners.',
                    'The infield fly rule prevents unfair double plays.',
                ]
            ),
        ),
        (
            StatementGeneratorInput(
                question='What is a balk in baseball?',
                answer='Great question! A balk is an illegal pitching motion. It deceives baserunners. A balk results in all runners advancing one base.',
            ),
            StatementGeneratorOutput(
                statements=[
                    'A balk is an illegal pitching motion.',
                    'A balk deceives baserunners.',
                    'A balk results in all runners advancing one base.',
                ]
            ),
        ),
    ]


class StatementVerdict(RichBaseModel):
    statement: str = Field(..., description='The statement being evaluated.')
    is_supported: int = Field(
        ...,
        description='1 if the statement is strictly supported by ground truth, 0 otherwise.',
    )
    reason: str = Field(..., description='Brief reason for the verdict.')
    model_config = {'extra': 'forbid'}


class FactualityReport(RichBaseModel):
    verdicts: List[StatementVerdict] = Field(
        ..., description='Evaluation of each statement.'
    )
    model_config = {'extra': 'forbid'}


class FactualityJudgeInput(RichBaseModel):
    question: str = Field(..., description='The original question.')
    statements: List[str] = Field(
        ..., description='List of statements from the actual answer to verify.'
    )
    ground_truth: str = Field(
        ..., description='The expected answer/context to verify against.'
    )
    model_config = {'extra': 'forbid'}


class FactualityJudge(BaseMetric[FactualityJudgeInput, FactualityReport]):
    """
    Classifies a list of statements as 1 (Supported) or 0 (Unsupported).
    """

    instruction = """
    You are a strict fact-checker. 
    Evaluate each statement provided in the 'statements' list against the 'ground_truth' text.
    
    **Note on Ground Truth:**
    The ground truth may be unstructured text, structured JSON data, or a mix of both. You must parse it accordingly.

    **Evaluation Protocol:**
    1. **Search:** Locate the specific fact or data point in the ground truth.
    2. **Verify:** - **Supported (1):** The fact is explicitly present.
         - If checking text: The meaning must match.
         - If checking data/JSON: The values must be mathematically equivalent (e.g., "$2.5M" matches `2500000` or `2.5e6`).
    3. **Reject:**
       - **Unsupported (0):** The fact is missing, contradictory, or hallucinates a value not found in the source.

    **Guidelines:**
    - Ignore minor formatting differences (e.g., casing, punctuation, currency symbols).
    - Be strict on numbers: "5 employees" does NOT support "10 employees".
    """

    input_model = FactualityJudgeInput
    output_model = FactualityReport

    examples = [
        (
            FactualityJudgeInput(
                question='What is the sun?',
                statements=[
                    'The sun is powered by nuclear fission.',
                    'It provides light to the solar system.',
                ],
                ground_truth='The sun is a star powered by nuclear fusion that provides light and heat.',
            ),
            FactualityReport(
                verdicts=[
                    StatementVerdict(
                        statement='The sun is powered by nuclear fission.',
                        is_supported=0,
                        reason="Contradiction: Ground truth says 'nuclear fusion', not fission.",
                    ),
                    StatementVerdict(
                        statement='It provides light to the solar system.',
                        is_supported=1,
                        reason='Supported: Ground truth confirms it provides light.',
                    ),
                ]
            ),
        )
    ]


@metric(
    name='Factual Accuracy',
    key='factual_accuracy',
    description='Calculates the percentage of statements in the answer that are factually supported by the ground truth.',
    required_fields=['query', 'actual_output', 'expected_output'],
    default_threshold=0.8,
    tags=['knowledge', 'agent', 'single_turn'],
)
class FactualAccuracy(BaseMetric):
    """
    Factual Accuracy Metric.

    Process:
    1. Decompose 'actual_output' into atomic statements.
    2. specific binary check (1/0) for each statement against 'expected_output'.
    3. Score = (Sum of 1s) / (Total Statements).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.statement_generator = StatementGenerator(**kwargs)
        self.judge = FactualityJudge(**kwargs)

    @trace(name='FactualAccuracy', capture_args=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        self._validate_required_metric_fields(item)

        stmt_result = await self.statement_generator.execute(
            StatementGeneratorInput(question=item.query, answer=item.actual_output)
        )
        statements = stmt_result.statements

        if not statements:
            if not item.actual_output.strip():
                return MetricEvaluationResult(
                    score=np.nan, explanation='Agent provided no response.'
                )
            else:
                # Usually purely chit-chat is considered "factually safe".
                return MetricEvaluationResult(
                    score=1.0, explanation='No factual statements found to verify.'
                )

        report = await self.judge.execute(
            FactualityJudgeInput(
                question=item.query,
                statements=statements,
                ground_truth=item.expected_output,
            )
        )

        total = len(report.verdicts)
        passed = sum(v.is_supported for v in report.verdicts)

        score = passed / total if total > 0 else 0.0

        # Create explanation
        failed_statements = [
            v.statement for v in report.verdicts if v.is_supported == 0
        ]
        if failed_statements:
            explanation = f'Factual Accuracy: {score:.0%} ({passed}/{total} correct). Failed facts: {failed_statements[:2]}...'
        else:
            explanation = (
                f'Factual Accuracy: 100% ({total}/{total} statements verified).'
            )

        self.compute_cost_estimate([self.statement_generator, self.judge])

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=report,
        )

    @staticmethod
    def get_signals(report: FactualityReport) -> List[SignalDescriptor]:
        """Display the binary checklist in the UI."""
        signals = []

        # Summary Signal
        total = len(report.verdicts)
        passed = sum(v.is_supported for v in report.verdicts)
        score = passed / total if total > 0 else 0.0

        signals.append(
            SignalDescriptor(
                name='accuracy_score',
                extractor=lambda r: score,
                description=f'{passed} out of {total} statements were correct.',
                headline_display=True,
            )
        )

        # Per-Statement Signals
        for i, verdict in enumerate(report.verdicts):
            # Shorten statement for group title
            preview = (
                (verdict.statement[:60] + '...')
                if len(verdict.statement) > 60
                else verdict.statement
            )
            # Icon based on pass/fail
            status = 'Supported: ' if verdict.is_supported else 'Not Supported: '
            group = f'{status} {preview}'

            signals.extend(
                [
                    SignalDescriptor(
                        name='statement',
                        group=group,
                        extractor=lambda r, idx=i: r.verdicts[idx].statement,
                        description='The atomic statement extracted from the answer.',
                    ),
                    SignalDescriptor(
                        name='is_correct',
                        group=group,
                        extractor=lambda r, idx=i: r.verdicts[idx].is_supported,
                        description='1 if Supported, 0 if Unsupported.',
                        headline_display=True,  # Show the 1/0 directly in header
                    ),
                    SignalDescriptor(
                        name='reason',
                        group=group,
                        extractor=lambda r, idx=i: r.verdicts[idx].reason,
                        description='Why the judge marked this correct/incorrect.',
                    ),
                ]
            )

        return signals
