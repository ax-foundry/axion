from typing import List, Optional

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


class RedundantSegment(RichBaseModel):
    """A specific segment of text identified as unnecessary or verbose."""

    segment: str = Field(
        ..., description='The specific text that is redundant or verbose.'
    )
    reason: str = Field(
        ...,
        description="Why this segment is considered unnecessary (e.g., 'Repetition', 'Fluff').",
    )
    category: str = Field(
        ...,
        description="Category of verbosity: 'Filler', 'Repetition', 'Over-explanation'.",
    )

    model_config = {'extra': 'forbid'}


class ConcisenessJudgeInput(RichBaseModel):
    """Input for the internal conciseness judge."""

    query: Optional[str] = Field(None, description='The user query.')
    actual_output: str = Field(..., description="The agent's response.")
    expected_output: str = Field(
        ..., description='The ground truth response (baseline for ideal length).'
    )

    model_config = {'extra': 'forbid'}


class ConcisenessJudgeOutput(RichBaseModel):
    """Output from the internal conciseness judge."""

    score: float = Field(
        ...,
        description='0.0 to 1.0 score (1.0 = Perfectly Concise, 0.0 = Extremely Verbose).',
    )

    redundant_segments: List[RedundantSegment] = Field(
        ...,
        description='List of specific verbose segments found. Return an empty list if none are found.',
    )

    reason: str = Field(..., description='Overall summary reasoning for the score.')

    model_config = {'extra': 'forbid'}


class ConcisenessJudge(BaseMetric[ConcisenessJudgeInput, ConcisenessJudgeOutput]):
    """
    Internal LLM Judge that evaluates the efficiency of language.
    """

    instruction = """
    You are an expert Editor evaluating the **Conciseness** of an AI response.
    Compare the 'Actual Output' against the 'Expected Output'.

    Your Goal: Determine if the Actual Output is efficient or if it contains "fluff."

    ### Evaluation Rules:
    1. **Baseline**: The 'Expected Output' represents the ideal information density.
    2. **Penalize**:
        - **Conversational Filler**: Excessive "I hope you are having a wonderful day" if not present in Expected.
        - **Repetition**: Repeating the same fact multiple times.
        - **Over-explanation**: Explaining concepts the user didn't ask for (if Expected didn't include them).
    3. **Do Not Penalize**:
        - Essential information that is also in the Expected Output.
        - Formatting changes (bullets vs paragraphs) unless they significantly bloat the text.

    ### Scoring Rubric:
    - **1.0 (Concise)**: Efficient, direct. Similar information density to Expected.
    - **0.8 (Acceptable)**: Slightly wordy (e.g., one extra sentence of politeness), but readable.
    - **0.5 (Verbose)**: Noticeable fluff. Takes 2x words to say what Expected said.
    - **0.0 (Bloated)**: Painful to read. Buries the answer in paragraphs of filler.

    Return the score and extract specific segments that are redundant.
    """

    input_model = ConcisenessJudgeInput
    output_model = ConcisenessJudgeOutput

    examples = [
        (
            ConcisenessJudgeInput(
                query='How do I reset my password?',
                expected_output="Go to Settings > Security and click 'Reset Password'.",
                actual_output="I can certainly help with that! To reset your password, you will want to navigate over to the Settings page. Once there, look for the Security tab. Then, simply click the button labeled 'Reset Password'.",
            ),
            ConcisenessJudgeOutput(
                score=0.6,
                reason='The response is overly wordy and uses too many filler phrases compared to the direct expected answer.',
                redundant_segments=[
                    RedundantSegment(
                        segment='I can certainly help with that!',
                        reason='Unnecessary pleasantry',
                        category='Filler',
                    ),
                    RedundantSegment(
                        segment='you will want to navigate over to',
                        reason='Wordy phrasing',
                        category='Over-explanation',
                    ),
                    RedundantSegment(
                        segment='simply', reason='Filler word', category='Filler'
                    ),
                ],
            ),
        ),
        (
            ConcisenessJudgeInput(
                query='What is the capital of France?',
                expected_output='Paris.',
                actual_output='Paris.',
            ),
            ConcisenessJudgeOutput(
                score=1.0, reason='Perfectly concise.', redundant_segments=[]
            ),
        ),
        (
            ConcisenessJudgeInput(
                query='Explain the refund policy.',
                expected_output='Refunds are available within 30 days of purchase. Contact support to initiate.',
                actual_output='Our refund policy is designed to be flexible. You can request a refund. This is possible within 30 days. You need to buy the item first. Then contact support.',
            ),
            ConcisenessJudgeOutput(
                score=0.5,
                reason='The response breaks simple concepts into choppy, repetitive sentences.',
                redundant_segments=[
                    RedundantSegment(
                        segment='Our refund policy is designed to be flexible.',
                        reason='Marketing fluff not in expected',
                        category='Filler',
                    ),
                    RedundantSegment(
                        segment='You need to buy the item first.',
                        reason='Obvious logic statement',
                        category='Over-explanation',
                    ),
                ],
            ),
        ),
    ]


@metric(
    name='Answer Conciseness',
    key='answer_conciseness',
    description='Evaluates if the response is concise or verbose compared to the expected answer, identifying specific redundant phrases.',
    required_fields=['actual_output', 'expected_output'],
    optional_fields=['query'],
    default_threshold=0.8,
    tags=['knowledge', 'agent', 'single_turn'],
)
class AnswerConciseness(BaseMetric):
    """
    Measures the Signal-to-Noise ratio of the response using an LLM.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.judge = ConcisenessJudge(**kwargs)

    @trace(name='AnswerConciseness', capture_args=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        self._validate_required_metric_fields(item)

        result = await self.judge.execute(
            ConcisenessJudgeInput(
                query=item.query,
                actual_output=item.actual_output,
                expected_output=item.expected_output,
            )
        )

        self.compute_cost_estimate([self.judge])

        return MetricEvaluationResult(
            score=result.score,
            explanation=f'Conciseness Score: {result.score:.2f}. {result.reason}',
            signals=result,
        )

    @staticmethod
    def get_signals(result: ConcisenessJudgeOutput) -> List[SignalDescriptor]:
        """Convert structured results into UI signals."""
        signals = [
            SignalDescriptor(
                name='conciseness_score',
                extractor=lambda r: r.score,
                description='Score measuring efficiency of language (1.0 = Concise, 0.0 = Verbose).',
                headline_display=True,
            ),
            SignalDescriptor(
                name='reason',
                extractor=lambda r: r.reason,
                description='Summary of why the text was considered concise or verbose.',
            ),
            SignalDescriptor(
                name='redundancy_count',
                extractor=lambda r: len(r.redundant_segments),
                description='Number of redundant segments identified.',
            ),
        ]

        # List out the specific "fluff" found, similar to False Positives in Correctness
        for i, seg in enumerate(result.redundant_segments):
            group = f'redundancy_{i}'
            signals.extend(
                [
                    SignalDescriptor(
                        name='segment',
                        group=group,
                        extractor=lambda r, idx=i: r.redundant_segments[idx].segment,
                        description='The text identified as redundant.',
                    ),
                    SignalDescriptor(
                        name='category',
                        group=group,
                        extractor=lambda r, idx=i: r.redundant_segments[idx].category,
                        description='Type of verbosity (Filler, Repetition, etc).',
                    ),
                    SignalDescriptor(
                        name='reason',
                        group=group,
                        extractor=lambda r, idx=i: r.redundant_segments[idx].reason,
                        description='Why this segment was flagged.',
                    ),
                ]
            )

        return signals
