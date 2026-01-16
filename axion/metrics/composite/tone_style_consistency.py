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


class ToneStyleDiff(RichBaseModel):
    """Details of a detected difference in tone or style."""

    dimension: str = Field(
        ...,
        description="The dimension of tone evaluated (e.g., 'Formal vs Casual', 'Empathy').",
    )
    expected_val: str = Field(
        ..., description='The tone/style observed in the expected output.'
    )
    actual_val: str = Field(
        ..., description='The tone/style observed in the actual output.'
    )
    verdict: str = Field(
        ...,
        description="Explanation of why this difference matters (or 'Match' if they align).",
    )
    model_config = {'extra': 'forbid'}


class ToneStyleResult(RichBaseModel):
    """Structured result for the Tone & Style metric."""

    final_score: float
    tone_match: bool
    style_match: bool
    differences: List[ToneStyleDiff]
    model_config = {'extra': 'forbid'}


class ToneJudgeInput(RichBaseModel):
    """Input for the internal tone judge."""

    actual_output: str = Field(..., description="The agent's response")
    expected_output: str = Field(..., description='The expected response')
    persona_description: Optional[str] = Field(None, description='Persona description')

    model_config = {'extra': 'forbid'}


class ToneJudgeOutput(RichBaseModel):
    """Output from the internal tone judge."""

    score: float = Field(
        ..., description='0.0 to 1.0 score of how well the tone matches.'
    )
    tone_match: bool = Field(
        ..., description='True if the emotional tone (e.g., excited, serious) matches.'
    )
    style_match: bool = Field(
        ...,
        description='True if the writing style (e.g., verbose, list-based) matches.',
    )
    differences: List[ToneStyleDiff] = Field(
        ..., description='List of specific tone/style differences found.'
    )
    reason: str = Field(..., description='Overall summary reason for the score.')

    model_config = {'extra': 'forbid'}


class ToneJudge(BaseMetric[ToneJudgeInput, ToneJudgeOutput]):
    """
    Internal LLM Judge that evaluates Tone and Style alignment.
    """

    instruction = """
    You are an expert Copy Editor and Brand Voice Manager.
    Compare the 'Actual Output' (Agent) against the 'Expected Output' (Gold Standard).

    Evaluate if the Agent matches the **Tone** (emotional attitude, e.g., empathetic, enthusiastic, dry)
    and **Style** (formatting, length, vocabulary) of the Expected Output.

    ### Scoring Rubric:
    - **1.0 (Perfect Match)**: The agent captures the exact emotion, enthusiasm, and formatting style of the expected output.
    - **0.8 (Minor Drift)**: Tone is generally correct but slightly less enthusiastic/formal, or formatting differs slightly (e.g., bullets vs dashes).
    - **0.5 (Significant Mismatch)**: Tone is neutral when it should be excited, or style is completely different (e.g., long paragraph vs list).
    - **0.0 (Complete Failure)**: Tone is robotic, rude, or the response ignores the persona entirely.

    Evaluate the current input based on this rubric.
    """

    input_model = ToneJudgeInput
    output_model = ToneJudgeOutput

    examples = [
        (
            ToneJudgeInput(
                expected_output="I'm super excited to help! Check out these cool features:",
                actual_output="I'd love to help you with that! Here are some amazing features to explore:",
                persona_description='Enthusiastic support',
            ),
            ToneJudgeOutput(
                score=1.0,
                tone_match=True,
                style_match=True,
                differences=[],
                reason='Both are highly enthusiastic and helpful.',
            ),
        ),
        (
            ToneJudgeInput(
                expected_output="I'm sorry to hear that. Let me make it right.",
                actual_output='Refund processed. Next question?',
                persona_description='Empathetic support',
            ),
            ToneJudgeOutput(
                score=0.0,
                tone_match=False,
                style_match=True,
                differences=[
                    ToneStyleDiff(
                        dimension='Empathy',
                        expected_val='Empathetic',
                        actual_val='Robotic/Rude',
                        verdict='Critical Tone Mismatch',
                    )
                ],
                reason='Expected is empathetic; Actual is robotic and rude.',
            ),
        ),
        (
            ToneJudgeInput(
                expected_output='Steps: 1. Open settings. 2. Click Save.',
                actual_output='You need to open settings and then click save.',
                persona_description='Structured instruction',
            ),
            ToneJudgeOutput(
                score=0.7,
                tone_match=True,
                style_match=False,
                differences=[
                    ToneStyleDiff(
                        dimension='Formatting',
                        expected_val='Numbered List',
                        actual_val='Paragraph',
                        verdict='Style Mismatch',
                    )
                ],
                reason='Content is correct, but failed to use the list format.',
            ),
        ),
    ]


@metric(
    name='Tone & Style Consistency',
    key='tone_style_consistency',
    description='Evaluates if the response matches the tone, persona, and formatting style of the expected answer.',
    required_fields=['actual_output', 'expected_output'],
    optional_fields=['persona_description'],
    default_threshold=0.8,
    tags=['knowledge', 'agent', 'single_turn'],
)
class ToneStyleConsistency(BaseMetric):
    """
    Measures how well the agent mimics the persona and style of the ground truth.
    Crucial for customer service agents where "Voice" is as important as "Fact".
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tone_judge = ToneJudge(**kwargs)

    @trace(name='ToneStyleConsistency', capture_args=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        self._validate_required_metric_fields(item)

        # Optional: Allow user to pass a specific persona description (e.g., "Helpful, excited, professional")
        # to guide the judge if expected_output implies it but doesn't explicitly state it.
        persona = getattr(item, 'persona_description', None)

        result = await self.tone_judge.execute(
            ToneJudgeInput(
                actual_output=item.actual_output,
                expected_output=item.expected_output,
                persona_description=persona,
            )
        )

        result_data = ToneStyleResult(
            final_score=result.score,
            tone_match=result.tone_match,
            style_match=result.style_match,
            differences=result.differences,
        )

        self.compute_cost_estimate([self.tone_judge])

        return MetricEvaluationResult(
            score=result.score,
            explanation=f'Tone Score: {result.score:.2f}. {result.reason}',
            signals=result_data,
        )

    def get_signals(self, result: ToneStyleResult) -> List[SignalDescriptor]:
        """Convert structured results into UI signals."""
        signals = [
            SignalDescriptor(
                name='tone_score',
                extractor=lambda r: r.final_score,
                description='Overall score for Tone & Style alignment.',
                headline_display=True,
            ),
            SignalDescriptor(
                name='tone_match',
                extractor=lambda r: r.tone_match,
                description='Did the emotional tone (e.g. empathy, excitement) match?',
            ),
            SignalDescriptor(
                name='style_match',
                extractor=lambda r: r.style_match,
                description='Did the formatting/writing style match?',
            ),
        ]

        for i, diff in enumerate(result.differences):
            group = f'diff_{i}'
            signals.extend(
                [
                    SignalDescriptor(
                        name='dimension',
                        group=group,
                        extractor=lambda r, idx=i: r.differences[idx].dimension,
                        description='The specific dimension being compared (e.g. Empathy).',
                    ),
                    SignalDescriptor(
                        name='comparison',
                        group=group,
                        extractor=lambda r,
                        idx=i: f"Expected: '{r.differences[idx].expected_val}' vs Actual: '{r.differences[idx].actual_val}'",
                        description='Comparison of the expected vs actual style.',
                    ),
                    SignalDescriptor(
                        name='verdict',
                        group=group,
                        extractor=lambda r, idx=i: r.differences[idx].verdict,
                        description='Judge verdict on this difference.',
                    ),
                ]
            )

        return signals
