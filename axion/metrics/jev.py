"""A metric base class backed by the Jev decision endpoint.

A subclass declares the questions it wants answered about an item and how each
answer becomes a number. Every question travels in one call, because Jev bills
on input tokens and ingests the shared state once per call — asking N questions
in N calls re-sends the same state N times.

The bundle explodes into one sub-metric per question, each judged against its
own threshold. The questions are not commensurable — a grounding probability
and a severity rubric do not average into anything meaningful — so the parent
score is the worst part rather than the mean.
"""

from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from axion._core.logging import get_logger
from axion._handlers.jev.client import JevClient
from axion._handlers.jev.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    Question,
    ScoreAnswer,
)
from axion.dataset import DatasetItem
from axion.error import MetricValidationError
from axion.metrics.base import BaseMetric
from axion.metrics.schema import MetricEvaluationResult, SubMetricResult

logger = get_logger(__name__)

# Where `execute` leaves what `get_sub_metrics` needs. The two are called
# separately by the runner, with only the result passed between them.
_METADATA_KEY = 'jev'


class JevRubric(BaseModel):
    """One question, and how to read its answer as a score.

    The reading is declared here rather than inferred, because none of the
    three answer types has an obvious scalar. A `noul` is already a probability
    but may be asked in either direction; a `choice` is a label with no
    ordering at all; a `score` is a position on a rubric whose top end is
    sometimes the good outcome and sometimes the bad one.
    """

    question: Question
    threshold: Optional[float] = Field(
        default=None,
        description='Threshold for this question alone. Inherits the parent metric when unset.',
    )
    invert: bool = Field(
        default=False,
        description='Score as 1 - x. For a question whose high end is the bad outcome.',
    )
    label_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description='Score per label. Required for a choice question, unused otherwise.',
    )

    @model_validator(mode='after')
    def _check_labels(self) -> 'JevRubric':
        """A choice question needs every one of its labels scored.

        An unscored label is not a neutral default: it is a verdict the metric
        cannot express, which would surface as a missing score on whichever
        item happened to elicit it.
        """
        if not isinstance(self.question, ChoiceQuestion):
            return self
        if not self.label_scores:
            raise ValueError('a choice question needs `label_scores`')
        unscored = set(self.question.criteria) - set(self.label_scores)
        if unscored:
            raise ValueError(f'choice labels have no score: {sorted(unscored)}')
        return self

    def score_for(self, answer: Answer) -> Optional[float]:
        """Turn one answer into a score in [0, 1], or None if it has none."""
        value: Optional[float]
        if isinstance(answer, NoulAnswer):
            value = answer.noul
        elif isinstance(answer, ChoiceAnswer):
            value = (self.label_scores or {}).get(answer.choice)
            if value is None:
                # Jev returned a label outside the criteria it was given.
                logger.warning('Jev returned an unscored label: %s', answer.choice)
        elif isinstance(answer, ScoreAnswer):
            value = answer.normalized
        else:
            value = None

        if value is None:
            return None
        return 1.0 - value if self.invert else value

    def explanation_for(self, answer: Answer) -> str:
        """State what Jev said, in the vocabulary it said it in."""
        if isinstance(answer, NoulAnswer):
            return f'probability {answer.noul:.2f}'
        if isinstance(answer, ChoiceAnswer):
            return f'{answer.choice} (confidence {answer.confidence:.2f})'
        if isinstance(answer, ScoreAnswer):
            level = answer.legend.get(str(round(answer.score)), '')
            detail = f' — nearest level: {level}' if level else ''
            return (
                f'level {answer.score:.2f} of {max(answer.levels - 1, 0)} '
                f'(confidence {answer.confidence:.2f}){detail}'
            )
        return ''


class JevMetric(BaseMetric):
    """Scores an item by asking Jev a bundle of questions about it.

    Subclasses declare `questions` and either `state_template` or an override
    of `build_state`. Everything else — the call, the explosion into
    sub-metrics, the thresholds — is handled here.

    Example:
        @metric(
            key='grounding',
            name='Grounding',
            description='Is the answer supported by what the tools returned?',
            required_fields=['actual_output', 'retrieval_context'],
            default_threshold=0.7,
        )
        class Grounding(JevMetric):
            state_template = (
                'Answer: {actual_output}\\n\\nTool output: {retrieval_context}'
            )
            questions = {
                'supported': JevRubric(
                    question=NoulQuestion(instructions='Is every claim supported?'),
                    threshold=0.8,
                ),
            }

    The declared thresholds are the metric's defaults, not its settings. Each
    instance may be handed `question_thresholds` to move any of them, so one
    metric can be held to a different bar for each agent it grades without a
    subclass per agent.
    """

    # Jev is not a chat model and is not reached through the LLM registry, so
    # this metric neither resolves a model nor needs credentials for one.
    requires_llm: ClassVar[bool] = False

    is_multi_metric = True
    include_parent_score = True

    # Declared by the subclass.
    questions: ClassVar[Dict[str, JevRubric]] = {}
    state_template: ClassVar[Optional[str]] = None

    def __init__(
        self,
        *args,
        jev_client: Optional[JevClient] = None,
        question_thresholds: Optional[Dict[str, Optional[float]]] = None,
        state_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            jev_client: A client to reuse. When omitted, each call makes and
                closes its own, which is the safe default where metric
                instances are built per request and never disposed of.
            question_thresholds: Per-question thresholds for this instance,
                by question name. One agent's idea of a passing grounding
                probability is not another's, so the bar is configuration
                rather than a subclass.
            state_template: Overrides the class template for this instance,
                for an agent whose traces render differently.
        """
        super().__init__(*args, **kwargs)
        self._jev_client = jev_client
        if not self.questions:
            raise MetricValidationError(
                f'{type(self).__name__} asks Jev nothing; declare `questions`.'
            )
        if state_template is not None:
            self.state_template = state_template
        self.questions = self._calibrated(question_thresholds)

    def _calibrated(
        self, thresholds: Optional[Dict[str, Optional[float]]]
    ) -> Dict[str, JevRubric]:
        """Copy the declared questions with this instance's thresholds applied.

        The copy is the point: the declared questions live on the class, and
        writing a threshold into them would recalibrate every other instance
        of the metric in the process.
        """
        if not thresholds:
            return dict(self.questions)

        unknown = set(thresholds) - set(self.questions)
        if unknown:
            # Silently ignoring these would leave a miscalibrated metric
            # looking correctly configured, which is the failure that takes
            # longest to notice.
            raise MetricValidationError(
                f'{type(self).__name__} has no question named '
                f'{sorted(unknown)}; it asks {sorted(self.questions)}.'
            )
        return {
            name: (
                rubric.model_copy(update={'threshold': thresholds[name]})
                if name in thresholds
                else rubric
            )
            for name, rubric in self.questions.items()
        }

    def build_state(self, item: DatasetItem) -> str:
        """Render the material Jev judges.

        The default fills `state_template` from the metric's required and
        optional fields, resolved through `field_mapping` like any other field
        access. Override for anything a format string cannot express.
        """
        if not self.state_template:
            raise MetricValidationError(
                f'{type(self).__name__} needs `state_template` or a `build_state` override.'
            )
        fields = {
            name: self.get_field(item, name, default='')
            for name in list(self.required_fields) + list(self.optional_fields or [])
        }
        return self.state_template.format(**fields)

    def aggregate(self, scores: List[float]) -> Optional[float]:
        """Reduce the answered questions to the metric's own score.

        The worst part, not the mean: the questions measure different things on
        different scales, and a bundle that fails one of them has failed.
        """
        return min(scores) if scores else None

    async def execute(self, item: DatasetItem, **kwargs: Any) -> MetricEvaluationResult:
        """Ask every question in one call and score the answers."""
        state = self.build_state(item)
        questions = {name: rubric.question for name, rubric in self.questions.items()}

        if self._jev_client is not None:
            response = await self._jev_client.ask(state, questions)
        else:
            async with JevClient() as client:
                response = await client.ask(state, questions)

        parts = []
        for name, rubric in self.questions.items():
            answer = response.answers[name]
            parts.append(
                {
                    'name': name,
                    'score': rubric.score_for(answer),
                    'threshold': rubric.threshold,
                    'explanation': rubric.explanation_for(answer),
                }
            )

        scored = [part['score'] for part in parts if part['score'] is not None]
        return MetricEvaluationResult(
            score=self.aggregate(scored),
            explanation=(
                f'{len(scored)} of {len(parts)} questions scored; '
                'the metric reports its lowest.'
            ),
            metadata={
                _METADATA_KEY: {
                    'parts': parts,
                    # The version that answered, which is not the alias asked
                    # for: `jev-latest` resolves server-side.
                    'model': response.model,
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                }
            },
        )

    def get_sub_metrics(self, result: MetricEvaluationResult) -> List[SubMetricResult]:
        """One row per question, each carrying the threshold it is judged against."""
        payload = (result.metadata or {}).get(_METADATA_KEY) or {}
        return [
            SubMetricResult(
                name=part['name'],
                score=part['score'],
                threshold=part['threshold'],
                explanation=part['explanation'],
            )
            for part in payload.get('parts', [])
        ]
