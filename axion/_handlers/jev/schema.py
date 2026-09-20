"""Request and response types for the Jev decision endpoint.

Jev answers named questions about a shared `state` and returns a typed answer
per question rather than text. The three answer types are not interchangeable:
each carries a different notion of what the number means, so they are modelled
separately and discriminated on `type` rather than flattened into one score.
"""

from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class NoulQuestion(BaseModel):
    """A question answered with a probability that the statement holds."""

    type: Literal['noul'] = 'noul'
    instructions: str


class ChoiceQuestion(BaseModel):
    """A question answered with one label from a named set.

    `criteria` maps each label to the description Jev judges against, so the
    labels are the vocabulary of the answer as well as of the request.
    """

    type: Literal['choice'] = 'choice'
    instructions: str
    criteria: Dict[str, str]


class ScoreQuestion(BaseModel):
    """A question answered with a position on an ordered rubric.

    `criteria` is ordered: index 0 is level 0. The order carries the meaning,
    and nothing requires higher to be better — a severity rubric runs the other
    way. Callers decide the direction; this type does not.
    """

    type: Literal['score'] = 'score'
    instructions: str
    criteria: List[str]


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


class NoulAnswer(BaseModel):
    """A probability in [0, 1].

    It carries no `confidence`, because the probability already is one — a
    `noul` of 0.5 is the uncertain answer, not a certain answer about a coin.
    """

    type: Literal['noul']
    noul: float


class ChoiceAnswer(BaseModel):
    """The selected label, with the distribution it was selected from."""

    type: Literal['choice']
    choice: str
    confidence: float
    probabilities: Dict[str, float] = Field(default_factory=dict)


class ScoreAnswer(BaseModel):
    """A probability-weighted position on the rubric, so a fractional level.

    The value is an expectation over `probabilities`, not a level Jev picked:
    2.99 means almost all the mass sits on level 3. Rounding it to an integer
    discards the part that distinguishes a confident 3 from a split 2/3.
    """

    type: Literal['score']
    score: float
    confidence: float
    legend: Dict[str, str] = Field(default_factory=dict)
    probabilities: Dict[str, float] = Field(default_factory=dict)

    @property
    def levels(self) -> int:
        """How many rungs the rubric has."""
        return len(self.legend) or len(self.probabilities)

    @property
    def normalized(self) -> Optional[float]:
        """The level rescaled to [0, 1], or None if the rubric has one rung.

        Higher stays higher. A rubric whose top level is the bad outcome
        normalizes to a number where 1.0 is that bad outcome; inverting it is
        the caller's decision and belongs where the rubric was written.
        """
        if self.levels < 2:
            return None
        return self.score / (self.levels - 1)


Answer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator='type')
]


class JevUsage(BaseModel):
    """Token counts for one call. Jev bills on input only."""

    input_tokens: int = 0
    output_tokens: int = 0


class JevResponse(BaseModel):
    """One bundled call: every question answered against a single `state`."""

    model: str
    answers: Dict[str, Answer] = Field(default_factory=dict)
    usage: JevUsage = Field(default_factory=JevUsage)
