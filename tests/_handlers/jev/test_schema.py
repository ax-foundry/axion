import pytest

from axion._handlers.jev.schema import (
    ChoiceAnswer,
    JevResponse,
    NoulAnswer,
    ScoreAnswer,
)

# The body a live call returns, kept verbatim so a contract change breaks a test
# rather than a production metric.
LIVE_RESPONSE: dict = {
    'model': 'jev-1.13.0',
    'answers': {
        'faithful': {'type': 'noul', 'noul': 0.02},
        'category': {
            'type': 'choice',
            'choice': 'fabricated_source',
            'confidence': 0.93,
            'probabilities': {
                'none': 0.0,
                'arithmetic': 0.0,
                'unsupported_claim': 0.05,
                'fabricated_source': 0.95,
            },
        },
        'severity': {
            'type': 'score',
            'score': 2.99,
            'confidence': 0.99,
            'legend': {
                '0': 'No issue.',
                '1': 'Minor.',
                '2': 'Moderate.',
                '3': 'Severe.',
            },
            'probabilities': {'0': 0.0, '1': 0.0, '2': 0.01, '3': 0.99},
        },
    },
    'usage': {'input_tokens': 574, 'output_tokens': 86},
}


class TestParsing:
    def test_each_answer_parses_as_its_own_type(self):
        parsed = JevResponse.model_validate(LIVE_RESPONSE)

        assert isinstance(parsed.answers['faithful'], NoulAnswer)
        assert isinstance(parsed.answers['category'], ChoiceAnswer)
        assert isinstance(parsed.answers['severity'], ScoreAnswer)

    def test_carries_the_model_that_answered_not_the_alias_asked_for(self):
        """`jev-latest` is a request-side alias; the reply names the real version."""
        assert JevResponse.model_validate(LIVE_RESPONSE).model == 'jev-1.13.0'

    def test_usage_is_read(self):
        assert JevResponse.model_validate(LIVE_RESPONSE).usage.input_tokens == 574

    def test_an_unknown_answer_type_is_rejected(self):
        body = {'model': 'jev-1.13.0', 'answers': {'q': {'type': 'vibes', 'x': 1}}}

        with pytest.raises(Exception):
            JevResponse.model_validate(body)


class TestScoreNormalization:
    def test_rescales_the_level_to_the_unit_interval(self):
        answer = ScoreAnswer.model_validate(LIVE_RESPONSE['answers']['severity'])

        assert answer.levels == 4
        assert answer.normalized == pytest.approx(2.99 / 3)

    def test_keeps_the_fraction(self):
        """2.99 is an expectation over the rubric, not a level Jev picked."""
        answer = ScoreAnswer.model_validate(LIVE_RESPONSE['answers']['severity'])

        assert answer.score != int(answer.score)

    def test_a_one_rung_rubric_normalizes_to_nothing(self):
        answer = ScoreAnswer(
            type='score', score=0.0, confidence=1.0, legend={'0': 'only'}
        )

        assert answer.normalized is None

    def test_falls_back_to_the_distribution_when_no_legend_is_returned(self):
        """Gateways strip `legend`; the rubric size survives in probabilities."""
        answer = ScoreAnswer(
            type='score',
            score=1.5,
            confidence=0.8,
            probabilities={'0': 0.0, '1': 0.5, '2': 0.5},
        )

        assert answer.levels == 3
        assert answer.normalized == pytest.approx(0.75)


class TestNoul:
    def test_carries_no_confidence(self):
        """The probability is the answer, so a separate confidence is meaningless."""
        assert not hasattr(NoulAnswer(type='noul', noul=0.02), 'confidence')
