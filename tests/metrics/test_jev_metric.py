import json

import httpx
import pytest

from axion._handlers.jev.client import JevClient
from axion._handlers.jev.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from axion.dataset import DatasetItem
from axion.error import MetricValidationError
from axion.metrics.base import metric
from axion.metrics.jev import JevMetric, JevRubric
from axion.runners.metric import MetricRunnerFactory

RUBRICS = {
    'supported': JevRubric(
        question=NoulQuestion(instructions='Is every claim supported?'),
        threshold=0.8,
    ),
    'failure': JevRubric(
        question=ChoiceQuestion(
            instructions='Which failure?',
            criteria={'none': 'Fine.', 'fabricated_source': 'Invented a source.'},
        ),
        label_scores={'none': 1.0, 'fabricated_source': 0.0},
    ),
    'severity': JevRubric(
        question=ScoreQuestion(
            instructions='How bad?', criteria=['Fine.', 'Bad.', 'Worse.', 'Worst.']
        ),
        # The rubric's top level is the bad outcome, so the score runs backwards.
        invert=True,
        threshold=0.5,
    ),
}

ANSWERS = {
    'supported': {'type': 'noul', 'noul': 0.02},
    'failure': {
        'type': 'choice',
        'choice': 'fabricated_source',
        'confidence': 0.94,
        'probabilities': {'none': 0.05, 'fabricated_source': 0.95},
    },
    'severity': {
        'type': 'score',
        'score': 2.99,
        'confidence': 0.99,
        'legend': {'0': 'Fine.', '1': 'Bad.', '2': 'Worse.', '3': 'Worst.'},
        'probabilities': {'0': 0.0, '1': 0.0, '2': 0.01, '3': 0.99},
    },
}


@metric(
    key='test_jev_grounding',
    name='Test Jev Grounding',
    description='A Jev-backed metric for testing.',
    required_fields=['actual_output'],
    optional_fields=['retrieved_content'],
    default_threshold=0.7,
)
class GroundingForTest(JevMetric):
    state_template = 'Answer: {actual_output}\n\nTools: {retrieved_content}'
    questions = RUBRICS


def fake_client(answers=None, seen=None):
    """A client whose transport answers without leaving the process."""
    payload = ANSWERS if answers is None else answers

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(
            200,
            json={
                'model': 'jev-1.13.0',
                'answers': payload,
                'usage': {'input_tokens': 574, 'output_tokens': 86},
            },
        )

    return JevClient(
        api_key='test-key',
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def item():
    return DatasetItem(
        actual_output='I confirmed this against the filed rate pages.',
        retrieved_content=['No rate page was retrieved.'],
    )


class TestConstruction:
    def test_needs_no_llm_credentials(self, monkeypatch):
        """The whole point of the capability: no model is resolved."""
        from axion.metrics import base

        def fail(*args, **kwargs):
            raise AssertionError('the model registry was consulted')

        monkeypatch.setattr(base, 'LLMRegistry', fail)
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)

        instance = GroundingForTest()

        assert instance.requires_llm is False
        assert instance.model_name is None

    def test_a_metric_that_asks_nothing_is_refused(self):
        @metric(
            key='test_jev_empty',
            name='Empty',
            description='Asks nothing.',
            required_fields=['actual_output'],
        )
        class Empty(JevMetric):
            state_template = '{actual_output}'

        with pytest.raises(MetricValidationError, match='asks Jev nothing'):
            Empty()

    @pytest.mark.asyncio
    async def test_a_metric_with_no_state_is_refused(self):
        @metric(
            key='test_jev_stateless',
            name='Stateless',
            description='Renders nothing.',
            required_fields=['actual_output'],
        )
        class Stateless(JevMetric):
            questions = RUBRICS

        with pytest.raises(MetricValidationError, match='state_template'):
            await Stateless(jev_client=fake_client()).execute(item())


class TestRubricScoring:
    def test_a_noul_is_its_own_score(self):
        rubric = RUBRICS['supported']
        answer = _answer('supported')

        assert rubric.score_for(answer) == pytest.approx(0.02)

    def test_a_choice_is_scored_by_its_label(self):
        assert RUBRICS['failure'].score_for(_answer('failure')) == 0.0

    def test_a_score_is_normalized_and_inverted_when_asked(self):
        """2.99 of 3 on a severity rubric is a bad item, so the metric reads 0.01."""
        assert RUBRICS['severity'].score_for(_answer('severity')) == pytest.approx(
            1 - 2.99 / 3
        )

    def test_a_choice_question_must_score_every_label(self):
        with pytest.raises(ValueError, match='no score'):
            JevRubric(
                question=ChoiceQuestion(
                    instructions='Which?', criteria={'a': 'A.', 'b': 'B.'}
                ),
                label_scores={'a': 1.0},
            )

    def test_a_choice_question_must_have_label_scores(self):
        with pytest.raises(ValueError, match='label_scores'):
            JevRubric(
                question=ChoiceQuestion(instructions='Which?', criteria={'a': 'A.'})
            )

    def test_an_unscored_label_scores_nothing_rather_than_zero(self):
        """A label outside the criteria is an unknown, not a failing verdict."""
        rubric = RUBRICS['failure']
        answer = _answer('failure').model_copy(update={'choice': 'something_else'})

        assert rubric.score_for(answer) is None


class TestExecute:
    @pytest.mark.asyncio
    async def test_asks_every_question_in_one_call(self):
        seen = []

        await GroundingForTest(jev_client=fake_client(seen=seen)).execute(item())

        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_the_state_is_rendered_from_the_items_fields(self):
        seen = []

        await GroundingForTest(jev_client=fake_client(seen=seen)).execute(item())

        sent = json.loads(seen[0].content)['state']
        assert 'filed rate pages' in sent
        assert 'No rate page was retrieved.' in sent

    @pytest.mark.asyncio
    async def test_the_metric_score_is_the_worst_question(self):
        """The questions are not commensurable, so the bundle is its weakest part."""
        result = await GroundingForTest(jev_client=fake_client()).execute(item())

        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_records_the_version_that_answered(self):
        result = await GroundingForTest(jev_client=fake_client()).execute(item())

        assert result.metadata['jev']['model'] == 'jev-1.13.0'
        assert result.metadata['jev']['input_tokens'] == 574


class TestSubMetrics:
    @pytest.mark.asyncio
    async def test_one_row_per_question_with_its_own_threshold(self):
        instance = GroundingForTest(jev_client=fake_client())
        result = await instance.execute(item())

        rows = {row.name: row for row in instance.get_sub_metrics(result)}

        assert set(rows) == {'supported', 'failure', 'severity'}
        assert rows['supported'].threshold == 0.8
        assert rows['severity'].threshold == 0.5
        # Declared no threshold, so it inherits the parent's at explosion time.
        assert rows['failure'].threshold is None

    @pytest.mark.asyncio
    async def test_the_runner_explodes_the_bundle_into_judged_rows(self):
        """End to end through the runner: this is the shape echo-evals reads."""
        instance = GroundingForTest(jev_client=fake_client())
        executor = MetricRunnerFactory().create_executor(instance, threshold=0.7)

        scores = await executor.execute(item())

        rows = {score.name: score for score in scores}
        assert len(rows) == 4  # the parent plus one row per question

        supported = rows['Test Jev Grounding_supported']
        assert supported.score == pytest.approx(0.02)
        # 0.02 against its own 0.8, not the parent's 0.7.
        assert supported.passed is False

        severity = rows['Test Jev Grounding_severity']
        assert severity.score == pytest.approx(1 - 2.99 / 3)
        assert severity.passed is False

    @pytest.mark.asyncio
    async def test_a_passing_bundle_passes_every_row(self):
        clean = {
            'supported': {'type': 'noul', 'noul': 0.97},
            'failure': {
                'type': 'choice',
                'choice': 'none',
                'confidence': 0.99,
                'probabilities': {'none': 0.99, 'fabricated_source': 0.01},
            },
            'severity': {
                'type': 'score',
                'score': 0.02,
                'confidence': 0.99,
                'legend': {'0': 'Fine.', '1': 'Bad.', '2': 'Worse.', '3': 'Worst.'},
                'probabilities': {'0': 0.98, '3': 0.0},
            },
        }
        instance = GroundingForTest(jev_client=fake_client(answers=clean))
        executor = MetricRunnerFactory().create_executor(instance, threshold=0.7)

        scores = await executor.execute(item())

        assert all(
            score.passed for score in scores if score.name != 'Test Jev Grounding'
        )


def _answer(name):
    from axion._handlers.jev.schema import JevResponse

    return JevResponse.model_validate(
        {'model': 'jev-1.13.0', 'answers': {name: ANSWERS[name]}}
    ).answers[name]


class TestCalibration:
    def test_a_threshold_can_be_moved_per_instance(self):
        instance = GroundingForTest(question_thresholds={'supported': 0.4})

        assert instance.questions['supported'].threshold == 0.4

    def test_moving_one_threshold_leaves_the_others_alone(self):
        instance = GroundingForTest(question_thresholds={'supported': 0.4})

        assert instance.questions['severity'].threshold == 0.5
        assert instance.questions['failure'].threshold is None

    def test_a_threshold_can_be_removed(self):
        """None means 'inherit the parent's', which is a real calibration."""
        instance = GroundingForTest(question_thresholds={'severity': None})

        assert instance.questions['severity'].threshold is None

    def test_the_declared_questions_are_not_recalibrated(self):
        """Writing through to the class would move every other instance's bar."""
        calibrated = GroundingForTest(question_thresholds={'supported': 0.4})
        default = GroundingForTest()

        assert calibrated.questions['supported'].threshold == 0.4
        assert default.questions['supported'].threshold == 0.8
        assert GroundingForTest.questions['supported'].threshold == 0.8
        assert RUBRICS['supported'].threshold == 0.8

    def test_an_unknown_question_is_refused(self):
        with pytest.raises(MetricValidationError) as excinfo:
            GroundingForTest(question_thresholds={'supprted': 0.4})

        assert 'supprted' in str(excinfo.value)

    def test_the_state_template_can_be_replaced_per_instance(self):
        instance = GroundingForTest(state_template='Just: {actual_output}')

        assert instance.build_state(item()) == (
            'Just: I confirmed this against the filed rate pages.'
        )
        assert GroundingForTest.state_template.startswith('Answer:')

    @pytest.mark.asyncio
    async def test_a_calibrated_threshold_decides_the_sub_metric(self):
        """The whole point: the same answer passes for one agent, fails another."""
        strict = GroundingForTest(jev_client=fake_client())
        lenient = GroundingForTest(
            jev_client=fake_client(), question_thresholds={'supported': 0.01}
        )

        def supported(scores):
            return next(score for score in scores if score.name.endswith('_supported'))

        strict_scores = (
            await MetricRunnerFactory()
            .create_executor(strict, threshold=0.7)
            .execute(item())
        )
        lenient_scores = (
            await MetricRunnerFactory()
            .create_executor(lenient, threshold=0.7)
            .execute(item())
        )

        # Jev answered 0.02, which clears 0.01 and not the declared 0.8.
        assert supported(strict_scores).passed is False
        assert supported(lenient_scores).passed is True
