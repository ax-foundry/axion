import json

import httpx
import pytest

from axion._handlers.jev.client import JevClient
from axion._handlers.jev.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from axion.error import JevAuthError, JevError, JevRateLimitError
from tests._handlers.jev.test_schema import LIVE_RESPONSE

QUESTIONS = {
    'faithful': NoulQuestion(instructions='Is the answer supported?'),
    'category': ChoiceQuestion(
        instructions='Which failure?',
        criteria={'none': 'Fine.', 'fabricated_source': 'Invented a source.'},
    ),
    'severity': ScoreQuestion(
        instructions='How bad?', criteria=['Fine.', 'Bad.', 'Worse.', 'Worst.']
    ),
}


def client_for(handler, **kwargs):
    """A client wired to a fake transport, so nothing leaves the process."""
    transport = httpx.MockTransport(handler)
    return JevClient(
        api_key='test-key',
        client=httpx.AsyncClient(transport=transport),
        **kwargs,
    )


def answering(*names, status=200, body=None):
    """A handler that answers exactly `names`, recording what it was sent."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if body is not None:
            return httpx.Response(status, json=body)
        answers = {
            name: LIVE_RESPONSE['answers'][name]
            for name in names
            if name in LIVE_RESPONSE['answers']
        }
        return httpx.Response(status, json={**LIVE_RESPONSE, 'answers': answers})

    return handler, seen


class TestRequest:
    @pytest.mark.asyncio
    async def test_sends_every_question_in_one_call(self):
        """The bundle is the point: one `state` ingested once, not once per question."""
        handler, seen = answering('faithful', 'category', 'severity')

        await client_for(handler).ask('the state', QUESTIONS)

        assert len(seen) == 1
        sent = json.loads(seen[0].content)
        assert sent['state'] == 'the state'
        assert set(sent['questions']) == {'faithful', 'category', 'severity'}

    @pytest.mark.asyncio
    async def test_carries_the_bearer_token(self):
        handler, seen = answering('faithful')

        await client_for(handler).ask('s', {'faithful': QUESTIONS['faithful']})

        assert seen[0].headers['authorization'] == 'Bearer test-key'

    @pytest.mark.asyncio
    async def test_posts_to_systemone_under_the_configured_base(self):
        handler, seen = answering('faithful')

        await client_for(handler, base_url='https://gateway.example/typesafe/v1').ask(
            's', {'faithful': QUESTIONS['faithful']}
        )

        assert str(seen[0].url) == 'https://gateway.example/typesafe/v1/systemone'

    @pytest.mark.asyncio
    async def test_the_model_id_is_configurable_not_constant(self):
        """The spelling is route-specific; a gateway renames the same model."""
        handler, seen = answering('faithful')

        await client_for(handler, model='typesafe-ai/jev').ask(
            's', {'faithful': QUESTIONS['faithful']}
        )

        assert json.loads(seen[0].content)['model'] == 'typesafe-ai/jev'

    @pytest.mark.asyncio
    async def test_refuses_to_ask_nothing(self):
        handler, seen = answering()

        with pytest.raises(JevError):
            await client_for(handler).ask('s', {})

        assert seen == []


class TestResponse:
    @pytest.mark.asyncio
    async def test_returns_a_typed_answer_per_question(self):
        handler, _ = answering('faithful', 'category', 'severity')

        result = await client_for(handler).ask('s', QUESTIONS)

        assert result.answers['faithful'].noul == 0.02
        assert result.answers['category'].choice == 'fabricated_source'
        assert result.answers['severity'].normalized == pytest.approx(2.99 / 3)

    @pytest.mark.asyncio
    async def test_a_partly_answered_bundle_is_an_error(self):
        """Scoring what arrived would silently drop the questions that did not."""
        handler, _ = answering('faithful')

        with pytest.raises(JevError, match='missing'):
            await client_for(handler).ask('s', QUESTIONS)

    @pytest.mark.asyncio
    async def test_an_unparseable_body_is_an_error(self):
        def handler(request):
            return httpx.Response(200, text='not json')

        with pytest.raises(JevError, match='does not parse'):
            await client_for(handler).ask('s', {'faithful': QUESTIONS['faithful']})


class TestFailures:
    @pytest.mark.asyncio
    async def test_a_rejected_credential_is_not_retried(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(401, text='bad key')

        with pytest.raises(JevAuthError):
            await client_for(handler).ask('s', {'faithful': QUESTIONS['faithful']})

        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_a_client_error_is_not_retried(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(400, text='malformed')

        with pytest.raises(JevError, match='400'):
            await client_for(handler).ask('s', {'faithful': QUESTIONS['faithful']})

        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_rate_limiting_is_retried_then_raised(self, monkeypatch):
        monkeypatch.setattr('axion._handlers.jev.client.asyncio.sleep', _no_sleep)
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(429, text='slow down')

        with pytest.raises(JevRateLimitError):
            await client_for(handler, max_retries=2).ask(
                's', {'faithful': QUESTIONS['faithful']}
            )

        assert len(seen) == 3

    @pytest.mark.asyncio
    async def test_a_retry_that_succeeds_returns_the_answer(self, monkeypatch):
        monkeypatch.setattr('axion._handlers.jev.client.asyncio.sleep', _no_sleep)
        calls = {'n': 0}

        def handler(request):
            calls['n'] += 1
            if calls['n'] == 1:
                return httpx.Response(503, text='unavailable')
            return httpx.Response(
                200,
                json={
                    **LIVE_RESPONSE,
                    'answers': {'faithful': LIVE_RESPONSE['answers']['faithful']},
                },
            )

        result = await client_for(handler).ask('s', {'faithful': QUESTIONS['faithful']})

        assert calls['n'] == 2
        assert result.answers['faithful'].noul == 0.02

    @pytest.mark.asyncio
    async def test_a_transport_failure_raises_rather_than_returning_nothing(
        self, monkeypatch
    ):
        monkeypatch.setattr('axion._handlers.jev.client.asyncio.sleep', _no_sleep)

        def handler(request):
            raise httpx.ConnectError('no route')

        with pytest.raises(JevError, match='transport'):
            await client_for(handler, max_retries=1).ask(
                's', {'faithful': QUESTIONS['faithful']}
            )


class TestCredentialResolution:
    def test_constructing_a_client_needs_no_credential(self):
        """Resolution is lazy, so a metric can be built where no key is set."""
        JevClient()

    def test_asking_without_one_says_which_variable_is_missing(self, monkeypatch):
        from axion._handlers.jev import client as client_module

        monkeypatch.setattr(client_module.settings, 'typesafe_api_key', None, False)

        with pytest.raises(ValueError, match='TYPESAFE_API_KEY'):
            _ = JevClient().api_key


async def _no_sleep(_seconds):
    return None
