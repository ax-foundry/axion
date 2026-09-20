"""A client for the Jev decision endpoint.

Jev has no chat-completions surface, so it is not reachable through the LLM
registry or LiteLLM as a provider. This is a direct JSON call against the
documented contract.

One call carries one `state` and every question asked about it. That is the
native shape of the API, not an optimization: Jev bills on input tokens and
ingests `state` once per call, so asking N questions in N calls re-sends the
same `state` N times and pays for it each time.
"""

import asyncio
from typing import Mapping, Optional

import httpx

from axion._core.environment import resolve_api_key, settings
from axion._core.logging import get_logger
from axion._handlers.jev.schema import JevResponse, Question
from axion.error import JevAuthError, JevError, JevRateLimitError

logger = get_logger(__name__)

# Retried because the endpoint or the hop in front of it is momentarily
# unavailable, not because the answer was wrong — Jev is deterministic enough
# that re-asking an answered question would just buy the same answer twice.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class JevClient:
    """Asks Jev a bundle of questions about one piece of state."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Args:
            api_key: TypeSafe key. Falls back to `settings.typesafe_api_key`.
            base_url: API root. Falls back to `settings.typesafe_base_url`.
            model: Jev model id. Falls back to `settings.typesafe_model`. The
                spelling is route-specific, which is why it is a setting and not
                a constant.
            timeout: Per-attempt timeout in seconds.
            max_retries: Attempts after the first for a retryable status.
            client: An `httpx.AsyncClient` to borrow. When passed, the caller
                owns it and `aclose` does not close it.
        """
        self._api_key = api_key
        self._base_url = (base_url or settings.typesafe_base_url).rstrip('/')
        self.model = model or settings.typesafe_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client
        self._owns_client = client is None

    @property
    def api_key(self) -> str:
        """Resolved lazily so constructing a client needs no credential."""
        return resolve_api_key(self._api_key, 'TYPESAFE_API_KEY', 'TypeSafe')

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying transport, unless it was lent to us."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> 'JevClient':
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def ask(
        self,
        state: str,
        questions: Mapping[str, Question],
        model: Optional[str] = None,
    ) -> JevResponse:
        """Ask every question about `state` in a single call.

        Args:
            state: The material being judged. Documented limit is 32k tokens,
                within a 64k budget for the whole request.
            questions: Question objects by name. The names come back as the
                keys of `answers`, so they are the join between what was asked
                and what was answered.
            model: Overrides the client's model for this call.

        Returns:
            The parsed response, with one typed answer per question asked.

        Raises:
            JevAuthError: The credential was rejected.
            JevRateLimitError: Rate-limited, and out of retries.
            JevError: Any other non-200, a transport failure, or a body that
                does not parse as answers.
        """
        if not questions:
            raise JevError('Jev was asked nothing; `questions` is empty.')

        body = {
            'state': state,
            'model': model or self.model,
            'questions': {
                name: question.model_dump() for name, question in questions.items()
            },
        }
        response = await self._post(body)

        try:
            parsed = JevResponse.model_validate(response.json())
        except Exception as error:
            raise JevError(
                f'Jev returned a body that does not parse as answers: {error}'
            ) from error

        missing = set(questions) - set(parsed.answers)
        if missing:
            # Every question is supposed to come back answered. A partial body
            # is worse than an error, because the caller would score whatever
            # did arrive and never learn the rest was dropped.
            raise JevError(
                f'Jev answered {len(parsed.answers)} of {len(questions)} questions; '
                f'missing: {sorted(missing)}'
            )
        return parsed

    async def _post(self, body: dict) -> httpx.Response:
        """POST the bundle, retrying the statuses worth retrying."""
        url = f'{self._base_url}/systemone'
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        client = self._get_client()
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(url, headers=headers, json=body)
            except httpx.HTTPError as error:
                last_error = JevError(f'Jev call failed in transport: {error}')
            else:
                if response.status_code == 200:
                    return response
                detail = response.text[:500]
                if response.status_code in (401, 403):
                    # Never retried: a rejected credential is rejected again.
                    raise JevAuthError(
                        f'Jev rejected the credential (HTTP {response.status_code}): {detail}'
                    )
                if response.status_code not in _RETRY_STATUS:
                    raise JevError(
                        f'Jev returned HTTP {response.status_code}: {detail}'
                    )
                last_error = (
                    JevRateLimitError(f'Jev rate-limited the call: {detail}')
                    if response.status_code == 429
                    else JevError(f'Jev returned HTTP {response.status_code}: {detail}')
                )

            if attempt < self._max_retries:
                delay = 2**attempt
                logger.warning(
                    'Jev call failed (attempt %d/%d), retrying in %ds: %s',
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    last_error,
                )
                await asyncio.sleep(delay)

        raise last_error
