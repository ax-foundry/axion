"""A fake Langfuse ``client.api`` serving observation and score pages, and row builders."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip('langfuse.api')

from langfuse.api import (  # noqa: E402
    GetScoresV3Response,
    ObservationsV2Response,
    ObservationV2,
)
from langfuse.api.core.pydantic_utilities import parse_obj_as  # noqa: E402

from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader  # noqa: E402
from axion._core.tracing.loaders.langfuse_rows import (  # noqa: E402
    METADATA_CUT_CHARS,
)

PROJECT_ID = 'proj-test'


def _time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def row(
    trace_id: str,
    obs_id: str,
    start: str,
    *,
    parent: str | None = None,
    root: bool | None = None,
    name: str | None = 'turn',
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: Any = None,
    input: Any = None,
    environment: str = 'production',
) -> ObservationV2:
    """A hand-made observation row."""
    return parse_obj_as(
        ObservationV2,
        {
            'id': obs_id,
            'traceId': trace_id,
            'startTime': start,
            'projectId': PROJECT_ID,
            'parentObservationId': parent,
            'type': 'SPAN',
            'isRootObservation': (parent is None) if root is None else root,
            'name': obs_id,
            'environment': environment,
            'sessionId': session_id,
            'traceName': name,
            'tags': tags or [],
            'metadata': metadata,
            'input': input,
        },
    )


def score(
    score_id: str,
    name: str,
    data_type: str,
    value: Any,
    timestamp: str,
    *,
    trace_id: str | None = None,
    observation_id: str | None = None,
    session_id: str | None = None,
    comment: str | None = None,
) -> Any:
    """A scores-read row; ``session_id`` is what the fake's session filter matches."""
    if observation_id:
        subject = {'kind': 'observation', 'id': observation_id, 'traceId': trace_id}
    elif trace_id:
        subject = {'kind': 'trace', 'id': trace_id}
    else:
        subject = {'kind': 'session', 'id': session_id}
    parsed = parse_obj_as(
        GetScoresV3Response,
        {
            'data': [
                {
                    'id': score_id,
                    'projectId': PROJECT_ID,
                    'name': name,
                    'source': 'API',
                    'timestamp': timestamp,
                    'environment': 'production',
                    'createdAt': timestamp,
                    'updatedAt': timestamp,
                    'comment': comment,
                    'subject': subject,
                    'dataType': data_type,
                    'value': value,
                }
            ],
            'meta': {'limit': 100},
        },
    ).data[0]
    return SimpleNamespace(raw=parsed, session_id=session_id)


def _cut(metadata: Any, expand: set[str]) -> Any:
    if not isinstance(metadata, dict):
        return metadata
    out = {}
    for key, value in metadata.items():
        text = value if isinstance(value, str) else json.dumps(value)
        cut = len(text) > METADATA_CUT_CHARS and key not in expand
        out[key] = text[:METADATA_CUT_CHARS] if cut else value
    return out


def _matches(r: ObservationV2, condition: dict[str, Any]) -> bool:
    column, op, value = condition['column'], condition['operator'], condition['value']
    if column == 'startTime':
        start, bound = _time(r.start_time), _time(value)
        return {'>=': start >= bound, '<': start < bound, '<=': start <= bound}[op]
    if column == 'traceName':
        return r.trace_name == value
    if column == 'tags':
        return set(value) <= set(r.tags or [])
    if column == 'environment':
        return r.environment in value
    if column == 'traceId':
        return r.trace_id in value
    raise AssertionError(f'fake API has no filter for {column}')


class FakeObservations:
    """
    Serves rows a page at a time.

    With ``empty_pages`` every continuation is first an empty page that still carries
    a cursor, which the API is allowed to return.
    """

    def __init__(
        self, rows: list[ObservationV2], page_size: int, empty_pages: bool = False
    ) -> None:
        self.rows = rows
        self.page_size = page_size
        self.empty_pages = empty_pages
        self.calls: list[dict[str, Any]] = []

    def get_many(self, **kw: Any) -> ObservationsV2Response:
        self.calls.append(kw)
        rows = self.rows
        if kw.get('trace_id'):
            rows = [r for r in rows if r.trace_id == kw['trace_id']]
        if kw.get('session_id'):
            rows = [r for r in rows if r.session_id == kw['session_id']]
        for condition in json.loads(kw['filter']) if kw.get('filter') else []:
            rows = [r for r in rows if _matches(r, condition)]
        expand = set(filter(None, (kw.get('expand_metadata') or '').split(',')))
        cursor = kw.get('cursor') or '0'
        if cursor.startswith('gap:'):
            return ObservationsV2Response(data=[], meta={'cursor': cursor[4:]})
        offset = int(cursor)
        size = min(kw.get('limit') or 50, self.page_size)
        page = [
            r.model_copy(update={'metadata': _cut(r.metadata, expand)})
            for r in rows[offset : offset + size]
        ]
        more = offset + size < len(rows)
        gap = 'gap:' if self.empty_pages else ''
        return ObservationsV2Response(
            data=page, meta={'cursor': f'{gap}{offset + size}' if more else None}
        )


class FakeScores:
    def __init__(self, scores: list[Any], page_size: int) -> None:
        self.scores = scores
        self.page_size = page_size
        self.calls: list[dict[str, Any]] = []

    def get_many_v3(self, **kw: Any) -> GetScoresV3Response:
        self.calls.append(kw)

        def keep(s: Any) -> bool:
            raw = s.raw
            subject = raw.subject
            trace_id = (
                subject.id
                if subject.kind == 'trace'
                else getattr(subject, 'trace_id', None)
            )
            ts = _time(raw.timestamp)
            return (
                (not kw.get('trace_id') or trace_id == kw['trace_id'])
                and (not kw.get('session_id') or s.session_id == kw['session_id'])
                and (not kw.get('name') or raw.name == kw['name'])
                and _time(kw['from_timestamp']) <= ts < _time(kw['to_timestamp'])
            )

        kept = [s.raw for s in self.scores if keep(s)]
        offset = int(kw.get('cursor') or 0)
        size = min(kw.get('limit') or 50, self.page_size)
        more = offset + size < len(kept)
        return GetScoresV3Response(
            data=kept[offset : offset + size],
            meta={'limit': size, 'cursor': str(offset + size) if more else None},
        )


def make_loader(
    rows: list[ObservationV2],
    scores: list[Any] = (),  # type: ignore[assignment]
    *,
    page_size: int = 1000,
    score_page_size: int = 100,
    empty_pages: bool = False,
) -> LangfuseTraceLoader:
    loader = LangfuseTraceLoader(
        public_key='pk-test', secret_key='sk-test', host='http://fake', request_pacing=0
    )
    loader.client = SimpleNamespace(  # type: ignore[assignment]
        api=SimpleNamespace(
            observations=FakeObservations(list(rows), page_size, empty_pages),
            scores_v3=FakeScores(list(scores), score_page_size),
        )
    )
    loader._client_initialized = True
    return loader
