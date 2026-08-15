from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TraceScore:
    """
    Normalised representation of a Langfuse eval score attached to a trace.

    Covers all four Langfuse score types (NUMERIC, CATEGORICAL, BOOLEAN,
    CORRECTION). ``value`` is always present; ``string_value`` is set for
    CATEGORICAL, BOOLEAN, and CORRECTION variants.
    """

    name: str
    value: float
    data_type: str  # 'NUMERIC' | 'CATEGORICAL' | 'BOOLEAN' | 'CORRECTION'
    trace_id: Optional[str] = None
    observation_id: Optional[str] = None
    string_value: Optional[str] = None
    comment: Optional[str] = None
    source: Optional[str] = None
    timestamp: Optional[Any] = None

    @classmethod
    def from_langfuse(cls, raw: Any) -> 'TraceScore':
        """Map a Langfuse SDK Score union object to ``TraceScore``."""
        # SDK uses camelCase aliases internally but exposes snake_case attrs via Pydantic.
        return cls(
            name=raw.name,
            value=float(raw.value),
            data_type=str(getattr(raw, 'data_type', 'NUMERIC')),
            trace_id=getattr(raw, 'trace_id', None),
            observation_id=getattr(raw, 'observation_id', None),
            string_value=getattr(raw, 'string_value', None),
            comment=getattr(raw, 'comment', None),
            source=str(raw.source)
            if getattr(raw, 'source', None) is not None
            else None,
            timestamp=getattr(raw, 'timestamp', None),
        )


def scores_from_raw_trace(raw: Any) -> list['TraceScore']:
    """
    Read eval scores off a raw Langfuse trace payload.

    Returns an empty list when the payload carries no usable scores — a payload
    whose ``scores`` field holds bare ids rather than score objects reads as
    empty rather than raising.
    """
    parsed: list[TraceScore] = []
    for raw_score in getattr(raw, 'scores', None) or []:
        try:
            parsed.append(TraceScore.from_langfuse(raw_score))
        except (AttributeError, TypeError, ValueError):
            continue
    return parsed


def attach_langfuse_scores(traces: Any, loader: Any, session_id: str) -> None:
    """
    Back-fill Langfuse eval scores onto already-built traces.

    Runs after construction because the traces have to exist to be matched by id,
    which also means a session built with ``turns_only=True`` has already dropped
    its non-turn traces and those scores have nowhere to land.

    Langfuse's session-scoped score query filters on a score's own ``sessionId``,
    which is unset on a score written against a trace, so that query returns only
    session-level scores. Per-trace scores are recovered from the trace payload,
    which already carries them and costs no extra request.

    One paginated call per session, and a no-op against a loader that cannot
    fetch scores, so a caller need not know which loader it holds.
    """
    scores_by_trace: dict[str, list[TraceScore]] = {}
    if hasattr(loader, 'fetch_scores_for_session'):
        scores_by_trace = loader.fetch_scores_for_session(session_id)

    for trace in traces:
        tid = str(getattr(trace.raw, 'id', '') or '')
        trace_scores = list(scores_by_trace.get(tid, []))
        if not trace_scores:
            trace_scores = scores_from_raw_trace(trace.raw)
        if trace_scores:
            trace._scores = trace_scores
