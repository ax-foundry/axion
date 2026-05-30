from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Optional

from axion._core.logging import get_logger
from axion._core.tracing.collection.session import Session, TurnPredicate

logger = get_logger(__name__)


class SessionCollection:
    """
    Wraps many Langfuse **sessions**, mirroring :class:`TraceCollection`'s
    ergonomics one level up.

    Each item is a :class:`Session` (which itself composes a ``TraceCollection``),
    so a ``SessionCollection`` supports iteration, indexing, filtering, JSON
    round-tripping, cross-session aggregation, and conversion to a multi-turn
    :class:`~axion.dataset.Dataset` (one item per session).
    """

    def __init__(
        self,
        data: List[Any],
        prompt_patterns: Any = None,
        turn_name: Optional[str] = None,
        turn_predicate: Optional[TurnPredicate] = None,
        turns_only: bool = True,
    ):
        self._prompt_patterns = prompt_patterns
        self._turn_name = turn_name
        self._turn_predicate = turn_predicate
        self._turns_only = turns_only
        # Already-built Session instances keep their own turn config; raw items
        # are constructed with the collection-level default selector.
        self._sessions = [
            item
            if isinstance(item, Session)
            else Session(
                item,
                prompt_patterns=prompt_patterns,
                turn_name=turn_name,
                turn_predicate=turn_predicate,
                turns_only=turns_only,
            )
            for item in data
        ]

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    @classmethod
    def from_langfuse(
        cls,
        session_ids: List[str],
        *,
        loader: Any = None,
        prompt_patterns: Any = None,
        show_progress: bool = True,
        enrich: bool = True,
        turn_name: Optional[str] = None,
        turn_predicate: Optional[TurnPredicate] = None,
        turns_only: bool = True,
        trace_name: Optional[str] = None,
        trace_predicate: Optional[Callable[[Any], bool]] = None,
    ) -> SessionCollection:
        """
        Fetch the given Langfuse sessions and wrap them.

        Missing-session policy: not-found ids are **skipped** (logged at WARNING),
        so the collection contains only sessions that exist. This differs from
        :meth:`Session.from_langfuse`, which returns an empty identifiable
        ``Session`` for a single missing id.

        Args:
            session_ids: Explicit Langfuse session IDs to fetch.
            loader: Pre-configured ``LangfuseTraceLoader``. If *None*, one is
                created from environment variables.
            prompt_patterns: Optional ``PromptPatternsBase`` for variable extraction.
            show_progress: Show a tqdm progress bar while fetching each session.
            enrich: When ``True`` (default), fetch full traces with observations
                (one API call per trace). When ``False``, use only the session
                stub traces (one API call per session): the conversation is still
                reconstructable from trace-level I/O, but observation-level access
                (``by_type``/``tools``/``find_all``) will be empty. Much faster for
                conversation-only workflows over many sessions.
            turn_name: Default turn trace name applied to every session's
                ``conversation()``/``to_dataset()``/``turn_count`` (e.g.
                ``'chat-turn'``); overridable per-call. Reliable -- bypasses
                best-effort auto-detection.
            turn_predicate: Default ``(Trace) -> bool`` turn selector; wins over
                ``turn_name`` when both are given.
            turns_only: When ``True`` (default), additionally prune each
                session's ``traces`` down to the resolved turn selector, so
                ``session[i]``/``by_type()`` only ever see turn traces. Set
                ``False`` to keep every trace (e.g. pipeline runs needed for
                tool aggregation).
            trace_name: When given, only traces whose name equals this value are
                fetched/enriched -- non-matching traces are filtered out at the
                loader **before** any per-trace API call, so they are never pulled.
                Use this (rather than ``turns_only``) when you want to avoid
                fetching pipeline traces entirely. Often set equal to ``turn_name``.
            trace_predicate: When given, only stub traces for which this
                ``(stub) -> bool`` returns ``True`` are fetched (combined with
                ``trace_name``). Filters at the loader, before enrichment.
        """
        if loader is None:
            from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader

            loader = LangfuseTraceLoader()

        sessions: List[Session] = []
        for session_id in session_ids:
            session_obj, full_traces = loader.get_session_with_traces(
                session_id,
                show_progress=show_progress,
                enrich=enrich,
                trace_name=trace_name,
                trace_predicate=trace_predicate,
            )
            if session_obj is None:
                logger.warning('Session %s not found; skipping.', session_id)
                continue
            sessions.append(
                Session(
                    session_obj,
                    full_traces=full_traces,
                    prompt_patterns=prompt_patterns,
                    turn_name=turn_name,
                    turn_predicate=turn_predicate,
                    turns_only=turns_only,
                )
            )

        return cls(
            sessions,
            prompt_patterns=prompt_patterns,
            turn_name=turn_name,
            turn_predicate=turn_predicate,
            turns_only=turns_only,
        )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        prompt_patterns: Any = None,
        turn_name: Optional[str] = None,
        turn_predicate: Optional[TurnPredicate] = None,
        turns_only: bool = True,
    ) -> SessionCollection:
        """Load a SessionCollection from a JSON file (list of session dicts)."""
        source = Path(path).expanduser()
        data = json.loads(source.read_text())
        if not isinstance(data, list):
            raise ValueError('SessionCollection.load_json expects a JSON list.')
        return cls(
            data,
            prompt_patterns=prompt_patterns,
            turn_name=turn_name,
            turn_predicate=turn_predicate,
            turns_only=turns_only,
        )

    # ------------------------------------------------------------------ #
    # Sequence protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._sessions)

    def __getitem__(self, index: int) -> Session:
        return self._sessions[index]

    def __iter__(self):
        return iter(self._sessions)

    def __repr__(self):
        return f'<SessionCollection count={len(self._sessions)}>'

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #

    def filter(self, condition: Callable[[Session], bool]) -> SessionCollection:
        """Return a new SessionCollection of sessions matching *condition*."""
        filtered = [s for s in self._sessions if condition(s)]
        return self._from_sessions(filtered)

    def filter_by(self, **kwargs: Any) -> SessionCollection:
        """
        Simple attribute-equality filter.

        Example::

            collection.filter_by(environment='production')
        """
        return self.filter(
            lambda session: all(
                self._session_attr_equals(session, attr_name, expected)
                for attr_name, expected in kwargs.items()
            )
        )

    @staticmethod
    def _session_attr_equals(session: Session, attr_name: str, expected: Any) -> bool:
        """Safely compare a session attribute against an expected value."""
        try:
            return getattr(session, attr_name) == expected
        except AttributeError:
            return False

    def _from_sessions(self, sessions: List[Session]) -> SessionCollection:
        """Build a new SessionCollection from already-wrapped Session objects."""
        return SessionCollection(
            sessions,
            prompt_patterns=self._prompt_patterns,
            turn_name=self._turn_name,
            turn_predicate=self._turn_predicate,
            turns_only=self._turns_only,
        )

    # ------------------------------------------------------------------ #
    # Aggregation across sessions
    # ------------------------------------------------------------------ #

    def by_type(self, type_str: str) -> List[Any]:
        """All observations of *type_str* across every trace in every session."""
        return [obs for session in self._sessions for obs in session.by_type(type_str)]

    def tools(self) -> List[Any]:
        """All ``TOOL`` observations across every session."""
        return self.by_type('TOOL')

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_list(self) -> List[Any]:
        """Return each session as a JSON-able dict."""
        return [s.to_dict() for s in self._sessions]

    def save_json(self, path: str | Path) -> None:
        """Serialize to a JSON file (list of session dicts)."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_list(), indent=2, sort_keys=True))

    # ------------------------------------------------------------------ #
    # Dataset conversion
    # ------------------------------------------------------------------ #

    def to_dataset(
        self,
        name: str = 'session_dataset',
        transform: Optional[Callable[[Session], Any]] = None,
    ) -> Any:
        """
        Convert to an axion ``Dataset`` with one multi-turn ``DatasetItem`` per
        session (sessions with no turns are skipped).

        A ``transform(session)`` may return a ``DatasetItem``, a dict, or a list
        of those (flattened) -- the same contract as :meth:`Session.to_dataset`.
        """
        from axion.dataset import Dataset

        items: List[Any] = []
        for session in self._sessions:
            items.extend(session._build_dataset_items(transform))
        return Dataset(name=name, items=items)
