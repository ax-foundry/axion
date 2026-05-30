from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from axion._core.logging import get_logger
from axion._core.schema import AIMessage, HumanMessage, ToolCall, ToolMessage
from axion._core.tracing.collection._io import (
    _OUTPUT_KEYS,
    _QUERY_KEYS,
    _coerce_ts,
    _extract_output,
    _extract_query,
    _extract_text_with_match,
    _extract_trace_io,
    _safe_json_load,
)
from axion._core.tracing.collection.observation_node import ObservationNode
from axion._core.tracing.collection.trace import Trace
from axion._core.tracing.collection.trace_collection import TraceCollection

logger = get_logger(__name__)

# Predicate deciding whether a Trace represents a conversational turn.
TurnPredicate = Callable[[Trace], bool]


class Session:
    """
    Wraps a Langfuse **session** -- a group of traces that together form a
    multi-turn conversation -- exposing session-level metadata, the
    conversation, per-trace access, and aggregation across all traces.

    Composition: a ``Session`` holds a :class:`TraceCollection` of its (chrono-
    logically sorted) traces, so ``session.traces`` and ``session[i]`` reuse all
    existing trace tooling. Session metadata (id, environment, ...) is preserved
    separately and never flattened away.
    """

    def __init__(
        self,
        session_data: Any,
        full_traces: Optional[List[Any]] = None,
        prompt_patterns: Any = None,
        sort: bool = True,
        turn_name: Optional[str] = None,
        turn_predicate: Optional[TurnPredicate] = None,
    ):
        self._prompt_patterns = prompt_patterns
        # Default turn selector applied by conversation()/to_dataset()/turn_count
        # when no per-call name=/is_turn= is given. Lets callers pin turn
        # selection once (e.g. turn_name='athena-chat') instead of repeating it.
        self._turn_name = turn_name
        self._turn_predicate = turn_predicate
        meta, coerced_traces = self._coerce_session_input(session_data)
        self._meta: Dict[str, Any] = meta

        raw_traces = full_traces if full_traces is not None else coerced_traces
        if sort:
            raw_traces = self._sort_chronologically(raw_traces)

        self._traces = TraceCollection(raw_traces, prompt_patterns=prompt_patterns)

        # Lazily computed auto turn-name discovery cache.
        self._auto_turn_names_cache: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    # Input coercion / sorting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_session_input(session_data: Any) -> tuple[Dict[str, Any], List[Any]]:
        """Return ``(metadata_dict, raw_traces)`` from supported session shapes."""
        # SDK object (SessionWithTraces) or any object exposing ``.traces``.
        if hasattr(session_data, 'traces') and not isinstance(session_data, dict):
            traces = list(getattr(session_data, 'traces') or [])
            return Session._session_meta_to_dict(session_data), traces

        # Dict payload (e.g. from a JSON round-trip): metadata + nested traces.
        if isinstance(session_data, dict):
            traces = list(session_data.get('traces') or [])
            meta = {k: v for k, v in session_data.items() if k != 'traces'}
            return meta, traces

        # Bare list of traces, no session-level metadata.
        if isinstance(session_data, list):
            return {}, list(session_data)

        return {}, []

    @staticmethod
    def _session_meta_to_dict(obj: Any) -> Dict[str, Any]:
        """Extract all session-level fields (excluding traces) into a dict."""
        model_dump = getattr(obj, 'model_dump', None)
        if callable(model_dump):
            try:
                data = model_dump()
            except TypeError:
                data = None
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if k != 'traces'}

        if hasattr(obj, '__dict__'):
            return {k: v for k, v in vars(obj).items() if k != 'traces'}

        # Fallback: read the known core attributes individually.
        meta: Dict[str, Any] = {}
        for attr in ('id', 'created_at', 'project_id', 'environment'):
            value = getattr(obj, attr, None)
            if value is not None:
                meta[attr] = value
        return meta

    @staticmethod
    def _sort_chronologically(raw_traces: List[Any]) -> List[Any]:
        """Sort traces by (UTC-aware timestamp, original index), stable."""

        def _key(item: tuple[int, Any]) -> tuple:
            idx, trace = item
            _, _, _, ts = _extract_trace_io(trace)
            return (_coerce_ts(ts), idx)

        ordered = sorted(enumerate(raw_traces), key=_key)
        return [trace for _, trace in ordered]

    # ------------------------------------------------------------------ #
    # Metadata properties
    # ------------------------------------------------------------------ #

    @property
    def metadata(self) -> Dict[str, Any]:
        """Full session-level metadata (all captured fields)."""
        return self._meta

    def _meta_get(self, *keys: str) -> Any:
        """First present (non-None) value among *keys* -- tolerates camelCase."""
        for key in keys:
            value = self._meta.get(key)
            if value is not None:
                return value
        return None

    @property
    def id(self) -> Optional[str]:
        return self._meta_get('id')

    @property
    def created_at(self) -> Any:
        return self._meta_get('created_at', 'createdAt')

    @property
    def project_id(self) -> Optional[str]:
        return self._meta_get('project_id', 'projectId')

    @property
    def environment(self) -> Optional[str]:
        return self._meta_get('environment')

    @property
    def traces(self) -> TraceCollection:
        """The session's traces as a :class:`TraceCollection`."""
        return self._traces

    # ------------------------------------------------------------------ #
    # Turn detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _trace_qualifies_as_turn(trace: Trace) -> bool:
        """A trace qualifies when both input AND output yield real (key-matched) text."""
        raw_in, raw_out, _, _ = _extract_trace_io(trace._trace_obj)
        in_text, in_match = _extract_text_with_match(raw_in, _QUERY_KEYS)
        out_text, out_match = _extract_text_with_match(raw_out, _OUTPUT_KEYS)
        return bool(in_match and in_text.strip() and out_match and out_text.strip())

    def _qualifying_names(self) -> List[str]:
        """First-seen-ordered list of distinct trace names that qualify as turns."""
        if self._auto_turn_names_cache is not None:
            return self._auto_turn_names_cache

        names: List[str] = []
        for trace in self._traces:
            if self._trace_qualifies_as_turn(trace):
                name = trace.name or 'unnamed'
                if name not in names:
                    names.append(name)
        self._auto_turn_names_cache = names
        return names

    @property
    def turn_trace_names(self) -> List[str]:
        """All distinct trace names that auto-detect as conversational turns."""
        return list(self._qualifying_names())

    @property
    def turn_trace_name(self) -> Optional[str]:
        """The single dominant qualifying trace name (highest count, then first-seen)."""
        counts: Dict[str, int] = {}
        order: Dict[str, int] = {}
        for idx, trace in enumerate(self._traces):
            if not self._trace_qualifies_as_turn(trace):
                continue
            name = trace.name or 'unnamed'
            counts[name] = counts.get(name, 0) + 1
            order.setdefault(name, idx)

        if not counts:
            return None
        # Highest count, tie-break by first-seen order (lowest index).
        return min(counts, key=lambda n: (-counts[n], order[n]))

    def _default_is_turn(self) -> TurnPredicate:
        """Best-effort default predicate: qualifying text AND dominant trace name.

        Documented as best-effort: a pipeline/workflow trace whose payload happens
        to contain ``input``/``output``/``message``/``result`` keys can be
        misclassified. Use ``name=`` or ``is_turn=`` for reliable selection.
        """
        dominant = self.turn_trace_name

        def predicate(trace: Trace) -> bool:
            if not self._trace_qualifies_as_turn(trace):
                return False
            return (trace.name or 'unnamed') == dominant

        return predicate

    def _resolve_turn_predicate(
        self,
        name: Optional[str],
        is_turn: Optional[TurnPredicate],
    ) -> TurnPredicate:
        """Resolve the turn predicate.

        Priority: per-call ``is_turn`` > per-call ``name`` > session-level
        ``turn_predicate`` > session-level ``turn_name`` > best-effort auto default.
        """
        if is_turn is not None:
            return is_turn
        if name is not None:
            return lambda trace: trace.name == name
        if self._turn_predicate is not None:
            return self._turn_predicate
        if self._turn_name is not None:
            configured = self._turn_name
            return lambda trace: trace.name == configured
        return self._default_is_turn()

    # ------------------------------------------------------------------ #
    # Conversation
    # ------------------------------------------------------------------ #

    def conversation(
        self,
        name: Optional[str] = None,
        is_turn: Optional[TurnPredicate] = None,
        include_tools: bool = False,
    ) -> Optional[Any]:
        """
        Reconstruct the session's multi-turn conversation.

        Args:
            name: Only treat traces with this exact ``name`` as turns (reliable).
            is_turn: Custom predicate ``(Trace) -> bool`` selecting turns (reliable,
                wins over ``name``).
            include_tools: When True, best-effort attach ``tool_calls`` and paired
                ``ToolMessage``s from each turn's ``TOOL`` observations. Observations
                missing a name are skipped (never fabricated).

        Returns:
            A ``MultiTurnConversation`` with session metadata attached, or ``None``
            when no traces qualify as turns.
        """
        predicate = self._resolve_turn_predicate(name, is_turn)
        return _build_conversation(
            list(self._traces),
            predicate,
            include_tools=include_tools,
            metadata=self._meta or None,
        )

    @property
    def turn_count(self) -> int:
        """Number of traces counted as turns under the session's default selector."""
        predicate = self._resolve_turn_predicate(None, None)
        return sum(1 for t in self._traces if predicate(t))

    # ------------------------------------------------------------------ #
    # Aggregation across traces
    # ------------------------------------------------------------------ #

    def by_type(self, type_str: str) -> List[Any]:
        """All observations of *type_str* across every trace (case-insensitive)."""
        return [obs for trace in self._traces for obs in trace.by_type(type_str)]

    @property
    def observation_types(self) -> List[str]:
        """First-seen-ordered union of observation types across all traces."""
        seen: List[str] = []
        for trace in self._traces:
            for t in trace.observation_types:
                if t not in seen:
                    seen.append(t)
        return seen

    def tools(self) -> List[Any]:
        """All ``TOOL`` observations across every trace."""
        return self.by_type('TOOL')

    def find_all(
        self,
        name: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[ObservationNode]:
        """Every node matching name and/or type across all traces."""
        results: List[ObservationNode] = []
        for trace in self._traces:
            results.extend(trace.find_all(name=name, type=type))
        return results

    # ------------------------------------------------------------------ #
    # Dataset conversion
    # ------------------------------------------------------------------ #

    def to_dataset(
        self,
        name: str = 'session_dataset',
        transform: Optional[Callable[['Session'], Any]] = None,
    ) -> Any:
        """
        Convert this session to an axion ``Dataset``.

        Default: one multi-turn ``DatasetItem`` built from the conversation
        (skipped entirely when there are no turns). A ``transform(session)`` may
        return a ``DatasetItem``, a dict, or a list of those (flattened).
        """
        from axion.dataset import Dataset

        items = self._build_dataset_items(transform)
        return Dataset(name=name, items=items)

    def _build_dataset_items(
        self,
        transform: Optional[Callable[['Session'], Any]],
    ) -> List[Any]:
        from axion.dataset import DatasetItem

        if transform is not None:
            result = transform(self)
        else:
            conversation = self.conversation()
            result = (
                DatasetItem(multi_turn_conversation=conversation)
                if conversation is not None
                else None
            )

        items: List[Any] = []
        for entry in result if isinstance(result, list) else [result]:
            if not entry:
                continue
            if isinstance(entry, DatasetItem):
                items.append(entry)
            elif isinstance(entry, dict):
                items.append(DatasetItem(**entry))
        return items

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Full session metadata + JSON-able traces, round-trippable via __init__."""
        payload = {k: TraceCollection._to_jsonable(v) for k, v in self._meta.items()}
        payload['traces'] = [
            TraceCollection._to_jsonable(t._trace_obj) for t in self._traces
        ]
        return payload

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    @classmethod
    def from_langfuse(
        cls,
        session_id: str,
        *,
        loader: Any = None,
        prompt_patterns: Any = None,
        show_progress: bool = True,
        enrich: bool = True,
        turn_name: Optional[str] = None,
        turn_predicate: Optional[TurnPredicate] = None,
    ) -> 'Session':
        """Fetch a Langfuse session and wrap it.

        Missing-session policy: if the id is not found, this returns an empty but
        *identifiable* ``Session`` (``id == session_id``, no traces) rather than
        raising. This differs from :meth:`SessionCollection.from_langfuse`, which
        **skips** not-found ids -- single-id callers usually want a usable handle
        back, whereas a collection caller wants only the sessions that exist.

        ``enrich`` controls trace fetching: ``True`` (default) fetches full traces
        with observations (one API call per trace); ``False`` uses only the
        session's stub traces (a single API call). Stubs carry trace-level
        input/output -- enough to reconstruct the conversation -- but
        observation-level access (``by_type``/``tools``/``find_all``) will be empty.

        ``turn_name``/``turn_predicate`` set the default turn selector applied by
        ``conversation()``/``to_dataset()``/``turn_count`` (overridable per-call).
        """
        if loader is None:
            from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader

            loader = LangfuseTraceLoader()

        session_obj, full_traces = loader.get_session_with_traces(
            session_id, show_progress=show_progress, enrich=enrich
        )
        if session_obj is None:
            # Build from the raw id so the Session is still usable/identifiable.
            session_obj = {'id': session_id}
        return cls(
            session_obj,
            full_traces=full_traces,
            prompt_patterns=prompt_patterns,
            turn_name=turn_name,
            turn_predicate=turn_predicate,
        )

    # ------------------------------------------------------------------ #
    # Dunders (delegate to the trace collection)
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._traces)

    def __getitem__(self, index: int) -> Trace:
        return self._traces[index]

    def __iter__(self):
        return iter(self._traces)

    def __repr__(self):
        return (
            f"<Session id='{self.id}' traces={len(self._traces)} "
            f'turns={self.turn_count}>'
        )


def _build_conversation(
    traces: List[Trace],
    is_turn: TurnPredicate,
    include_tools: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Build a ``MultiTurnConversation`` from the traces that satisfy *is_turn*.

    Module-level so :class:`SessionCollection` can reuse it without instantiating
    a throwaway ``Session``. Returns ``None`` when no messages are produced
    (``MultiTurnConversation`` rejects an empty message list).
    """
    from axion.dataset_schema import MultiTurnConversation

    messages: List[Any] = []
    for trace in traces:
        if not is_turn(trace):
            continue

        raw_in, raw_out, _, _ = _extract_trace_io(trace._trace_obj)
        user_text = _extract_query(raw_in) if raw_in is not None else ''
        ai_text = _extract_output(raw_out) if raw_out is not None else ''

        if not user_text and not ai_text:
            continue

        if user_text:
            messages.append(HumanMessage(content=user_text))

        tool_calls, tool_messages = (
            _build_tool_messages(trace) if include_tools else ([], [])
        )

        if ai_text or tool_calls:
            ai_kwargs: Dict[str, Any] = {'content': ai_text or None}
            if tool_calls:
                ai_kwargs['tool_calls'] = tool_calls
            messages.append(AIMessage(**ai_kwargs))

        messages.extend(tool_messages)

    if not messages:
        return None
    return MultiTurnConversation(messages=messages, metadata=metadata)


def _raw_obs_value(obs: Any, key: str) -> Any:
    """Read a raw (unwrapped) value off an observation view/dict/object.

    ``by_type`` yields ``ObservationsView``s whose dot-access re-wraps dicts as
    ``SmartDict``; reading ``_data`` directly keeps plain values for schema
    construction and JSON serialization.
    """
    data = getattr(obs, '_data', None)
    if isinstance(data, dict):
        return data.get(key)
    if isinstance(obs, dict):
        return obs.get(key)
    return getattr(obs, key, None)


def _build_tool_messages(trace: Trace) -> tuple[List[ToolCall], List[ToolMessage]]:
    """Best-effort ``(tool_calls, tool_messages)`` from a trace's TOOL observations.

    Skips any observation without a name; never fabricates a tool call. The
    synthesized ``ToolCall.id`` is shared with its paired ``ToolMessage``.
    """
    tool_calls: List[ToolCall] = []
    tool_messages: List[ToolMessage] = []

    for obs in trace.by_type('TOOL'):
        name = _raw_obs_value(obs, 'name')
        if not name:
            continue

        # Tool inputs may be stored as a dict or as a JSON string; decode the
        # latter before deciding the args are unusable.
        args = _safe_json_load(_raw_obs_value(obs, 'input'))
        if not isinstance(args, dict):
            args = {}

        call = ToolCall(name=name, args=args)
        tool_calls.append(call)

        raw_output = _raw_obs_value(obs, 'output')
        if isinstance(raw_output, str):
            content = raw_output
        elif raw_output is not None:
            try:
                content = json.dumps(raw_output, default=str)
            except TypeError:
                content = str(raw_output)
        else:
            content = '[Tool execution completed]'

        tool_messages.append(ToolMessage(content=content, tool_call_id=call.id))

    return tool_calls, tool_messages
