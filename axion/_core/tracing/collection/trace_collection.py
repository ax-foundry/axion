from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from axion._core.logging import get_logger
from axion._core.tracing.collection.models import ObservationsView, TraceView
from axion._core.tracing.collection.smart_access import SmartAccess
from axion._core.tracing.collection.trace import Trace

logger = get_logger(__name__)


class TraceCollection:
    """
    Wraps a list of raw trace data items, converting each to a :class:`Trace`.

    Supports iteration, indexing, filtering, JSON round-tripping, and
    conversion to an axion :class:`~axion.dataset.Dataset`.
    """

    def __init__(
        self,
        data: List[Any],
        prompt_patterns: Any = None,
    ):
        self._prompt_patterns = prompt_patterns
        self._traces = [Trace(item, prompt_patterns=prompt_patterns) for item in data]

    @classmethod
    def from_langfuse(
        cls,
        trace_ids: list[str] | None = None,
        *,
        limit: int = 50,
        days_back: int = 7,
        tags: list[str] | None = None,
        name: str | None = None,
        loader: Any = None,
        show_progress: bool = True,
        prompt_patterns: Any = None,
        **kwargs: Any,
    ) -> TraceCollection:
        """
        Fetch traces from Langfuse and return a TraceCollection.

        Args:
            trace_ids: Specific trace IDs to fetch (bypasses other filters).
            limit: Maximum number of traces when not using trace_ids.
            days_back: How many days back to look.
            tags: Filter by Langfuse tags.
            name: Filter by trace name.
            loader: Pre-configured LangfuseTraceLoader instance. If *None*,
                a new one is created from environment variables.
            show_progress: Show a tqdm progress bar while fetching.
            prompt_patterns: Optional PromptPatternsBase for variable extraction.
            **kwargs: Forwarded to ``LangfuseTraceLoader.fetch_traces()``.
        """
        if loader is None:
            from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader

            loader = LangfuseTraceLoader()

        if trace_ids is not None:
            raw_traces = loader.fetch_traces(
                trace_ids=trace_ids, show_progress=show_progress, **kwargs
            )
        else:
            raw_traces = loader.fetch_traces(
                limit=limit,
                days_back=days_back,
                tags=tags,
                name=name,
                show_progress=show_progress,
                **kwargs,
            )

        return cls(raw_traces, prompt_patterns=prompt_patterns)

    @classmethod
    def from_raw_traces(
        cls,
        raw_traces: List[Any],
        *,
        name: str | None = None,
        prompt_patterns: Any = None,
    ) -> TraceCollection:
        """
        Wrap pre-fetched raw trace objects.

        Args:
            raw_traces: List of raw trace objects (dicts, SDK objects, etc.).
            name: Optional collection name (currently informational).
            prompt_patterns: Optional PromptPatternsBase for variable extraction.
        """
        return cls(raw_traces, prompt_patterns=prompt_patterns)

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        prompt_patterns: Any = None,
    ) -> TraceCollection:
        """Load a TraceCollection from a JSON file."""
        source = Path(path).expanduser()
        data = json.loads(source.read_text())
        if not isinstance(data, list):
            raise ValueError('TraceCollection.load_json expects a JSON list.')
        return cls(data, prompt_patterns=prompt_patterns)

    def __len__(self) -> int:
        return len(self._traces)

    def __getitem__(self, index: int) -> Trace:
        return self._traces[index]

    def __iter__(self):
        return iter(self._traces)

    def __repr__(self):
        return f'<TraceCollection count={len(self._traces)}>'

    def filter(self, condition: Callable[[Trace], bool]) -> TraceCollection:
        """
        Return a new TraceCollection containing only traces that match *condition*.

        Args:
            condition: A callable that receives a :class:`Trace` and returns bool.
        """
        filtered = [t for t in self._traces if condition(t)]
        return self._from_traces(filtered)

    def filter_by(self, **kwargs: Any) -> TraceCollection:
        """
        Simple attribute-equality filter.

        Example::

            collection.filter_by(name='my-trace')
        """
        return self.filter(
            lambda trace: all(
                self._trace_attr_equals(trace, attr_name, expected)
                for attr_name, expected in kwargs.items()
            )
        )

    @staticmethod
    def _trace_attr_equals(trace: Trace, attr_name: str, expected: Any) -> bool:
        """Safely compare a trace attribute against an expected value."""
        try:
            return getattr(trace, attr_name) == expected
        except AttributeError:
            return False

    def save_json(self, path: str | Path) -> None:
        """Serialize to a JSON file."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [self._to_jsonable(t._trace_obj) for t in self._traces]
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def to_list(self) -> List[Any]:
        """Return the raw trace objects."""
        return [t._trace_obj for t in self._traces]

    def to_dataset(
        self,
        name: str = 'trace_dataset',
        transform: Optional[Callable[[Trace], Any]] = None,
    ) -> Any:
        """
        Convert to an axion :class:`~axion.dataset.Dataset`.

        Args:
            name: Name for the resulting dataset.
            transform: Optional callable ``(Trace) -> DatasetItem | dict``.
                Can return a :class:`~axion.dataset.DatasetItem` directly
                or a dict of fields to construct one.  When *None*, uses the
                default extraction logic (query/actual_output from trace-level
                input/output).

        Returns:
            An axion ``Dataset`` instance.
        """
        from axion.dataset import Dataset, DatasetItem

        items: list[DatasetItem] = []
        for trace in self._traces:
            if transform is not None:
                result = transform(trace)
            else:
                result = self._default_transform(trace)

            if not result:
                continue

            if isinstance(result, DatasetItem):
                items.append(result)
            elif isinstance(result, dict):
                items.append(DatasetItem(**result))

        return Dataset(name=name, items=items)

    @staticmethod
    def _default_transform(trace: Trace) -> Dict[str, Any]:
        """
        Default extraction: pull query from trace input and actual_output
        from trace output, using the same key-search logic as BaseTraceLoader.
        """
        trace_obj = trace._trace_obj
        if trace_obj is None:
            return {}

        raw_input, raw_output, trace_id = TraceCollection._extract_trace_io(trace_obj)

        query = _extract_query(raw_input) if raw_input is not None else ''
        actual_output = _extract_output(raw_output) if raw_output is not None else ''

        if not query and not actual_output:
            return {}

        result: Dict[str, Any] = {
            'query': query,
            'actual_output': actual_output,
        }
        if trace_id:
            result['trace_id'] = str(trace_id)

        return result

    @staticmethod
    def _extract_trace_io(trace_obj: Any) -> tuple[Any, Any, Any]:
        """Extract raw input/output/id from supported trace object shapes."""
        # Access raw data directly to avoid SmartAccess wrapping.
        if isinstance(trace_obj, (TraceView, ObservationsView)):
            data = trace_obj._data
            return data.get('input'), data.get('output'), data.get('id')

        if isinstance(trace_obj, dict):
            return trace_obj.get('input'), trace_obj.get('output'), trace_obj.get('id')

        return (
            getattr(trace_obj, 'input', None),
            getattr(trace_obj, 'output', None),
            getattr(trace_obj, 'id', None),
        )

    def _from_traces(self, traces: List[Trace]) -> TraceCollection:
        """Build a new TraceCollection from already-wrapped Trace objects."""
        raw = [t._trace_obj for t in traces]
        return TraceCollection(raw, prompt_patterns=self._prompt_patterns)

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """Recursively convert a value to a JSON-serializable form."""
        convert = TraceCollection._to_jsonable

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}

        if isinstance(value, list):
            return [convert(v) for v in value]

        # SmartAccess wrappers (TraceView, ObservationsView) store data in _data
        if isinstance(value, SmartAccess) and hasattr(value, '_data'):
            return convert(value._data)

        if is_dataclass(value) and not isinstance(value, type):
            return convert(asdict(value))

        model_dump = getattr(value, 'model_dump', None)
        if callable(model_dump):
            return convert(model_dump())

        if hasattr(value, 'dict'):
            try:
                return convert(value.dict())
            except TypeError:
                pass

        if hasattr(value, 'to_dict'):
            return convert(value.to_dict())

        if hasattr(value, '__dict__'):
            return convert(vars(value))

        return str(value)


# ---------------------------------------------------------------------------
# Lightweight query/output extraction (mirrors BaseTraceLoader helpers)
# ---------------------------------------------------------------------------

_QUERY_KEYS = ('query', 'question', 'input', 'message', 'prompt', 'user_input', 'text')
_OUTPUT_KEYS = ('output', 'response', 'answer', 'result', 'content', 'text', 'message')


def _safe_json_load(data: Any) -> Any:
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data
    return data


def _extract_query(input_data: Any) -> str:
    return _extract_by_keys(input_data, _QUERY_KEYS)


def _extract_output(output_data: Any) -> str:
    return _extract_by_keys(output_data, _OUTPUT_KEYS)


def _extract_by_keys(payload: Any, keys: tuple[str, ...]) -> str:
    """Extract text from a payload by prioritized keys, with safe fallbacks."""
    data = _safe_json_load(payload)
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return str(data[key])
        return json.dumps(data)
    return str(data)
