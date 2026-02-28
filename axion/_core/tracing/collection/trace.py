from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

from axion._core.logging import get_logger
from axion._core.tracing.collection.models import ObservationsView, TraceView
from axion._core.tracing.collection.smart_access import SmartAccess, _normalize_key

if TYPE_CHECKING:
    from axion._core.tracing.collection.observation_node import ObservationNode

logger = get_logger(__name__)

_MISSING = object()


def _safe_get(obs: Any, field: str) -> Any:
    """Safely extract a field from an ObservationsView or dict. Local to trace.py."""
    try:
        return obs._lookup(field)
    except (KeyError, AttributeError, TypeError, NotImplementedError):
        return getattr(obs, field, None)


class TraceStep(SmartAccess):
    """
    Represents a specific named step (e.g. 'recommendation').

    SmartAccess enables: ``step.generation``, ``step.variables.caseAssessment``
    """

    def __init__(
        self,
        name: str,
        observations: List[ObservationsView],
        prompt_patterns: Any = None,
    ):
        self.name = name
        self.observations = observations
        self.prompt_patterns = prompt_patterns

    @property
    def count(self) -> int:
        return len(self.observations)

    @property
    def first(self) -> Optional[ObservationsView]:
        return self.observations[0] if self.observations else None

    @property
    def last(self) -> Optional[ObservationsView]:
        return self.observations[-1] if self.observations else None

    def extract_variables(self) -> Dict[str, str]:
        """Extract prompt variables using the configured prompt patterns."""
        return self._extract_variables()

    def _lookup(self, key: str) -> Any:
        if key == 'variables':
            return self._extract_variables()

        target_type = self._resolve_type_alias(key)

        for obs in self.observations:
            if getattr(obs, 'type', '').upper() == target_type:
                return obs

        raise KeyError(
            f"No observation of type '{target_type}' found in step '{self.name}'."
        )

    _TYPE_ALIASES = {
        'CONTEXT': 'SPAN',
        'SPAN': 'SPAN',
        'GENERATION': 'GENERATION',
        'EVENT': 'EVENT',
    }

    @classmethod
    def _resolve_type_alias(cls, key: str) -> str:
        key_upper = key.upper()
        return cls._TYPE_ALIASES.get(key_upper, key_upper)

    def _extract_variables(self) -> Dict[str, str]:
        """Lazy extraction of prompt variables."""
        if self.prompt_patterns is None:
            return {}
        try:
            gen = self._lookup('GENERATION')
            raw_text = getattr(gen, 'input', '')
            prompt_text = self._normalize_prompt_text(raw_text)
            if not prompt_text:
                return {}

            patterns = self.prompt_patterns.get_for(self.name)
            extracted = {}
            for k, pattern in patterns.items():
                match = re.search(pattern, prompt_text, re.DOTALL)
                if match:
                    extracted[k] = match.group(1).strip()
            return extracted
        except (KeyError, AttributeError):
            return {}

    @staticmethod
    def _normalize_prompt_text(raw_text: Any) -> str:
        if isinstance(raw_text, str):
            return raw_text
        if isinstance(raw_text, dict):
            for key in ('content', 'prompt', 'text'):
                value = raw_text.get(key)
                if isinstance(value, str):
                    return value
            return ''
        if isinstance(raw_text, list):
            parts: list[str] = []
            for item in raw_text:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for key in ('content', 'prompt', 'text'):
                        value = item.get(key)
                        if isinstance(value, str):
                            parts.append(value)
                            break
            return '\n'.join(parts)
        return ''

    def __repr__(self):
        types = [getattr(o, 'type', '') for o in self.observations]
        return f"<TraceStep name='{self.name}' types={types}>"


class Trace(SmartAccess):
    """
    Main trace wrapper.

    Wraps a raw trace object (dict, SDK object, or list of observations),
    groups observations by name into steps, and exposes both step-based
    and trace-level attribute access through SmartAccess.
    """

    def __init__(
        self,
        trace_data: Any,
        prompt_patterns: Any = None,
    ):
        self._trace_obj, raw_observations = self._coerce_trace_input(trace_data)

        # Normalize observation dicts into ObservationsView
        self._raw_observations = [
            ObservationsView(data=obs) if isinstance(obs, dict) else obs
            for obs in raw_observations
        ]

        self._grouped: dict[str, list[ObservationsView]] = {}
        self._prompt_patterns = prompt_patterns
        self._steps_cache: dict[str, TraceStep] = {}
        self._tree_roots_cache: Optional[list] = None
        self._group_observations()

    @staticmethod
    def _coerce_trace_input(trace_data: Any) -> tuple[Any, list[Any]]:
        # SDK object style: trace.observations exists and is a list
        if hasattr(trace_data, 'observations') and isinstance(
            trace_data.observations, list
        ):
            return trace_data, trace_data.observations

        # Dict payload style
        if isinstance(trace_data, dict) and isinstance(
            trace_data.get('observations'), list
        ):
            return TraceView(data=trace_data), trace_data.get('observations', [])

        # Raw observations list style
        if isinstance(trace_data, list):
            return None, trace_data

        # Unknown/partial object style: keep object, no observations
        return trace_data, []

    def _group_observations(self):
        for obs in self._raw_observations:
            name = getattr(obs, 'name', 'unnamed')
            self._grouped.setdefault(name, []).append(obs)

    def _get_step(self, step_name: str) -> TraceStep:
        step = self._steps_cache.get(step_name)
        if step is None:
            step = TraceStep(step_name, self._grouped[step_name], self._prompt_patterns)
            self._steps_cache[step_name] = step
        return step

    @property
    def steps(self) -> Dict[str, TraceStep]:
        """All named steps as a dict."""
        return {name: self._get_step(name) for name in self._grouped}

    @property
    def step_names(self) -> List[str]:
        """Ordered list of step names."""
        return list(self._grouped.keys())

    @property
    def observations(self) -> List[ObservationsView]:
        """Flat list of all observations."""
        return list(self._raw_observations)

    @property
    def raw(self) -> Any:
        """The underlying raw trace object."""
        return self._trace_obj

    @property
    def name(self) -> Optional[str]:
        """Trace-level name (workflow / pipeline name)."""
        return self._get_trace_attr('name')

    @property
    def input(self) -> Any:
        """Trace-level input."""
        return self._get_trace_attr('input')

    @property
    def output(self) -> Any:
        """Trace-level output."""
        return self._get_trace_attr('output')

    def _get_trace_attr(self, key: str) -> Any:
        """Read an attribute directly from the root trace object, bypassing steps."""
        if self._trace_obj is None:
            return None
        try:
            if hasattr(self._trace_obj, '_lookup'):
                return self._trace_obj._lookup(key)
        except (KeyError, AttributeError, TypeError):
            pass
        return getattr(self._trace_obj, key, None)

    def walk(self) -> Generator[ObservationNode, None, None]:
        """
        Pre-order depth-first traversal across **all** tree roots.

        Works regardless of whether the trace has one root or many::

            for node in trace.walk():
                print("  " * node.depth + node.name)
        """
        for root in self.tree_roots:
            yield from root.walk()

    def find(
        self,
        name: Optional[str] = None,
        type: Optional[str] = None,
    ) -> Optional[ObservationNode]:
        """
        Return the first node matching *name* and/or *type* across all roots.

        Searches every root's subtree in order and returns the first match,
        or ``None`` if nothing matches.
        """
        for node in self.walk():
            if name is not None:
                node_name = _safe_get(node._observation, 'name')
                if node_name != name:
                    continue
            if type is not None:
                node_type = _safe_get(node._observation, 'type')
                if node_type != type:
                    continue
            return node
        return None

    def _lookup(self, key: str) -> Any:
        # Check grouped step names
        if key in self._grouped:
            return self._get_step(key)

        # Check root trace object attributes
        value = self._lookup_root_exact(key)
        if value is not _MISSING:
            return value

        raise KeyError(f"Attribute '{key}' not found in Trace steps or properties.")

    def _lookup_insensitive(self, key: str) -> Any:
        target = _normalize_key(key)

        # Check step names
        for step_name in self._grouped:
            if _normalize_key(step_name) == target:
                return self._get_step(step_name)

        # Check root object attributes
        value = self._lookup_root_insensitive(key, target)
        if value is not _MISSING:
            return value

        return None

    def _lookup_root_exact(self, key: str) -> Any:
        if self._trace_obj is None:
            return _MISSING

        if hasattr(self._trace_obj, key):
            return getattr(self._trace_obj, key)

        if isinstance(self._trace_obj, (TraceView, dict, ObservationsView)):
            try:
                if hasattr(self._trace_obj, '_lookup'):
                    return self._trace_obj._lookup(key)
                return self._trace_obj[key]
            except (KeyError, AttributeError, TypeError):
                return _MISSING

        return _MISSING

    def _lookup_root_insensitive(self, key: str, target: str) -> Any:
        if self._trace_obj is None:
            return _MISSING

        if hasattr(self._trace_obj, 'keys'):
            try:
                if hasattr(self._trace_obj, '_lookup_insensitive'):
                    value = self._trace_obj._lookup_insensitive(key)
                    if value is not None:
                        return value

                for obj_key in self._trace_obj.keys():
                    if _normalize_key(obj_key) == target:
                        return self._trace_obj[obj_key]
            except (AttributeError, KeyError, TypeError):
                pass

        for attr_name in dir(self._trace_obj):
            if _normalize_key(attr_name) == target:
                try:
                    return getattr(self._trace_obj, attr_name)
                except AttributeError:
                    continue

        return _MISSING

    @property
    def tree_roots(self) -> list:
        """
        Root :class:`ObservationNode` instances reconstructed from
        ``parent_observation_id`` fields.

        Lazy-built and cached. Always returns a list (empty when there
        are no observations).
        """
        if self._tree_roots_cache is None:
            self._tree_roots_cache = self._build_tree_roots()
        return self._tree_roots_cache

    @property
    def tree(self) -> Any:
        """
        Convenience accessor: the single root node when exactly one root
        exists, otherwise ``None``.

        Use :attr:`tree_roots` when you need to handle the multi-root case.
        """
        roots = self.tree_roots
        if len(roots) == 1:
            return roots[0]
        return None

    def _build_tree_roots(self) -> list:
        from axion._core.tracing.collection.observation_node import (
            ObservationNode,
            _safe_obs_get,
        )

        if not self._raw_observations:
            return []

        # Create nodes, index by id.
        nodes: dict[str, ObservationNode] = {}
        obs_order: dict[str, int] = {}
        for idx, obs in enumerate(self._raw_observations):
            obs_id = _safe_get(obs, 'id') or f'__idx_{idx}'
            if obs_id in nodes:
                logger.debug(
                    'Duplicate observation id %r at index %d, skipping.', obs_id, idx
                )
                continue
            nodes[obs_id] = ObservationNode(obs)
            obs_order[obs_id] = idx

        # Wire parent/child relationships
        roots: list[ObservationNode] = []
        seen: set[str] = set()
        for idx, obs in enumerate(self._raw_observations):
            obs_id = _safe_get(obs, 'id') or f'__idx_{idx}'
            if obs_id not in nodes or obs_id in seen:
                continue  # was a duplicate, already skipped
            seen.add(obs_id)
            node = nodes[obs_id]
            parent_id = _safe_get(obs, 'parent_observation_id')

            if not parent_id or parent_id == obs_id or parent_id not in nodes:
                roots.append(node)
            else:
                nodes[parent_id]._add_child(node)

        # Sort roots and children by (start_time, original_index)
        def _sort_key(n: ObservationNode) -> tuple:
            t = n.start_time
            oid = _safe_obs_get(n.observation, 'id') or ''
            return (t or datetime.min, obs_order.get(str(oid), 0))

        roots.sort(key=_sort_key)
        for node in _all_nodes(roots):
            node._children.sort(key=_sort_key)

        return roots

    def __repr__(self):
        base = f'<Trace steps={list(self._grouped.keys())}'
        if self._trace_obj is not None:
            tid = getattr(self._trace_obj, 'id', 'unknown')
            base += f" id='{tid}'"
        base += '>'
        return base


def _all_nodes(roots: list) -> list:
    """Flatten a list of root ObservationNodes into all nodes (pre-order)."""
    result: list = []
    for root in roots:
        result.extend(root.walk())
    return result
