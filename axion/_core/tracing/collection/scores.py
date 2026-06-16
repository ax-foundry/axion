from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TraceScore:
    """Normalised representation of a Langfuse eval score attached to a trace.

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
