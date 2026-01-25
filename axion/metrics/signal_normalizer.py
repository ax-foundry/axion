from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NormalizedSignal:
    metric_name: str
    group: str
    name: str
    value: Any
    score: Optional[float]
    description: Optional[str] = None
    headline_display: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


def coerce_score(value: Any) -> Optional[float]:
    """
    Coerce a value into a numeric score when possible.

    Mirrors the SignalNode.to_score heuristics for common types.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        val_lower = value.lower()
        if val_lower in ['yes', 'relevant', 'faithful', 'correct']:
            return 1.0
        if val_lower in [
            'no',
            'irrelevant',
            'unfaithful',
            'incorrect',
            'contradictory',
        ]:
            return 0.0
    return None


def _normalize_signal_dict(
    metric_name: str, group: str, signal_dict: Dict[str, Any]
) -> NormalizedSignal:
    name = signal_dict.get('name', 'unknown')
    value = signal_dict.get('value')
    score = signal_dict.get('score')
    if score is None:
        score = coerce_score(value)
    return NormalizedSignal(
        metric_name=metric_name,
        group=group,
        name=name,
        value=value,
        score=score,
        description=signal_dict.get('description'),
        headline_display=bool(signal_dict.get('headline_display', False)),
        raw=signal_dict,
    )


def normalize_signals(metric_name: str, signals: Any) -> List[NormalizedSignal]:
    if signals is None:
        return []

    if hasattr(signals, 'model_dump'):
        signals = signals.model_dump()

    normalized: List[NormalizedSignal] = []

    if isinstance(signals, dict):
        # Grouped dict format: {group: [signal_dict, ...]}
        grouped = all(isinstance(value, list) for value in signals.values())
        if grouped:
            for group, group_signals in signals.items():
                if not isinstance(group_signals, list):
                    continue
                for signal in group_signals:
                    if hasattr(signal, 'model_dump'):
                        signal = signal.model_dump()
                    if isinstance(signal, dict):
                        normalized.append(
                            _normalize_signal_dict(metric_name, group, signal)
                        )
            return normalized

        # Single signal dict or flat key/value map
        if any(key in signals for key in ('name', 'value', 'score')):
            return [_normalize_signal_dict(metric_name, 'overall', signals)]

        for key, value in signals.items():
            if hasattr(value, 'model_dump'):
                value = value.model_dump()
            if isinstance(value, dict) and any(
                subkey in value for subkey in ('name', 'value', 'score')
            ):
                signal_dict = dict(value)
                signal_dict.setdefault('name', key)
                normalized.append(
                    _normalize_signal_dict(metric_name, 'overall', signal_dict)
                )
            else:
                normalized.append(
                    _normalize_signal_dict(
                        metric_name, 'overall', {'name': key, 'value': value}
                    )
                )
        return normalized

    if isinstance(signals, list):
        for index, item in enumerate(signals):
            if hasattr(item, 'model_dump'):
                item = item.model_dump()
            if isinstance(item, dict):
                group = item.get('group', 'overall')
                normalized.append(_normalize_signal_dict(metric_name, group, item))
            else:
                normalized.append(
                    _normalize_signal_dict(
                        metric_name,
                        'overall',
                        {'name': f'item_{index}', 'value': item},
                    )
                )
        return normalized

    return normalized
