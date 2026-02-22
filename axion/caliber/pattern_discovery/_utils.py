import hashlib
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from axion.caliber.pattern_discovery.models import EvidenceItem, LearningArtifact

DEFAULT_DENIED_KEYS: frozenset = frozenset(
    {
        'email',
        'phone',
        'name',
        'address',
        'ip',
        'user_id',
        'account_id',
        'order_id',
        'ssn',
        'password',
        'token',
        'credit_card',
        'dob',
        'date_of_birth',
    }
)
DEFAULT_MAX_TAG_LENGTH = 50
DEFAULT_MAX_TAGS = 10


@dataclass
class MetadataConfig:
    """Configuration for metadata handling in clustering and distillation."""

    allowed_keys: Optional[Set[str]] = None
    denied_keys: Set[str] = field(default_factory=lambda: set(DEFAULT_DENIED_KEYS))
    max_keys: int = 6
    max_value_length: int = 50
    max_header_chars: int = 150
    include_in_clustering: bool = False
    include_in_distillation: bool = True


def _filter_metadata(
    metadata: Dict[str, Any], config: MetadataConfig
) -> Dict[str, str]:
    """Filter and truncate metadata keys/values according to config."""
    filtered: Dict[str, str] = {}
    for key, value in metadata.items():
        if config.allowed_keys is not None and key not in config.allowed_keys:
            continue
        if key in config.denied_keys:
            continue
        str_val = str(value)[: config.max_value_length]
        filtered[key] = str_val
        if len(filtered) >= config.max_keys:
            break
    return filtered


def format_metadata_header(
    metadata: Dict[str, Any], config: MetadataConfig
) -> Optional[str]:
    """Format metadata into a compact header string.

    Returns ``[meta: key=val, key=val]`` capped at ``max_header_chars``,
    or ``None`` if disabled or no keys pass the filter.
    """
    filtered = _filter_metadata(metadata, config)
    if not filtered:
        return None

    parts = [f'{k}={v}' for k, v in filtered.items()]
    header = '[meta: ' + ', '.join(parts) + ']'
    if len(header) > config.max_header_chars:
        header = header[: config.max_header_chars - 1] + ']'
    return header


def aggregate_cluster_metadata(
    items: Dict[str, EvidenceItem],
    item_ids: List[str],
    config: MetadataConfig,
) -> Optional[str]:
    """Aggregate metadata across cluster members into a summary string.

    Returns something like ``"failed_step: checkout (80%), billing (20%)"``.
    """
    if not item_ids:
        return None

    key_counters: Dict[str, Counter] = {}
    for iid in item_ids:
        item = items.get(iid)
        if item is None:
            continue
        filtered = _filter_metadata(item.metadata, config)
        for k, v in filtered.items():
            if k not in key_counters:
                key_counters[k] = Counter()
            key_counters[k][v] += 1

    if not key_counters:
        return None

    total = len([iid for iid in item_ids if iid in items])
    if total == 0:
        return None

    parts: List[str] = []
    for key, counter in key_counters.items():
        top = counter.most_common(3)
        dist = ', '.join(f'{val} ({count * 100 // total}%)' for val, count in top)
        parts.append(f'{key}: {dist}')

    return '; '.join(parts)


ExcerptFn = Callable[[EvidenceItem], str]


def default_excerpt(item: EvidenceItem, max_chars: int = 200) -> str:
    """First N chars of text."""
    return item.text[:max_chars]


def validate_learning(
    learning: Any,
    cluster_item_ids: Set[str],
    min_actions_confidence: float = 0.7,
) -> Tuple[Optional[LearningArtifact], int]:
    """Validate and repair a learning artifact.

    1. ``supporting_item_ids`` must be a subset of ``cluster_item_ids``.
       Remove invalid IDs and track repair count.
       Return ``None`` if ALL supporting_item_ids are invalid.
    2. ``counterexamples`` must be a subset of ``cluster_item_ids``.
       Remove invalid, track repairs.
    3. If ``confidence >= min_actions_confidence`` and no ``recommended_actions``:
       demote confidence to ``min_actions_confidence - 0.01``.

    Returns:
        ``(repaired_artifact_or_None, num_repairs)``
    """
    repairs = 0

    # Validate supporting_item_ids
    valid_supporting = [
        sid for sid in learning.supporting_item_ids if sid in cluster_item_ids
    ]
    repairs += len(learning.supporting_item_ids) - len(valid_supporting)

    if not valid_supporting:
        return None, repairs

    # Validate counterexamples
    valid_counter = [cid for cid in learning.counterexamples if cid in cluster_item_ids]
    repairs += len(learning.counterexamples) - len(valid_counter)

    # Quality gate: high confidence requires recommended_actions
    confidence = learning.confidence
    actions = list(learning.recommended_actions)
    if confidence >= min_actions_confidence and not actions:
        confidence = round(min_actions_confidence - 0.01, 2)
        repairs += 1

    return (
        LearningArtifact(
            title=learning.title,
            content=learning.content,
            tags=list(learning.tags),
            confidence=confidence,
            supporting_item_ids=valid_supporting,
            recommended_actions=actions,
            counterexamples=valid_counter,
            scope=learning.scope,
            when_not_to_apply=learning.when_not_to_apply,
        ),
        repairs,
    )


RecurrenceKeyFn = Callable[[EvidenceItem], str]


def default_recurrence_key(item: EvidenceItem) -> str:
    return item.id


def check_recurrence(
    supporting_item_ids: List[str],
    evidence: Dict[str, EvidenceItem],
    threshold: int,
    key_fn: RecurrenceKeyFn = default_recurrence_key,
) -> bool:
    """Count UNIQUE recurrence keys, not just unique item IDs.

    For conversations: ``key_fn = lambda item: item.source_ref or item.id``
    ensures 2 chunks from the same conversation count as 1 occurrence.
    """
    unique_keys = {
        key_fn(evidence[iid]) for iid in supporting_item_ids if iid in evidence
    }
    return len(unique_keys) >= threshold


def deterministic_sample(
    items: Dict[str, EvidenceItem],
    max_items: int,
    seed: Optional[int] = None,
) -> Dict[str, EvidenceItem]:
    """Hash-based deterministic selection.

    Default (no seed): sort by ``sha256(key)`` and take first ``max_items``.
    With seed: ``Random(seed).sample()`` for reproducible random sampling.
    """
    if len(items) <= max_items:
        return dict(items)

    keys = list(items.keys())

    if seed is not None:
        rng = random.Random(seed)
        selected = rng.sample(keys, max_items)
    else:
        selected = sorted(keys, key=lambda k: hashlib.sha256(k.encode()).hexdigest())[
            :max_items
        ]

    return {k: items[k] for k in selected}


def default_tag_normalizer(tags: List[str]) -> List[str]:
    """Deterministic tag normalization.

    - Lowercase, strip whitespace
    - Replace spaces with underscores
    - Drop empty and tags > DEFAULT_MAX_TAG_LENGTH chars
    - Deduplicate (preserving order)
    - Cap at DEFAULT_MAX_TAGS tags
    """
    seen: set = set()
    result: List[str] = []
    for tag in tags:
        normalized = tag.lower().strip().replace(' ', '_')
        if not normalized or len(normalized) > DEFAULT_MAX_TAG_LENGTH:
            continue
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
        if len(result) >= DEFAULT_MAX_TAGS:
            break
    return result
