import hashlib
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from axion.caliber.pattern_discovery.models import (
    EvidenceItem,
    LearningArtifact,
    Provenance,
)

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
    """
    Format metadata into a compact header string.

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
    """
    Aggregate metadata across cluster members into a summary string.

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
    """
    Validate and repair a learning artifact.

    1. ``supporting_item_ids`` must be a subset of ``cluster_item_ids`` and
       contain no repeats. Remove invalid/repeated IDs and track repair count.
       Return ``None`` if ALL supporting_item_ids are invalid.
    2. ``counterexamples`` must be a subset of ``cluster_item_ids``, contain no
       repeats, and not overlap ``supporting_item_ids``. Remove invalid, track
       repairs.
    3. If ``confidence >= min_actions_confidence`` and no ``recommended_actions``:
       demote confidence to ``min_actions_confidence - 0.01``.

    Returns:
        ``(repaired_artifact_or_None, num_repairs)``
    """
    repairs = 0

    # Validate supporting_item_ids: in-cluster, de-duplicated, order preserved
    valid_supporting: List[str] = []
    seen_supporting: Set[str] = set()
    for sid in learning.supporting_item_ids:
        if sid in cluster_item_ids and sid not in seen_supporting:
            valid_supporting.append(sid)
            seen_supporting.add(sid)
    repairs += len(learning.supporting_item_ids) - len(valid_supporting)

    if not valid_supporting:
        return None, repairs

    # Validate counterexamples: in-cluster, de-duplicated, and never also cited as
    # support (an id cannot both support and contradict the same learning)
    valid_counter: List[str] = []
    seen_counter: Set[str] = set()
    for cid in learning.counterexamples:
        if (
            cid in cluster_item_ids
            and cid not in seen_supporting
            and cid not in seen_counter
        ):
            valid_counter.append(cid)
            seen_counter.add(cid)
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
    """
    Count UNIQUE recurrence keys, not just unique item IDs.

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
    """
    Hash-based deterministic selection.

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
    """
    Deterministic tag normalization.

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


def group_ids_by_key(
    item_ids: List[str],
    evidence: Dict[str, EvidenceItem],
    key_fn: RecurrenceKeyFn = default_recurrence_key,
) -> Dict[str, List[str]]:
    """Group item ids by recurrence key, preserving first-seen order of both."""
    groups: Dict[str, List[str]] = {}
    for iid in item_ids:
        item = evidence.get(iid)
        if item is None:
            continue
        groups.setdefault(key_fn(item), []).append(iid)
    return groups


def count_distinct_keys(
    item_ids: List[str],
    evidence: Dict[str, EvidenceItem],
    key_fn: RecurrenceKeyFn = default_recurrence_key,
) -> int:
    """Number of distinct recurrence keys among ``item_ids`` present in ``evidence``."""
    return len(group_ids_by_key(item_ids, evidence, key_fn))


def round_robin_by_key(
    item_ids: List[str],
    evidence: Dict[str, EvidenceItem],
    limit: int,
    key_fn: RecurrenceKeyFn = default_recurrence_key,
) -> List[str]:
    """
    Pick up to ``limit`` ids spread across recurrence keys, deterministically.

    Takes the first id of every key (in first-seen key order), then the second
    of every key, and so on. Ids missing from ``evidence`` are skipped. With the
    default key (item id) this is simply ``item_ids[:limit]``.
    """
    if limit <= 0:
        return []
    groups = list(group_ids_by_key(item_ids, evidence, key_fn).values())
    picked: List[str] = []
    depth = 0
    while len(picked) < limit:
        progressed = False
        for group in groups:
            if depth < len(group):
                picked.append(group[depth])
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
        depth += 1
    return picked


@dataclass
class ClusterRepairs:
    """Counts of fixes applied to raw clustering output before it is trusted."""

    unknown_ids: int = 0
    duplicate_ids: int = 0
    cross_pattern_duplicates: int = 0
    uncategorized_unknown: int = 0
    uncategorized_assigned: int = 0
    uncategorized_duplicates: int = 0

    @property
    def total(self) -> int:
        return (
            self.unknown_ids
            + self.duplicate_ids
            + self.cross_pattern_duplicates
            + self.uncategorized_unknown
            + self.uncategorized_assigned
            + self.uncategorized_duplicates
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            'unknown_ids': self.unknown_ids,
            'duplicate_ids': self.duplicate_ids,
            'cross_pattern_duplicates': self.cross_pattern_duplicates,
            'uncategorized_unknown': self.uncategorized_unknown,
            'uncategorized_assigned': self.uncategorized_assigned,
            'uncategorized_duplicates': self.uncategorized_duplicates,
            'total': self.total,
        }


def repair_cluster_assignments(
    pattern_ids: List[List[str]],
    uncategorized: List[str],
    known_ids: Set[str],
    disjoint: bool = False,
) -> Tuple[List[List[str]], List[str], ClusterRepairs]:
    """
    Validate raw model cluster assignments against the ids that were submitted.

    - Ids not in ``known_ids`` are dropped from patterns and ``uncategorized``.
    - Repeats within one pattern are collapsed.
    - With ``disjoint=True`` an id keeps only its first pattern; later patterns
      lose it (``cross_pattern_duplicates``). With ``disjoint=False`` (default)
      an id may legitimately appear in several patterns.
    - ``uncategorized`` loses repeats and any id that is also assigned.

    Returns the repaired pattern id lists (same order and length as the input,
    possibly empty), the repaired ``uncategorized`` list, and the repair counts.
    """
    repairs = ClusterRepairs()
    assigned: Set[str] = set()
    repaired_patterns: List[List[str]] = []
    for ids in pattern_ids:
        kept: List[str] = []
        seen: Set[str] = set()
        for rid in ids:
            if rid not in known_ids:
                repairs.unknown_ids += 1
                continue
            if rid in seen:
                repairs.duplicate_ids += 1
                continue
            if disjoint and rid in assigned:
                repairs.cross_pattern_duplicates += 1
                continue
            seen.add(rid)
            kept.append(rid)
        assigned.update(kept)
        repaired_patterns.append(kept)

    repaired_uncat: List[str] = []
    seen_uncat: Set[str] = set()
    for rid in uncategorized:
        if rid not in known_ids:
            repairs.uncategorized_unknown += 1
            continue
        if rid in assigned:
            repairs.uncategorized_assigned += 1
            continue
        if rid in seen_uncat:
            repairs.uncategorized_duplicates += 1
            continue
        seen_uncat.add(rid)
        repaired_uncat.append(rid)

    return repaired_patterns, repaired_uncat, repairs


def build_coverage(
    submitted_ids: List[str],
    assigned_ids: Set[str],
    uncategorized_ids: List[str],
    filtered_small_cluster_ids: Set[str],
    ineligible: int = 0,
    sampled_out: int = 0,
) -> Dict[str, Any]:
    """
    Account for every submitted id in exactly one bucket.

    ``ratio`` is model coverage: the share of *submitted* ids the model placed
    somewhere (a pattern or ``uncategorized``). Items the caller filtered out
    before or after the call (``ineligible``, ``sampled_out``,
    ``filtered_small_cluster``) are reported but never counted against the model.
    """
    submitted = set(submitted_ids)
    placed = assigned_ids | set(uncategorized_ids) | filtered_small_cluster_ids
    omitted = sorted(submitted - placed)
    n_submitted = len(submitted)
    covered = n_submitted - len(omitted)
    return {
        'submitted': n_submitted,
        'assigned': len(assigned_ids & submitted),
        'uncategorized': len(set(uncategorized_ids) & submitted),
        'filtered_small_cluster': len(filtered_small_cluster_ids & submitted),
        'omitted_by_model': len(omitted),
        'omitted_ids': omitted,
        'ineligible': ineligible,
        'sampled_out': sampled_out,
        'ratio': (covered / n_submitted) if n_submitted else 1.0,
    }


def normalize_evidence(
    evidence: Any,
) -> Tuple[Dict[str, EvidenceItem], int]:
    """
    Turn a sequence or dict of ``EvidenceItem`` into an id-keyed dict.

    A dict is validated (key must equal ``item.id``) and returned as-is.
    In a sequence, two items with the same id must be identical: exact
    repeats are collapsed and counted; conflicting repeats raise
    ``ValueError`` rather than one silently overwriting the other.

    Returns ``(evidence_dict, exact_duplicates_dropped)``.
    """
    if isinstance(evidence, dict):
        for key, item in evidence.items():
            if key != item.id:
                raise ValueError(f'Dict key {key!r} does not match item.id {item.id!r}')
        return evidence, 0

    result: Dict[str, EvidenceItem] = {}
    dropped = 0
    for item in evidence:
        existing = result.get(item.id)
        if existing is None:
            result[item.id] = item
        elif existing == item:
            dropped += 1
        else:
            raise ValueError(
                f'Conflicting evidence items share id {item.id!r}; '
                'give each distinct item its own id'
            )
    return result, dropped


ScoreFn = Callable[[LearningArtifact, Provenance, List[EvidenceItem]], float]
"""Ranks a learning. Receives the learning, its provenance, and the evidence
items it cites. Higher sorts first."""


def score_by_confidence(
    learning: LearningArtifact,
    provenance: Provenance,
    supporting_items: List[EvidenceItem],
) -> float:
    """The default ordering: the writer's own confidence."""
    return learning.confidence


def score_by_distinct_keys(
    learning: LearningArtifact,
    provenance: Provenance,
    supporting_items: List[EvidenceItem],
) -> float:
    """
    Rank by independent recurrence, confidence as tie-break.

    ``distinct_key_count`` strictly dominates: one more distinct key outranks
    any confidence, and confidence (in ``[0, 1]``) only orders learnings with
    the same number of distinct keys.
    """
    return 2 * provenance.distinct_key_count + learning.confidence


SamplerFn = Callable[
    [Dict[str, EvidenceItem], int, Optional[int]], Dict[str, EvidenceItem]
]
"""Reduces evidence to at most ``max_items``: ``(items, max_items, seed) -> items``.
Must be deterministic for a given input and seed."""


def _stable_order(keys: List[str], seed: Optional[int]) -> List[str]:
    """Hash order without a seed, seeded shuffle with one. Mirrors deterministic_sample."""
    if seed is not None:
        ordered = list(keys)
        random.Random(seed).shuffle(ordered)
        return ordered
    return sorted(keys, key=lambda k: hashlib.sha256(k.encode()).hexdigest())


def recurrence_balanced_sampler(
    key_fn: RecurrenceKeyFn = default_recurrence_key,
) -> SamplerFn:
    """
    Build a sampler that keeps as many recurrence keys represented as it can.

    Items are grouped by ``key_fn``. Keys are visited in a stable order (hash
    order, or a seeded shuffle when ``seed`` is given) and one item per key is
    taken per round, round after round, until ``max_items`` is reached. Within a
    key, items are taken in the same stable order.

    Guarantees, best effort and deterministic:

    - When distinct keys <= ``max_items``, every key keeps at least one item.
    - When distinct keys > ``max_items``, the first ``max_items`` keys in the
      stable order keep one item each and the remaining keys are dropped whole.
    - Within the budget, items are spread as evenly across keys as the counts allow.

    Keeping a key does not guarantee keeping the particular item that would
    establish a pattern; it guarantees the parent is still in the corpus.
    The returned dict preserves the input's item order.
    """

    def sample(
        items: Dict[str, EvidenceItem],
        max_items: int,
        seed: Optional[int] = None,
    ) -> Dict[str, EvidenceItem]:
        if len(items) <= max_items:
            return dict(items)
        if max_items <= 0:
            return {}
        groups = group_ids_by_key(list(items.keys()), items, key_fn)
        ordered_keys = _stable_order(list(groups.keys()), seed)
        ordered_groups = [_stable_order(groups[k], seed) for k in ordered_keys]
        selected: Set[str] = set()
        depth = 0
        while len(selected) < max_items:
            progressed = False
            for group in ordered_groups:
                if depth < len(group):
                    selected.add(group[depth])
                    progressed = True
                    if len(selected) >= max_items:
                        break
            if not progressed:
                break
            depth += 1
        return {k: v for k, v in items.items() if k in selected}

    return sample
