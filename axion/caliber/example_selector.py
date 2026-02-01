import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from axion.caliber.pattern_discovery import DiscoveredPattern


class SelectionStrategy(str, Enum):
    """Strategies for selecting few-shot examples."""

    BALANCED = 'balanced'  # 50/50 accept/reject, random
    MISALIGNMENT_GUIDED = 'misalignment_guided'  # Prioritize FP/FN cases
    PATTERN_AWARE = 'pattern_aware'  # Cover discovered patterns


@dataclass
class SelectionResult:
    """Result of example selection."""

    examples: List[Dict[str, Any]]
    strategy_used: SelectionStrategy
    metadata: Dict[str, Any]  # Stats about selection (e.g., FP/FN counts)


class ExampleSelector:
    """
    Selects few-shot examples for LLM-as-judge calibration.

    Example:
        >>> selector = ExampleSelector()
        >>>
        >>> # Simple balanced selection
        >>> result = selector.select(records, annotations, count=6)
        >>>
        >>> # Misalignment-guided (requires eval results)
        >>> result = selector.select(
        ...     records, annotations, count=6,
        ...     strategy=SelectionStrategy.MISALIGNMENT_GUIDED,
        ...     eval_results=results
        ... )
        >>>
        >>> # Pattern-aware (requires Pattern Discovery results)
        >>> result = selector.select(
        ...     records, annotations, count=6,
        ...     strategy=SelectionStrategy.PATTERN_AWARE,
        ...     patterns=discovered_patterns
        ... )
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize selector.

        Args:
            seed: Random seed for reproducibility
        """
        self._rng = random.Random(seed)

    def select(
        self,
        records: List[Dict[str, Any]],
        annotations: Dict[str, int],
        count: int = 6,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        eval_results: Optional[List[Dict[str, Any]]] = None,
        patterns: Optional[List[DiscoveredPattern]] = None,
    ) -> SelectionResult:
        """
        Select few-shot examples.

        Args:
            records: List of records with 'id', 'query', 'actual_output', etc.
            annotations: Dict mapping record_id -> human score (0 or 1)
            count: Number of examples to select
            strategy: Selection strategy to use
            eval_results: Evaluation results (required for MISALIGNMENT_GUIDED)
            patterns: Discovered patterns (required for PATTERN_AWARE)

        Returns:
            SelectionResult with selected examples and metadata
        """
        if strategy == SelectionStrategy.BALANCED:
            return self._select_balanced(records, annotations, count)
        elif strategy == SelectionStrategy.MISALIGNMENT_GUIDED:
            if eval_results is None:
                raise ValueError(
                    'eval_results required for MISALIGNMENT_GUIDED strategy'
                )
            return self._select_misalignment_guided(
                records, annotations, count, eval_results
            )
        elif strategy == SelectionStrategy.PATTERN_AWARE:
            if patterns is None:
                raise ValueError('patterns required for PATTERN_AWARE strategy')
            return self._select_pattern_aware(records, annotations, count, patterns)
        else:
            raise ValueError(f'Unknown strategy: {strategy}')

    def _select_balanced(
        self,
        records: List[Dict[str, Any]],
        annotations: Dict[str, int],
        count: int,
    ) -> SelectionResult:
        """Select with 50/50 accept/reject balance."""
        accepts = [r for r in records if annotations.get(self._get_id(r)) == 1]
        rejects = [r for r in records if annotations.get(self._get_id(r)) == 0]

        half = count // 2
        selected = []

        # Sample from each, handling case where one side has fewer
        n_accepts = min(half, len(accepts))
        n_rejects = min(count - n_accepts, len(rejects))

        if accepts:
            selected.extend(self._rng.sample(accepts, min(n_accepts, len(accepts))))
        if rejects:
            selected.extend(self._rng.sample(rejects, min(n_rejects, len(rejects))))

        # If still need more, take from whichever has extras
        remaining = count - len(selected)
        if remaining > 0:
            pool = [r for r in records if r not in selected]
            if pool:
                selected.extend(self._rng.sample(pool, min(remaining, len(pool))))

        return SelectionResult(
            examples=selected,
            strategy_used=SelectionStrategy.BALANCED,
            metadata={'accepts': n_accepts, 'rejects': n_rejects},
        )

    def _select_misalignment_guided(
        self,
        records: List[Dict[str, Any]],
        annotations: Dict[str, int],
        count: int,
        eval_results: List[Dict[str, Any]],
    ) -> SelectionResult:
        """Prioritize cases where LLM judge disagreed with human."""
        # Build lookup for eval results
        eval_by_id = {self._get_id(r): r for r in eval_results}

        # Find misaligned cases
        false_positives = []  # LLM=1, Human=0
        false_negatives = []  # LLM=0, Human=1
        aligned = []

        for record in records:
            rid = self._get_id(record)
            human = annotations.get(rid, 0)
            eval_r = eval_by_id.get(rid, {})
            llm = eval_r.get(
                'llm_score', eval_r.get('score', human)
            )  # Fallback to human if no eval

            if llm == 1 and human == 0:
                false_positives.append(record)
            elif llm == 0 and human == 1:
                false_negatives.append(record)
            else:
                aligned.append(record)

        selected = []

        # Prioritize misaligned cases (balance FP/FN)
        fp_count = min(count // 3, len(false_positives))
        fn_count = min(count // 3, len(false_negatives))

        if false_positives:
            selected.extend(self._rng.sample(false_positives, fp_count))
        if false_negatives:
            selected.extend(self._rng.sample(false_negatives, fn_count))

        # Fill remaining with balanced aligned examples
        remaining = count - len(selected)
        if remaining > 0 and aligned:
            # Use balanced selection for the rest
            aligned_result = self._select_balanced(aligned, annotations, remaining)
            selected.extend(aligned_result.examples)

        return SelectionResult(
            examples=selected[:count],
            strategy_used=SelectionStrategy.MISALIGNMENT_GUIDED,
            metadata={
                'false_positives_selected': fp_count,
                'false_negatives_selected': fn_count,
                'total_fp_available': len(false_positives),
                'total_fn_available': len(false_negatives),
            },
        )

    def _select_pattern_aware(
        self,
        records: List[Dict[str, Any]],
        annotations: Dict[str, int],
        count: int,
        patterns: List[DiscoveredPattern],
    ) -> SelectionResult:
        """Sample from discovered patterns to ensure coverage."""
        records_by_id = {self._get_id(r): r for r in records}
        selected = []
        selected_ids = set()
        patterns_covered = []

        # Take 1 example from each pattern (up to count)
        for pattern in patterns[:count]:
            available = [
                rid
                for rid in pattern.record_ids
                if rid in records_by_id and rid not in selected_ids
            ]
            if available:
                rid = self._rng.choice(available)
                selected.append(records_by_id[rid])
                selected_ids.add(rid)
                patterns_covered.append(pattern.category)

        # Fill remaining with balanced selection
        remaining = count - len(selected)
        if remaining > 0:
            pool = [r for r in records if self._get_id(r) not in selected_ids]
            if pool:
                balanced = self._select_balanced(pool, annotations, remaining)
                selected.extend(balanced.examples)

        return SelectionResult(
            examples=selected[:count],
            strategy_used=SelectionStrategy.PATTERN_AWARE,
            metadata={
                'patterns_covered': patterns_covered,
                'total_patterns': len(patterns),
            },
        )

    @staticmethod
    def _get_id(record: Dict[str, Any]) -> str:
        """Extract record ID, handling multiple field names."""
        return str(record.get('id') or record.get('record_id') or '')
