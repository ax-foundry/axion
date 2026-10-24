"""Sampler hook and the recurrence-balanced sampler."""

import pytest

from axion.caliber.pattern_discovery._utils import (
    deterministic_sample,
    recurrence_balanced_sampler,
)
from axion.caliber.pattern_discovery.pipeline import EvidencePipeline
from tests.caliber.conftest import (
    ScriptedClusterer,
    ScriptedWriter,
    make_grouped_evidence,
    parent_key,
)


def _parents(items):
    return {parent_key(i) for i in items.values()}


class TestRecurrenceBalancedSampler:
    def test_under_budget_is_identity(self):
        evidence = make_grouped_evidence(2, 2)
        out = recurrence_balanced_sampler(parent_key)(evidence, 10, None)
        assert out == evidence
        assert out is not evidence

    def test_every_parent_kept_when_keys_fit(self):
        evidence = make_grouped_evidence(5, 6)  # 30 items, 5 parents
        out = recurrence_balanced_sampler(parent_key)(evidence, 12, None)
        assert len(out) == 12
        assert _parents(out) == {f'parent_{p}' for p in range(5)}
        per_parent = sorted(
            sum(1 for i in out.values() if parent_key(i) == k) for k in _parents(out)
        )
        assert per_parent == [2, 2, 2, 3, 3]

    def test_overflow_keeps_first_budget_keys_one_each(self):
        evidence = make_grouped_evidence(10, 3)  # 30 items, 10 parents
        out = recurrence_balanced_sampler(parent_key)(evidence, 4, None)
        assert len(out) == 4
        assert len(_parents(out)) == 4

    def test_deterministic_without_seed_and_with_seed(self):
        evidence = make_grouped_evidence(6, 4)
        sampler = recurrence_balanced_sampler(parent_key)
        assert sampler(evidence, 7, None) == sampler(evidence, 7, None)
        assert sampler(evidence, 7, 3) == sampler(evidence, 7, 3)

    def test_seed_changes_selection_but_not_balance(self):
        evidence = make_grouped_evidence(4, 8)
        sampler = recurrence_balanced_sampler(parent_key)
        a = sampler(evidence, 8, 1)
        b = sampler(evidence, 8, 2)
        assert _parents(a) == _parents(b) == {f'parent_{p}' for p in range(4)}
        assert set(a) != set(b)

    def test_preserves_input_order(self):
        evidence = make_grouped_evidence(3, 5)
        out = recurrence_balanced_sampler(parent_key)(evidence, 6, None)
        positions = [list(evidence).index(k) for k in out]
        assert positions == sorted(positions)

    def test_default_key_matches_item_level_sampling_size(self):
        evidence = make_grouped_evidence(1, 20)
        out = recurrence_balanced_sampler()(evidence, 5, None)
        assert len(out) == 5

    def test_uniform_sampler_can_lose_parents(self):
        """The failure mode the balanced sampler exists for."""
        evidence = make_grouped_evidence(12, 10)  # 120 items
        lost_any = False
        for seed in range(20):
            kept = _parents(deterministic_sample(evidence, 12, seed))
            if len(kept) < 12:
                lost_any = True
                break
        assert lost_any
        balanced = recurrence_balanced_sampler(parent_key)(evidence, 12, 0)
        assert len(_parents(balanced)) == 12


class TestPipelineSamplerHook:
    def test_sampler_imports_from_caliber(self):
        from axion.caliber import recurrence_balanced_sampler as exported

        assert exported is recurrence_balanced_sampler

    @pytest.mark.asyncio
    async def test_default_sampler_unchanged(self):
        evidence = make_grouped_evidence(4, 5)
        clusterer = ScriptedClusterer([])
        pipeline = EvidencePipeline(
            clusterer=clusterer, writer=ScriptedWriter(), max_items=7, seed=11
        )
        await pipeline.run(evidence)
        assert clusterer.calls[0] == deterministic_sample(evidence, 7, 11)

    @pytest.mark.asyncio
    async def test_balanced_sampler_reports_key_coverage(self):
        evidence = make_grouped_evidence(8, 10)  # 80 items, 8 parents
        clusterer = ScriptedClusterer([])
        pipeline = EvidencePipeline(
            clusterer=clusterer,
            writer=ScriptedWriter(),
            max_items=16,
            recurrence_key_fn=parent_key,
            sampler=recurrence_balanced_sampler(parent_key),
        )
        result = await pipeline.run(evidence)
        assert result.input_count == 80
        assert result.sampled_out == 64
        assert len(clusterer.calls[0]) == 16
        assert result.coverage['distinct_keys_input'] == 8
        assert result.coverage['distinct_keys_sampled'] == 8
        assert result.coverage['sampled_out'] == 64

    @pytest.mark.asyncio
    async def test_custom_sampler_is_called_with_seed(self):
        evidence = make_grouped_evidence(2, 5)
        seen = {}

        def sampler(items, max_items, seed):
            seen['args'] = (len(items), max_items, seed)
            return dict(list(items.items())[:max_items])

        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([]),
            writer=ScriptedWriter(),
            max_items=3,
            seed=5,
            sampler=sampler,
        )
        result = await pipeline.run(evidence)
        assert seen['args'] == (10, 3, 5)
        assert result.sampled_out == 7

    @pytest.mark.asyncio
    async def test_key_stats_without_sampling(self):
        evidence = make_grouped_evidence(3, 2)
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([]),
            writer=ScriptedWriter(),
            recurrence_key_fn=parent_key,
        )
        result = await pipeline.run(evidence)
        assert result.sampled_out == 0
        assert result.coverage['distinct_keys_input'] == 3
        assert result.coverage['distinct_keys_sampled'] == 3
