"""Determinism + calibration knobs for pattern discovery.

Covers the corpus-scaled clustering category range, the per-range clustering-handler
cache, the ``cluster_llm`` split on ``EvidencePipeline`` (clustering on one model,
distillation on another), and the evidence-anchored confidence rubric in the
distillation instruction. The seeded-UMAP determinism fix is pinned in
``test_pattern_discovery.py::TestBertopicSmallCorpusResilience``.
"""

import pytest

from axion.caliber.pattern_discovery.discovery import PatternDiscovery
from axion.caliber.pattern_discovery.handlers import (
    DEFAULT_DISTILLATION_INSTRUCTION,
    DEFAULT_EVIDENCE_CLUSTERING_INSTRUCTION,
    default_evidence_clustering_instruction,
    evidence_category_range,
)
from axion.caliber.pattern_discovery.pipeline import EvidencePipeline


class _FakeLLM:
    """Minimal runnable satisfying the handler LLM validation."""

    model = 'fake-model'

    async def acomplete(self, *args, **kwargs):  # pragma: no cover - never invoked
        raise AssertionError('fake LLM must not be called in these tests')


class TestEvidenceCategoryRange:
    """The clustering category range scales with corpus size (was hardcoded 3-6)."""

    @pytest.mark.parametrize(
        ('n_items', 'expected'),
        [
            (5, '2-4'),  # tiny corpus: floor lo at 2, keep span >= 2
            (10, '2-4'),
            (30, '2-6'),
            (54, '4-10'),  # the Package-digest corpus size
            (100, '8-15'),
            (170, '14-15'),  # the BOP-digest corpus size; hi capped at 15
            (1000, '15-15'),  # lo would exceed the cap without min()
        ],
    )
    def test_range_values(self, n_items, expected):
        assert evidence_category_range(n_items) == expected

    def test_lo_never_exceeds_hi(self):
        for n in range(2, 500):
            lo, hi = map(int, evidence_category_range(n).split('-'))
            assert 2 <= lo <= hi <= 15, f'n={n} produced {lo}-{hi}'

    def test_rendered_instruction_contains_range(self):
        instruction = default_evidence_clustering_instruction(100)
        assert 'group them into 8-15 meaningful categories' in instruction

    def test_backcompat_constant_keeps_fixed_range(self):
        assert 'group them into 3-6 meaningful categories' in (
            DEFAULT_EVIDENCE_CLUSTERING_INSTRUCTION
        )


class TestScaledClusteringHandler:
    """PatternDiscovery renders the corpus-scaled instruction into its handler."""

    def test_handler_gets_scaled_instruction(self):
        discovery = PatternDiscovery(llm=_FakeLLM())
        handler = discovery._get_evidence_clustering_handler(n_items=100)
        assert 'group them into 8-15 meaningful categories' in handler.instruction

    def test_handlers_cached_per_range(self):
        discovery = PatternDiscovery(llm=_FakeLLM())
        h_small = discovery._get_evidence_clustering_handler(n_items=10)
        h_small_again = discovery._get_evidence_clustering_handler(n_items=11)
        h_large = discovery._get_evidence_clustering_handler(n_items=100)
        # 10 and 11 items render the same range -> same cached handler.
        assert h_small is h_small_again
        assert h_small is not h_large

    def test_custom_instruction_never_rescaled(self):
        custom = 'Cluster these however you like.'
        discovery = PatternDiscovery(llm=_FakeLLM(), instruction=custom)
        handler = discovery._get_evidence_clustering_handler(n_items=100)
        assert handler.instruction == custom


class TestClusterLLMSplit:
    """EvidencePipeline can run clustering and distillation on different models."""

    def test_cluster_llm_routes_to_discovery_only(self):
        cluster_llm = _FakeLLM()
        distill_llm = _FakeLLM()
        pipeline = EvidencePipeline(llm=distill_llm, cluster_llm=cluster_llm)
        assert pipeline._clusterer._discovery._llm is cluster_llm
        assert pipeline._writer._handler.llm is distill_llm

    def test_default_shares_main_llm(self):
        llm = _FakeLLM()
        pipeline = EvidencePipeline(llm=llm)
        assert pipeline._clusterer._discovery._llm is llm
        assert pipeline._writer._handler.llm is llm

    def test_cluster_model_name_routes_to_discovery_only(self):
        pipeline = EvidencePipeline(
            llm=_FakeLLM(),
            cluster_llm=_FakeLLM(),
            model_name='gpt-5.2',
            cluster_model_name='gpt-4.1',
        )
        assert pipeline._clusterer._discovery._model_name == 'gpt-4.1'
        assert pipeline._writer._handler.model_name == 'gpt-5.2'


class TestConfidenceRubric:
    """The distillation instruction anchors confidence to evidence counts."""

    def test_rubric_present(self):
        assert '0.9+: >= 10 consistent supporting items' in (
            DEFAULT_DISTILLATION_INSTRUCTION
        )
        assert '0.75-0.89: 5-9 supporting items' in DEFAULT_DISTILLATION_INSTRUCTION
        assert '< 0.6: weak or mixed support' in DEFAULT_DISTILLATION_INSTRUCTION
