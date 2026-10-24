"""Evidence integrity, coverage accounting, and distiller attribution."""

import pytest

from axion.caliber.pattern_discovery._utils import (
    build_coverage,
    normalize_evidence,
    repair_cluster_assignments,
    round_robin_by_key,
    validate_learning,
)
from axion.caliber.pattern_discovery.discovery import (
    LowCoverageError,
    PatternDiscovery,
)
from axion.caliber.pattern_discovery.handlers import LearningArtifactOutput
from axion.caliber.pattern_discovery.models import ClusteringMethod, EvidenceItem
from axion.caliber.pattern_discovery.pipeline import EvidencePipeline
from tests.caliber.conftest import (
    ScriptedClusterer,
    ScriptedClusteringHandler,
    ScriptedWriter,
    clustering_output,
    make_grouped_evidence,
    parent_key,
)


class _Wrap:
    """Expose a ``PatternDiscovery`` through the ``EvidenceClusterer`` protocol."""

    def __init__(self, discovery: PatternDiscovery) -> None:
        self._discovery = discovery

    async def cluster(self, evidence, method):
        return await self._discovery.discover_from_evidence(evidence, method)


def _discovery(output, **kwargs) -> PatternDiscovery:
    """A PatternDiscovery whose LLM clustering call is scripted."""
    d = PatternDiscovery(llm=object(), **kwargs)
    handler = ScriptedClusteringHandler(output)
    d._get_evidence_clustering_handler = lambda n_items=None: handler  # type: ignore
    d._scripted_handler = handler  # type: ignore[attr-defined]
    return d


# --------------------------------------------------------------------------- #
# Input integrity
# --------------------------------------------------------------------------- #


class TestNormalizeEvidence:
    def test_exact_duplicates_collapse_and_count(self):
        item = EvidenceItem(id='a', text='t')
        result, dropped = normalize_evidence([item, EvidenceItem(id='a', text='t')])
        assert list(result) == ['a']
        assert dropped == 1

    def test_conflicting_duplicates_raise(self):
        with pytest.raises(ValueError, match='Conflicting evidence items'):
            normalize_evidence(
                [EvidenceItem(id='a', text='t1'), EvidenceItem(id='a', text='t2')]
            )

    def test_dict_passthrough(self):
        d = {'a': EvidenceItem(id='a', text='t')}
        result, dropped = normalize_evidence(d)
        assert result is d
        assert dropped == 0

    def test_to_evidence_dict_rejects_conflicts(self):
        with pytest.raises(ValueError):
            PatternDiscovery._to_evidence_dict(
                [EvidenceItem(id='a', text='t1'), EvidenceItem(id='a', text='t2')]
            )

    @pytest.mark.asyncio
    async def test_pipeline_reports_duplicates_dropped(self):
        evidence = list(make_grouped_evidence(2, 2).values())
        evidence.append(evidence[0])
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([[e.id for e in evidence[:4]]]),
            writer=ScriptedWriter(),
        )
        result = await pipeline.run(evidence)
        assert result.duplicates_dropped == 1
        assert result.input_count == 4


class TestValidateLearningIntegrity:
    def _learning(self, supporting, counter):
        return LearningArtifactOutput(
            title='t',
            content='c',
            tags=[],
            confidence=0.5,
            supporting_item_ids=supporting,
            counterexamples=counter,
        )

    def test_supporting_ids_deduplicated(self):
        artifact, repairs = validate_learning(
            self._learning(['a', 'a', 'b'], []), {'a', 'b'}
        )
        assert artifact.supporting_item_ids == ['a', 'b']
        assert repairs == 1

    def test_counterexample_cannot_also_support(self):
        artifact, repairs = validate_learning(
            self._learning(['a', 'b'], ['b', 'c', 'c']), {'a', 'b', 'c'}
        )
        assert artifact.counterexamples == ['c']
        assert repairs == 2


# --------------------------------------------------------------------------- #
# Cluster output validation
# --------------------------------------------------------------------------- #


class TestRepairClusterAssignments:
    def test_unknown_and_repeated_ids_dropped(self):
        patterns, uncat, repairs = repair_cluster_assignments(
            [['a', 'ghost', 'a', 'b']], ['c', 'ghost2', 'c', 'a'], {'a', 'b', 'c'}
        )
        assert patterns == [['a', 'b']]
        assert uncat == ['c']
        assert repairs.unknown_ids == 1
        assert repairs.duplicate_ids == 1
        assert repairs.uncategorized_unknown == 1
        assert repairs.uncategorized_duplicates == 1
        assert repairs.uncategorized_assigned == 1

    def test_overlap_allowed_by_default(self):
        patterns, _, repairs = repair_cluster_assignments(
            [['a', 'b'], ['b', 'c']], [], {'a', 'b', 'c'}
        )
        assert patterns == [['a', 'b'], ['b', 'c']]
        assert repairs.cross_pattern_duplicates == 0

    def test_disjoint_keeps_first_assignment(self):
        patterns, _, repairs = repair_cluster_assignments(
            [['a', 'b'], ['b', 'c']], [], {'a', 'b', 'c'}, disjoint=True
        )
        assert patterns == [['a', 'b'], ['c']]
        assert repairs.cross_pattern_duplicates == 1


class TestDiscoveryOutputValidation:
    @pytest.mark.asyncio
    async def test_hallucinated_ids_do_not_inflate_count(self):
        evidence = make_grouped_evidence(1, 3)
        ids = list(evidence)
        d = _discovery(clustering_output([[*ids, 'ghost1', 'ghost2', ids[0]]]))
        result = await d.discover_from_evidence(evidence)
        assert len(result.patterns) == 1
        assert result.patterns[0].count == 3
        assert result.patterns[0].record_ids == ids
        assert result.metadata['cluster_repairs']['unknown_ids'] == 2
        assert result.metadata['cluster_repairs']['duplicate_ids'] == 1

    @pytest.mark.asyncio
    async def test_cluster_that_shrinks_below_min_size_is_dropped(self):
        evidence = make_grouped_evidence(1, 3)
        ids = list(evidence)
        d = _discovery(clustering_output([[ids[0], 'ghost'], ids[1:]]))
        result = await d.discover_from_evidence(evidence)
        assert [p.record_ids for p in result.patterns] == [ids[1:]]
        assert result.metadata['coverage']['filtered_small_cluster'] == 1

    @pytest.mark.asyncio
    async def test_disjoint_clusters_option(self):
        evidence = make_grouped_evidence(1, 4)
        ids = list(evidence)
        output = clustering_output([ids[:3], ids[1:]])
        strict = await _discovery(
            output, disjoint_clusters=True
        ).discover_from_evidence(evidence)
        # Second pattern keeps only ids[3], falls below min_category_size, and
        # that one id is accounted for as filtered rather than omitted.
        assert [p.record_ids for p in strict.patterns] == [ids[:3]]
        assert strict.metadata['cluster_repairs']['cross_pattern_duplicates'] == 2
        assert strict.metadata['coverage']['filtered_small_cluster'] == 1
        assert strict.metadata['coverage']['omitted_by_model'] == 0
        loose = await _discovery(output).discover_from_evidence(evidence)
        assert sum(p.count for p in loose.patterns) == 6

    @pytest.mark.asyncio
    async def test_uncategorized_validated(self):
        evidence = make_grouped_evidence(1, 3)
        ids = list(evidence)
        d = _discovery(clustering_output([ids[:2]], [ids[2], ids[0], 'ghost']))
        result = await d.discover_from_evidence(evidence)
        assert result.uncategorized == [ids[2]]


# --------------------------------------------------------------------------- #
# Coverage accounting
# --------------------------------------------------------------------------- #


class TestCoverage:
    def test_build_coverage_buckets(self):
        cov = build_coverage(
            submitted_ids=['a', 'b', 'c', 'd', 'e'],
            assigned_ids={'a', 'b'},
            uncategorized_ids=['c'],
            filtered_small_cluster_ids={'d'},
            ineligible=2,
            sampled_out=3,
        )
        assert cov['assigned'] == 2
        assert cov['uncategorized'] == 1
        assert cov['filtered_small_cluster'] == 1
        assert cov['omitted_by_model'] == 1
        assert cov['omitted_ids'] == ['e']
        assert cov['ineligible'] == 2
        assert cov['sampled_out'] == 3
        assert cov['ratio'] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_omitted_distinct_from_filtered(self):
        """Items the model never returned are not confused with filtered ones."""
        evidence = make_grouped_evidence(1, 6)
        ids = list(evidence)
        # ids[0:3] assigned; ids[3] in a too-small cluster; ids[4] uncategorized;
        # ids[5] never mentioned.
        d = _discovery(clustering_output([ids[:3], [ids[3]]], [ids[4]]))
        result = await d.discover_from_evidence(evidence)
        cov = result.metadata['coverage']
        assert cov['submitted'] == 6
        assert cov['assigned'] == 3
        assert cov['filtered_small_cluster'] == 1
        assert cov['uncategorized'] == 1
        assert cov['omitted_by_model'] == 1
        assert cov['omitted_ids'] == [ids[5]]
        assert cov['ratio'] == pytest.approx(5 / 6)

    @pytest.mark.asyncio
    async def test_ineligible_not_counted_against_model(self):
        evidence = make_grouped_evidence(1, 3)
        ids = list(evidence)
        evidence['blank'] = EvidenceItem(id='blank', text='')
        d = _discovery(clustering_output([ids]))
        result = await d.discover_from_evidence(evidence)
        cov = result.metadata['coverage']
        assert cov['ineligible'] == 1
        assert cov['submitted'] == 3
        assert cov['ratio'] == 1.0

    @pytest.mark.asyncio
    async def test_min_coverage_warns_and_flags(self, caplog):
        evidence = make_grouped_evidence(1, 4)
        ids = list(evidence)
        d = _discovery(clustering_output([ids[:2]]), min_coverage=0.9)
        result = await d.discover_from_evidence(evidence)
        assert result.metadata['coverage']['below_min'] is True
        assert any('min_coverage' in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_min_coverage_raises_when_opted_in(self):
        evidence = make_grouped_evidence(1, 4)
        ids = list(evidence)
        d = _discovery(
            clustering_output([ids[:2]]), min_coverage=0.9, raise_on_low_coverage=True
        )
        with pytest.raises(LowCoverageError) as exc:
            await d.discover_from_evidence(evidence)
        assert exc.value.coverage['omitted_by_model'] == 2

    def test_min_coverage_validated(self):
        with pytest.raises(ValueError):
            PatternDiscovery(llm=object(), min_coverage=1.5)

    @pytest.mark.asyncio
    async def test_clustering_excerpt_cap(self):
        evidence = make_grouped_evidence(1, 2)
        ids = list(evidence)
        d = _discovery(clustering_output([ids]), clustering_excerpt_chars=8)
        await d.discover_from_evidence(evidence)
        sent = d._scripted_handler.inputs[0].items  # type: ignore[attr-defined]
        assert all(len(n.text) == 8 for n in sent)

    @pytest.mark.asyncio
    async def test_full_text_sent_by_default(self):
        evidence = make_grouped_evidence(1, 2)
        ids = list(evidence)
        d = _discovery(clustering_output([ids]))
        await d.discover_from_evidence(evidence)
        sent = d._scripted_handler.inputs[0].items  # type: ignore[attr-defined]
        assert [n.text for n in sent] == [evidence[i].text for i in ids]

    @pytest.mark.asyncio
    async def test_pipeline_rolls_up_accounting(self):
        evidence = make_grouped_evidence(4, 5)  # 20 items
        ids = list(evidence)
        clusterer = ScriptedClusterer(
            [ids[:8], ids[8:10]],
            metadata={'coverage': {'submitted': 12, 'omitted_by_model': 2}},
        )
        pipeline = EvidencePipeline(
            clusterer=clusterer,
            writer=ScriptedWriter(),
            max_items=12,
            recurrence_threshold=3,
            recurrence_key_fn=parent_key,
        )
        result = await pipeline.run(evidence)
        assert result.input_count == 20
        assert result.sampled_out == 8
        assert len(clusterer.calls[0]) == 12
        assert result.coverage['input_count'] == 20
        assert result.coverage['sampled_out'] == 8
        assert result.coverage['omitted_by_model'] == 2

    @pytest.mark.asyncio
    async def test_pipeline_buckets_stay_exclusive_after_prefilter(self):
        """Items of a pre-filtered cluster leave ``assigned``; ratio is untouched."""
        evidence = make_grouped_evidence(1, 2)  # two items, one parent
        ids = list(evidence)
        clustering = _discovery(clustering_output([ids]))
        pipeline = EvidencePipeline(
            clusterer=_Wrap(clustering),
            writer=ScriptedWriter(),
            recurrence_threshold=2,
            recurrence_key_fn=parent_key,
        )
        result = await pipeline.run(evidence)
        cov = result.coverage
        assert result.learnings == []
        assert cov['input_count'] == 2
        assert cov['assigned'] == 0
        assert cov['filtered_below_threshold'] == 2
        assert cov['ratio'] == 1.0
        assert (
            cov['assigned']
            + cov['uncategorized']
            + cov['filtered_small_cluster']
            + cov['filtered_below_threshold']
            + cov['omitted_by_model']
            + cov['ineligible']
            == cov['input_count']
        )

    @pytest.mark.asyncio
    async def test_pipeline_derives_buckets_for_bare_clusterer(self):
        """A clusterer that reports no coverage still yields every documented key."""
        evidence = make_grouped_evidence(2, 3)
        ids = list(evidence)
        clusterer = ScriptedClusterer([ids[:4]], uncategorized=[ids[4]])
        result = await EvidencePipeline(
            clusterer=clusterer, writer=ScriptedWriter()
        ).run(evidence)
        cov = result.coverage
        assert cov['submitted'] == 6
        assert cov['assigned'] == 4
        assert cov['uncategorized'] == 1
        assert cov['omitted_by_model'] == 1
        assert cov['omitted_ids'] == [ids[5]]
        assert cov['ineligible'] == 0
        assert cov['ratio'] == pytest.approx(5 / 6)

    @pytest.mark.asyncio
    async def test_blank_only_input_still_reports_coverage(self):
        evidence = {'blank': EvidenceItem(id='blank', text='')}
        d = _discovery(clustering_output([]))
        result = await d.discover_from_evidence(evidence)
        cov = result.metadata['coverage']
        assert cov['submitted'] == 0
        assert cov['ineligible'] == 1
        assert cov['assigned'] == 0
        assert cov['omitted_by_model'] == 0
        assert cov['ratio'] == 1.0
        pipeline = EvidencePipeline(clusterer=_Wrap(d), writer=ScriptedWriter())
        presult = await pipeline.run(evidence)
        assert presult.coverage['ineligible'] == 1
        assert presult.coverage['ratio'] == 1.0


# --------------------------------------------------------------------------- #
# Recurrence-key pre-filter
# --------------------------------------------------------------------------- #


class TestKeyAwarePreFilter:
    @pytest.mark.asyncio
    async def test_single_parent_cluster_not_distilled(self):
        evidence = make_grouped_evidence(3, 4)
        one_parent = [eid for eid in evidence if eid.startswith('p0_')]
        two_parents = ['p1_i0', 'p1_i1', 'p2_i0']
        writer = ScriptedWriter()
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([one_parent, two_parents]),
            writer=writer,
            recurrence_threshold=2,
            recurrence_key_fn=parent_key,
        )
        result = await pipeline.run(evidence)
        assert result.filtered_count == 1
        assert [c.item_ids for c in writer.clusters] == [two_parents]
        assert result.coverage['filtered_below_threshold'] == 4

    @pytest.mark.asyncio
    async def test_default_key_unchanged(self):
        """With the default key the pre-filter behaves exactly as before."""
        evidence = make_grouped_evidence(1, 4)
        ids = list(evidence)
        writer = ScriptedWriter()
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids[:1], ids[1:]]),
            writer=writer,
            recurrence_threshold=2,
        )
        result = await pipeline.run(evidence)
        assert result.filtered_count == 1
        assert len(writer.clusters) == 1


# --------------------------------------------------------------------------- #
# Distiller attribution
# --------------------------------------------------------------------------- #


class TestRoundRobinByKey:
    def test_spreads_across_keys_deterministically(self):
        evidence = make_grouped_evidence(3, 4)
        ids = list(evidence)  # p0 x4, p1 x4, p2 x4
        picked = round_robin_by_key(ids, evidence, 5, parent_key)
        assert picked == ['p0_i0', 'p1_i0', 'p2_i0', 'p0_i1', 'p1_i1']
        assert round_robin_by_key(ids, evidence, 5, parent_key) == picked

    def test_default_key_is_prefix(self):
        evidence = make_grouped_evidence(1, 4)
        ids = list(evidence)
        assert round_robin_by_key(ids, evidence, 2) == ids[:2]

    def test_skips_unknown_ids_and_short_groups(self):
        evidence = make_grouped_evidence(2, 1)
        picked = round_robin_by_key(
            ['p0_i0', 'ghost', 'p1_i0'], evidence, 10, parent_key
        )
        assert picked == ['p0_i0', 'p1_i0']


class TestDistillerSeesIds:
    @pytest.mark.asyncio
    async def test_examples_carry_ids_and_span_parents(self):
        evidence = make_grouped_evidence(4, 5)
        cluster = [eid for eid in evidence if eid.startswith(('p0_', 'p1_', 'p2_'))]
        writer = ScriptedWriter()
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([cluster]),
            writer=writer,
            recurrence_key_fn=parent_key,
        )
        await pipeline.run(evidence)
        sent = writer.clusters[0]
        assert len(sent.examples) == 5
        assert [e.text for e in sent.examples] == sent.example_texts
        assert {parent_key(evidence[e.item_id]) for e in sent.examples} == {
            'parent_0',
            'parent_1',
            'parent_2',
        }
        assert sent.distinct_source_count == 3
        assert sent.item_ids == cluster

    @pytest.mark.asyncio
    async def test_default_key_examples_are_first_five(self):
        evidence = make_grouped_evidence(1, 8)
        ids = list(evidence)
        writer = ScriptedWriter()
        pipeline = EvidencePipeline(clusterer=ScriptedClusterer([ids]), writer=writer)
        await pipeline.run(evidence)
        sent = writer.clusters[0]
        assert [e.item_id for e in sent.examples] == ids[:5]
        assert sent.distinct_source_count == 8

    @pytest.mark.asyncio
    async def test_more_items_from_one_parent_do_not_add_recurrence(self):
        """Grouped-evidence invariant: recurrence counts parents, not items."""
        few = make_grouped_evidence(1, 2)
        many = make_grouped_evidence(1, 10)

        async def run(evidence):
            ids = list(evidence)
            pipeline = EvidencePipeline(
                clusterer=ScriptedClusterer([ids]),
                writer=ScriptedWriter(pick=lambda c: c.item_ids),
                recurrence_threshold=2,
                recurrence_key_fn=parent_key,
            )
            return await pipeline.run(evidence)

        assert (await run(few)).learnings == []
        assert (await run(many)).learnings == []

    @pytest.mark.asyncio
    async def test_second_parent_adds_recurrence(self):
        evidence = make_grouped_evidence(2, 1)
        ids = list(evidence)
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids]),
            writer=ScriptedWriter(pick=lambda c: c.item_ids),
            recurrence_threshold=2,
            recurrence_key_fn=parent_key,
        )
        result = await pipeline.run(evidence)
        assert len(result.learnings) == 1


class TestBertopicCoverage:
    @pytest.mark.asyncio
    async def test_too_few_documents_path_unchanged(self):
        pytest.importorskip('bertopic')
        evidence = make_grouped_evidence(1, 2)
        d = PatternDiscovery(llm=object())
        result = await d.discover_from_evidence(evidence, ClusteringMethod.BERTOPIC)
        assert result.metadata['reason'] == 'too_few_documents'
