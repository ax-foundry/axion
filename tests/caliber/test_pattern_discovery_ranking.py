"""Per-learning records, recurrence-aware provenance, scoring, failure policy."""

import json

import pytest

from axion.caliber.pattern_discovery._utils import (
    score_by_confidence,
    score_by_distinct_keys,
)
from axion.caliber.pattern_discovery.handlers import LearningArtifactOutput
from axion.caliber.pattern_discovery.models import FailedCluster, LearningRecord
from axion.caliber.pattern_discovery.pipeline import EvidencePipeline
from axion.caliber.pattern_discovery.plugins import (
    InMemoryDeduper,
    InMemorySink,
    JsonlSink,
)
from tests.caliber.conftest import (
    ScriptedClusterer,
    ScriptedWriter,
    make_grouped_evidence,
    parent_key,
)


def _learning(ids, confidence, title='L'):
    return LearningArtifactOutput(
        title=title,
        content='c',
        tags=['t'],
        confidence=confidence,
        supporting_item_ids=list(ids),
        recommended_actions=['act'],
    )


class ListWriter:
    """Returns a scripted list of learnings for every cluster."""

    def __init__(self, learnings):
        self._learnings = learnings

    async def distill(self, cluster, domain_context):
        return list(self._learnings)


class FailingWriter:
    def __init__(self, fail_categories=None):
        self._fail = fail_categories

    async def distill(self, cluster, domain_context):
        if self._fail is None or cluster.category in self._fail:
            raise RuntimeError(f'boom {cluster.category}')
        return [_learning(cluster.item_ids[:2], 0.8, title=cluster.category)]


class FlakySink:
    """Fails on the titles it is told to; records the rest."""

    def __init__(self, fail_titles):
        self._fail = set(fail_titles)
        self.written = []

    async def write(self, artifact, provenance):
        if artifact.title in self._fail:
            raise IOError(f'cannot write {artifact.title}')
        self.written.append(artifact.title)
        return f'id_{len(self.written)}'


# --------------------------------------------------------------------------- #
# Records and provenance
# --------------------------------------------------------------------------- #


class TestRecords:
    @pytest.mark.asyncio
    async def test_records_present_without_sink(self):
        evidence = make_grouped_evidence(3, 2)
        cluster = list(evidence)
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([cluster]),
            writer=ScriptedWriter(pick=lambda c: ['p0_i0', 'p0_i1', 'p1_i0']),
            recurrence_key_fn=parent_key,
        )
        result = await pipeline.run(evidence)
        assert len(result.records) == 1
        rec = result.records[0]
        assert isinstance(rec, LearningRecord)
        prov = rec.provenance
        assert prov.distinct_keys == ['parent_0', 'parent_1']
        assert prov.distinct_key_count == 2
        assert prov.supporting_count == 3
        assert prov.cited_support_count == 3
        assert prov.cluster_size == 6
        assert prov.cluster_category == 'Cluster 0'
        assert rec.cluster is not None and rec.cluster.record_ids == cluster
        assert rec.sink_id is None and rec.sink_error is None
        assert rec.score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_records_align_with_learnings_and_sink_ids(self):
        evidence = make_grouped_evidence(2, 3)
        ids = list(evidence)
        sink = InMemorySink()
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids[:3], ids[3:]]),
            writer=ScriptedWriter(),
            sink=sink,
        )
        result = await pipeline.run(evidence)
        assert [r.learning for r in result.records] == result.learnings
        assert [r.sink_id for r in result.records] == result.sink_ids
        assert len(result.sink_ids) == 2

    @pytest.mark.asyncio
    async def test_legacy_provenance_fields_unchanged(self):
        evidence = make_grouped_evidence(1, 3)
        ids = list(evidence)
        sink = InMemorySink()
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids]),
            writer=ScriptedWriter(pick=lambda c: c.item_ids[:2]),
            sink=sink,
        )
        await pipeline.run(evidence)
        prov = list(sink.artifacts.values())[0]['provenance']
        assert prov.source_ref == 'parent_0'
        assert prov.supporting_count == 2
        assert prov.metadata['all_source_refs'] == ['parent_0', 'parent_0']
        assert prov.clustering_method == 'llm'

    @pytest.mark.asyncio
    async def test_jsonl_sink_writes_new_provenance_fields(self, tmp_path):
        evidence = make_grouped_evidence(2, 1)
        ids = list(evidence)
        path = tmp_path / 'out.jsonl'
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids]),
            writer=ScriptedWriter(pick=lambda c: c.item_ids),
            sink=JsonlSink(str(path)),
            recurrence_key_fn=parent_key,
        )
        await pipeline.run(evidence)
        record = json.loads(path.read_text().splitlines()[0])
        assert record['provenance']['distinct_keys'] == ['parent_0', 'parent_1']
        assert record['provenance']['distinct_key_count'] == 2
        assert record['provenance']['cluster_size'] == 2


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


class TestScoring:
    @pytest.mark.asyncio
    async def test_default_order_is_confidence(self):
        evidence = make_grouped_evidence(3, 2)
        ids = list(evidence)
        writer = ListWriter(
            [_learning(ids[:2], 0.6, 'low'), _learning(ids[2:4], 0.9, 'high')]
        )
        pipeline = EvidencePipeline(clusterer=ScriptedClusterer([ids]), writer=writer)
        result = await pipeline.run(evidence)
        assert [la.title for la in result.learnings] == ['high', 'low']
        assert [r.score for r in result.records] == [0.9, 0.6]

    @pytest.mark.asyncio
    async def test_score_by_distinct_keys_reorders(self):
        evidence = make_grouped_evidence(3, 3)
        ids = list(evidence)
        one_parent = ['p0_i0', 'p0_i1', 'p0_i2']
        three_parents = ['p0_i0', 'p1_i0', 'p2_i0']
        writer = ListWriter(
            [
                _learning(one_parent, 0.95, 'confident-one-parent'),
                _learning(three_parents, 0.6, 'spread'),
            ]
        )
        by_confidence = EvidencePipeline(
            clusterer=ScriptedClusterer([ids]),
            writer=writer,
            recurrence_threshold=1,
            recurrence_key_fn=parent_key,
        )
        by_keys = EvidencePipeline(
            clusterer=ScriptedClusterer([ids]),
            writer=writer,
            recurrence_threshold=1,
            recurrence_key_fn=parent_key,
            score_fn=score_by_distinct_keys,
        )
        assert [la.title for la in (await by_confidence.run(evidence)).learnings] == [
            'confident-one-parent',
            'spread',
        ]
        result = await by_keys.run(evidence)
        assert [la.title for la in result.learnings] == [
            'spread',
            'confident-one-parent',
        ]
        assert result.records[0].score == pytest.approx(3.6)
        assert result.records[1].score == pytest.approx(1.95)

    @pytest.mark.asyncio
    async def test_score_fn_drives_candidate_selection(self):
        """A score_fn sees every candidate, not only the top-confidence ones."""
        evidence = make_grouped_evidence(4, 2)
        ids = list(evidence)
        spread = ['p0_i0', 'p1_i0', 'p2_i0', 'p3_i0']
        narrow = ['p0_i0', 'p0_i1']
        writer = ListWriter(
            [
                _learning(narrow, 0.9, 'n1'),
                _learning(narrow, 0.85, 'n2'),
                _learning(narrow, 0.8, 'n3'),
                _learning(spread, 0.5, 'spread'),
            ]
        )
        common = dict(
            clusterer=ScriptedClusterer([ids]),
            writer=writer,
            recurrence_threshold=1,
            recurrence_key_fn=parent_key,
            max_learnings_per_cluster=3,
        )
        default = await EvidencePipeline(**common).run(evidence)
        assert 'spread' not in [la.title for la in default.learnings]
        scored = await EvidencePipeline(**common, score_fn=score_by_distinct_keys).run(
            evidence
        )
        assert [la.title for la in scored.learnings][0] == 'spread'
        assert len(scored.learnings) == 3

    @pytest.mark.asyncio
    async def test_custom_score_fn_reads_metadata(self):
        evidence = make_grouped_evidence(2, 2)
        for i, item in enumerate(evidence.values()):
            item.metadata['weight'] = 10 if item.source_ref == 'parent_1' else 1
        ids = list(evidence)
        writer = ListWriter(
            [
                _learning(['p0_i0', 'p0_i1'], 0.9, 'light'),
                _learning(['p1_i0', 'p1_i1'], 0.5, 'heavy'),
            ]
        )

        def weight(learning, provenance, items):
            return sum(it.metadata['weight'] for it in items)

        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids]), writer=writer, score_fn=weight
        )
        result = await pipeline.run(evidence)
        assert [la.title for la in result.learnings] == ['heavy', 'light']
        assert [r.score for r in result.records] == [20.0, 2.0]

    @pytest.mark.asyncio
    async def test_dedupe_keeps_best_scored_twin(self):
        evidence = make_grouped_evidence(3, 1)
        ids = list(evidence)
        writer = ListWriter(
            [
                _learning(ids[:1], 0.9, 'Same Title'),
                _learning(ids, 0.5, 'same title'),
            ]
        )
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids]),
            writer=writer,
            recurrence_threshold=1,
            recurrence_key_fn=parent_key,
            score_fn=score_by_distinct_keys,
            deduper=InMemoryDeduper(),
        )
        result = await pipeline.run(evidence)
        assert result.deduplicated_count == 1
        assert result.records[0].provenance.distinct_key_count == 3

    def test_builtin_scorers(self):
        from axion.caliber.pattern_discovery.models import (
            LearningArtifact,
            Provenance,
        )

        la = LearningArtifact('t', 'c', [], 0.7, ['a'])
        prov = Provenance(distinct_key_count=4)
        assert score_by_confidence(la, prov, []) == 0.7
        assert score_by_distinct_keys(la, prov, []) == pytest.approx(4.7)


# --------------------------------------------------------------------------- #
# Failure policy
# --------------------------------------------------------------------------- #


class TestFailurePolicy:
    @pytest.mark.asyncio
    async def test_failed_clusters_reported_and_others_continue(self):
        evidence = make_grouped_evidence(2, 3)
        ids = list(evidence)
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids[:3], ids[3:]]),
            writer=FailingWriter(fail_categories={'Cluster 0'}),
        )
        result = await pipeline.run(evidence)
        assert len(result.learnings) == 1
        assert result.failed_clusters == [
            FailedCluster(
                category='Cluster 0', error='RuntimeError: boom Cluster 0', item_count=3
            )
        ]

    @pytest.mark.asyncio
    async def test_all_failed_is_quiet_by_default(self):
        evidence = make_grouped_evidence(1, 4)
        ids = list(evidence)
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids[:2], ids[2:]]), writer=FailingWriter()
        )
        result = await pipeline.run(evidence)
        assert result.learnings == []
        assert len(result.failed_clusters) == 2

    @pytest.mark.asyncio
    async def test_fail_on_all_errors_raises(self):
        evidence = make_grouped_evidence(1, 4)
        ids = list(evidence)
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids[:2], ids[2:]]),
            writer=FailingWriter(),
            fail_on_all_errors=True,
        )
        with pytest.raises(RuntimeError, match='all 2 clusters'):
            await pipeline.run(evidence)

    @pytest.mark.asyncio
    async def test_fail_on_all_errors_tolerates_partial(self):
        evidence = make_grouped_evidence(2, 3)
        ids = list(evidence)
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids[:3], ids[3:]]),
            writer=FailingWriter(fail_categories={'Cluster 0'}),
            fail_on_all_errors=True,
        )
        result = await pipeline.run(evidence)
        assert len(result.learnings) == 1

    @pytest.mark.asyncio
    async def test_sink_failure_raises_by_default(self):
        evidence = make_grouped_evidence(2, 3)
        ids = list(evidence)
        sink = FlakySink(fail_titles={'Learning: Cluster 0'})
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids[:3], ids[3:]]),
            writer=ScriptedWriter(),
            sink=sink,
        )
        with pytest.raises(IOError):
            await pipeline.run(evidence)

    @pytest.mark.asyncio
    async def test_sink_continue_records_outcomes(self):
        evidence = make_grouped_evidence(2, 3)
        ids = list(evidence)
        sink = FlakySink(fail_titles={'Learning: Cluster 0'})
        pipeline = EvidencePipeline(
            clusterer=ScriptedClusterer([ids[:3], ids[3:]]),
            writer=ScriptedWriter(),
            sink=sink,
            sink_on_error='continue',
        )
        result = await pipeline.run(evidence)
        assert sink.written == ['Learning: Cluster 1']
        assert result.sink_ids == ['id_1']
        outcomes = {r.learning.title: (r.sink_id, r.sink_error) for r in result.records}
        assert outcomes['Learning: Cluster 1'] == ('id_1', None)
        assert outcomes['Learning: Cluster 0'] == (
            None,
            'OSError: cannot write Learning: Cluster 0',
        )

    def test_sink_on_error_validated(self):
        with pytest.raises(ValueError):
            EvidencePipeline(
                clusterer=ScriptedClusterer([]),
                writer=ScriptedWriter(),
                sink_on_error='ignore',  # type: ignore[arg-type]
            )
