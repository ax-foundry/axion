"""Accounting for clusters that distill without error but yield no learning."""

import pytest

from axion.caliber.pattern_discovery.handlers import LearningArtifactOutput
from axion.caliber.pattern_discovery.pipeline import EvidencePipeline
from tests.caliber.conftest import (
    ScriptedClusterer,
    ScriptedWriter,
    make_grouped_evidence,
    parent_key,
)


class _EmptyWriter:
    async def distill(self, cluster, domain_context):
        return []


class _TwoCandidateWriter:
    """One candidate cites nothing from the cluster, the other a single id."""

    async def distill(self, cluster, domain_context):
        return [
            LearningArtifactOutput(
                title='Outside',
                content='c',
                tags=['t'],
                confidence=0.9,
                supporting_item_ids=['not_in_cluster'],
            ),
            LearningArtifactOutput(
                title='Single',
                content='c',
                tags=['t'],
                confidence=0.8,
                supporting_item_ids=[cluster.item_ids[0]],
                recommended_actions=['act'],
            ),
        ]


def _pipeline(writer, clusters, **kw):
    return EvidencePipeline(clusterer=ScriptedClusterer(clusters), writer=writer, **kw)


@pytest.mark.asyncio
async def test_writer_returning_nothing_is_recorded_per_cluster():
    evidence = make_grouped_evidence(3, 2)
    ids = list(evidence)
    result = await _pipeline(_EmptyWriter(), [ids[:3], ids[3:]]).run(evidence)
    assert result.learnings == []
    assert result.failed_clusters == []
    assert [c.category for c in result.clusters_without_learnings] == [
        'Cluster 0',
        'Cluster 1',
    ]
    first = result.clusters_without_learnings[0]
    assert (first.item_count, first.candidates) == (3, 0)
    assert first.dropped_invalid == first.dropped_below_recurrence == 0
    assert result.learnings_dropped_invalid == 0
    assert result.learnings_dropped_below_recurrence == 0


@pytest.mark.asyncio
async def test_dropped_candidates_are_counted_by_reason():
    evidence = make_grouped_evidence(2, 3)
    ids = list(evidence)
    result = await _pipeline(_TwoCandidateWriter(), [ids], recurrence_threshold=2).run(
        evidence
    )
    assert result.learnings == []
    (empty,) = result.clusters_without_learnings
    assert empty.candidates == 2
    assert empty.dropped_invalid == 1
    assert empty.dropped_below_recurrence == 1
    assert result.learnings_dropped_invalid == 1
    assert result.learnings_dropped_below_recurrence == 1
    # The out-of-cluster id was removed as a repair before the candidate dropped.
    assert result.validation_repairs == 1


@pytest.mark.asyncio
async def test_productive_cluster_is_not_listed():
    evidence = make_grouped_evidence(2, 3)
    ids = list(evidence)
    result = await _pipeline(ScriptedWriter(), [ids]).run(evidence)
    assert len(result.learnings) == 1
    assert result.clusters_without_learnings == []


@pytest.mark.asyncio
async def test_scored_path_counts_drops_too():
    evidence = make_grouped_evidence(2, 3)
    ids = list(evidence)
    result = await _pipeline(
        _TwoCandidateWriter(),
        [ids],
        recurrence_threshold=2,
        score_fn=lambda learning, provenance, items: learning.confidence,
    ).run(evidence)
    assert result.learnings_dropped_invalid == 1
    assert result.learnings_dropped_below_recurrence == 1
    assert len(result.clusters_without_learnings) == 1


class TestMaxExamplesPerCluster:
    @pytest.mark.asyncio
    async def test_default_cap_is_five(self):
        evidence = make_grouped_evidence(4, 3)
        writer = ScriptedWriter()
        await _pipeline(writer, [list(evidence)]).run(evidence)
        assert len(writer.clusters[0].examples) == 5

    @pytest.mark.asyncio
    async def test_cap_bounds_cited_keys(self):
        evidence = make_grouped_evidence(8, 2)  # 16 items, 8 parents
        writer = ScriptedWriter()
        result = await _pipeline(
            writer,
            [list(evidence)],
            recurrence_key_fn=parent_key,
            max_examples_per_cluster=8,
        ).run(evidence)
        shown = writer.clusters[0].examples
        assert len(shown) == 8
        assert len({parent_key(evidence[e.item_id]) for e in shown}) == 8
        assert result.records[0].provenance.distinct_key_count == 8

    def test_cap_must_be_positive(self):
        with pytest.raises(ValueError, match='max_examples_per_cluster'):
            _pipeline(ScriptedWriter(), [], max_examples_per_cluster=0)

    def test_exported_from_caliber(self):
        from axion.caliber import ClusterWithoutLearnings
        from axion.caliber.pattern_discovery.models import (
            ClusterWithoutLearnings as inner,
        )

        assert ClusterWithoutLearnings is inner
