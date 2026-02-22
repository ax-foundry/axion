from unittest.mock import AsyncMock

import pytest

from axion.caliber.pattern_discovery._utils import MetadataConfig
from axion.caliber.pattern_discovery.handlers import (
    ClusterForDistillation,
    LearningArtifactOutput,
)
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    EvidenceItem,
    PatternDiscoveryResult,
)
from axion.caliber.pattern_discovery.pipeline import EvidencePipeline
from axion.caliber.pattern_discovery.plugins import (
    InMemoryDeduper,
    InMemorySink,
    NoopSanitizer,
)


def _make_evidence(n=5):
    return {
        f'e{i}': EvidenceItem(
            id=f'e{i}',
            text=f'Evidence text number {i}',
            metadata={'step': 'checkout' if i % 2 == 0 else 'billing'},
            source_ref=f'conv_{i}',
        )
        for i in range(n)
    }


def _make_clustering_result(evidence_ids, n_clusters=2):
    per_cluster = len(evidence_ids) // n_clusters
    patterns = []
    for c in range(n_clusters):
        start = c * per_cluster
        end = start + per_cluster
        rids = evidence_ids[start:end]
        if rids:
            patterns.append(
                DiscoveredPattern(
                    category=f'Cluster {c}',
                    description=f'Description {c}',
                    count=len(rids),
                    record_ids=rids,
                    examples=[f'example {c}'],
                )
            )

    return PatternDiscoveryResult(
        patterns=patterns,
        uncategorized=[],
        total_analyzed=len(evidence_ids),
        method=ClusteringMethod.LLM,
    )


def _make_learning_output(item_ids, confidence=0.8):
    return LearningArtifactOutput(
        title='Test Learning',
        content='Synthesized content',
        tags=['tag1', 'Tag 2'],
        confidence=confidence,
        supporting_item_ids=item_ids,
        recommended_actions=['Do something'],
        counterexamples=[],
    )


class TestEvidencePipeline:
    """Tests for the full evidence pipeline with mocked handlers."""

    @pytest.mark.asyncio
    async def test_basic_pipeline_run(self):
        evidence = _make_evidence(6)
        eids = list(evidence.keys())

        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(
            return_value=_make_clustering_result(eids, 2)
        )

        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            side_effect=lambda cluster, ctx: [
                _make_learning_output(cluster.item_ids[:2])
            ]
        )

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=2,
        )
        result = await pipeline.run(evidence)

        assert result.clustering_result.total_analyzed == 6
        assert len(result.learnings) >= 1

    @pytest.mark.asyncio
    async def test_pre_filter_below_recurrence(self):
        """Clusters with fewer items than recurrence_threshold should be dropped."""
        evidence = _make_evidence(5)
        eids = list(evidence.keys())

        # One cluster with 1 item, one with 4
        patterns = [
            DiscoveredPattern(
                category='Small',
                description='D',
                count=1,
                record_ids=[eids[0]],
            ),
            DiscoveredPattern(
                category='Big',
                description='D',
                count=4,
                record_ids=eids[1:],
            ),
        ]
        cr = PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=[],
            total_analyzed=5,
            method=ClusteringMethod.LLM,
        )

        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=cr)

        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            side_effect=lambda cluster, ctx: [
                _make_learning_output(cluster.item_ids[:2])
            ]
        )

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=2,
        )
        result = await pipeline.run(evidence)

        # Small cluster should be pre-filtered
        assert result.filtered_count == 1
        # Only the big cluster should have been distilled
        assert mock_writer.distill.call_count == 1

    @pytest.mark.asyncio
    async def test_validation_repairs_hallucinated_ids(self):
        """Hallucinated IDs in supporting_item_ids should be repaired."""
        evidence = _make_evidence(4)
        eids = list(evidence.keys())

        cr = _make_clustering_result(eids, 1)
        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=cr)

        # Return a learning with a hallucinated ID
        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            return_value=[
                _make_learning_output(['e0', 'e1', 'hallucinated_id'])
            ]
        )

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=2,
        )
        result = await pipeline.run(evidence)

        # The learning should survive with hallucinated ID removed
        for la in result.learnings:
            assert 'hallucinated_id' not in la.supporting_item_ids

        # validation_repairs should reflect the removed hallucinated IDs
        assert result.validation_repairs >= 1

    @pytest.mark.asyncio
    async def test_validation_drops_all_invalid_ids(self):
        """Learning where ALL IDs are invalid should be dropped entirely."""
        evidence = _make_evidence(4)
        eids = list(evidence.keys())

        cr = _make_clustering_result(eids, 1)
        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=cr)

        # All IDs are hallucinated
        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            return_value=[_make_learning_output(['bad1', 'bad2'])]
        )

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=1,
        )
        result = await pipeline.run(evidence)

        assert len(result.learnings) == 0
        # All IDs were invalid → repairs should be counted even for dropped learnings
        assert result.validation_repairs >= 2

    @pytest.mark.asyncio
    async def test_max_learnings_per_cluster(self):
        """More than MAX_LEARNINGS_PER_CLUSTER should be truncated."""
        evidence = _make_evidence(4)
        eids = list(evidence.keys())

        cr = _make_clustering_result(eids, 1)
        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=cr)

        # Return 5 learnings (above default MAX=3)
        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            return_value=[
                _make_learning_output(eids[:2], confidence=0.9),
                _make_learning_output(eids[:2], confidence=0.8),
                _make_learning_output(eids[:2], confidence=0.7),
                _make_learning_output(eids[:2], confidence=0.6),
                _make_learning_output(eids[:2], confidence=0.5),
            ]
        )

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=2,
        )
        result = await pipeline.run(evidence)

        # Should be capped at MAX_LEARNINGS_PER_CLUSTER (3)
        assert len(result.learnings) <= EvidencePipeline.MAX_LEARNINGS_PER_CLUSTER

    @pytest.mark.asyncio
    async def test_recurrence_check_with_custom_key(self):
        """Custom recurrence key should properly deduplicate sources."""
        # All items from same source
        evidence = {
            f'e{i}': EvidenceItem(
                id=f'e{i}',
                text=f'text {i}',
                source_ref='same_conv',
            )
            for i in range(4)
        }
        eids = list(evidence.keys())

        cr = _make_clustering_result(eids, 1)
        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=cr)

        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            return_value=[_make_learning_output(eids[:3])]
        )

        # Custom key: group by source_ref
        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=2,
            recurrence_key_fn=lambda item: item.source_ref or item.id,
        )
        result = await pipeline.run(evidence)

        # All items have same source → only 1 unique key → below threshold 2
        assert len(result.learnings) == 0

    @pytest.mark.asyncio
    async def test_tag_normalization(self):
        """Tags should be normalized by default."""
        evidence = _make_evidence(4)
        eids = list(evidence.keys())

        cr = _make_clustering_result(eids, 1)
        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=cr)

        learning = LearningArtifactOutput(
            title='Test',
            content='C',
            tags=['Missing Context', '  Bug Fix  ', 'missing context'],
            confidence=0.8,
            supporting_item_ids=eids[:2],
            recommended_actions=['Fix it'],
        )
        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(return_value=[learning])

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=2,
        )
        result = await pipeline.run(evidence)

        if result.learnings:
            tags = result.learnings[0].tags
            assert 'missing_context' in tags
            assert 'bug_fix' in tags
            # Deduplication: missing_context should appear only once
            assert tags.count('missing_context') == 1

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Deduper should remove duplicate learnings."""
        evidence = _make_evidence(6)
        eids = list(evidence.keys())

        # Two clusters returning same-titled learnings
        patterns = [
            DiscoveredPattern(
                category=f'Cluster {i}',
                description='D',
                count=3,
                record_ids=eids[i * 3 : (i + 1) * 3],
            )
            for i in range(2)
        ]
        cr = PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=[],
            total_analyzed=6,
            method=ClusteringMethod.LLM,
        )

        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=cr)

        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            side_effect=lambda cluster, ctx: [
                _make_learning_output(cluster.item_ids[:2])
            ]
        )

        deduper = InMemoryDeduper()
        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=2,
            deduper=deduper,
        )
        result = await pipeline.run(evidence)

        # Both clusters produce "Test Learning" → second one deduplicated
        assert result.deduplicated_count == 1
        assert len(result.learnings) == 1

    @pytest.mark.asyncio
    async def test_sink_writes(self):
        """Sink should receive all surviving learnings with populated provenance."""
        evidence = _make_evidence(4)
        eids = list(evidence.keys())

        cr = _make_clustering_result(eids, 1)
        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=cr)

        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            return_value=[_make_learning_output(eids[:2])]
        )

        sink = InMemorySink()
        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=2,
            sink=sink,
        )
        result = await pipeline.run(evidence)

        assert len(result.sink_ids) == len(result.learnings)
        assert len(sink.artifacts) == len(result.learnings)

        # Verify provenance fields are populated
        for sid in result.sink_ids:
            entry = sink.artifacts[sid]
            prov = entry['provenance']
            assert prov.clustering_method == 'llm'
            assert prov.total_analyzed == 4
            assert prov.supporting_count == 2
            assert prov.cluster_category is not None
            assert prov.cluster_category == 'Cluster 0'
            assert prov.timestamp is not None
            # source_ref should come from the first supporting item
            assert prov.source_ref is not None
            assert prov.source_ref.startswith('conv_')
            # all_source_refs should be in metadata
            assert 'all_source_refs' in prov.metadata
            assert len(prov.metadata['all_source_refs']) == 2

    @pytest.mark.asyncio
    async def test_quality_gate_in_pipeline(self):
        """Confidence >= 0.7 with no actions should be demoted."""
        evidence = _make_evidence(4)
        eids = list(evidence.keys())

        cr = _make_clustering_result(eids, 1)
        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=cr)

        # Learning with high confidence but no actions
        learning = LearningArtifactOutput(
            title='No Actions',
            content='Content',
            tags=['tag'],
            confidence=0.85,
            supporting_item_ids=eids[:2],
            recommended_actions=[],  # empty!
        )
        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(return_value=[learning])

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=2,
        )
        result = await pipeline.run(evidence)

        if result.learnings:
            assert result.learnings[0].confidence == 0.69

    @pytest.mark.asyncio
    async def test_sanitizer_called(self):
        """Sanitizer should be called on each item's text."""
        evidence = _make_evidence(3)

        mock_sanitizer = AsyncMock()
        mock_sanitizer.sanitize = AsyncMock(
            side_effect=lambda text: text.upper()
        )

        cr = PatternDiscoveryResult(
            patterns=[],
            uncategorized=list(evidence.keys()),
            total_analyzed=3,
            method=ClusteringMethod.LLM,
        )
        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=cr)

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=AsyncMock(),
            sanitizer=mock_sanitizer,
        )
        await pipeline.run(evidence)

        assert mock_sanitizer.sanitize.call_count == 3

    @pytest.mark.asyncio
    async def test_empty_evidence(self):
        """Pipeline should handle empty evidence gracefully."""
        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(
            return_value=PatternDiscoveryResult(
                patterns=[],
                uncategorized=[],
                total_analyzed=0,
                method=ClusteringMethod.LLM,
            )
        )

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=AsyncMock(),
        )
        result = await pipeline.run({})

        assert result.learnings == []
        assert result.clustering_result.total_analyzed == 0
