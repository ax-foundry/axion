import pytest

from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    EvidenceItem,
    LearningArtifact,
    PatternDiscoveryResult,
    PipelineResult,
    Provenance,
)
from axion.caliber.pattern_discovery.discovery import PatternDiscovery


class TestEvidenceItem:
    """Tests for EvidenceItem dataclass."""

    def test_create_minimal(self):
        item = EvidenceItem(id='e1', text='some text')
        assert item.id == 'e1'
        assert item.text == 'some text'
        assert item.metadata == {}
        assert item.source_ref is None

    def test_create_full(self):
        item = EvidenceItem(
            id='e1',
            text='some text',
            metadata={'sentiment': 'negative', 'failed_step': 'checkout'},
            source_ref='conv_123',
        )
        assert item.metadata['sentiment'] == 'negative'
        assert item.source_ref == 'conv_123'


class TestLearningArtifact:
    """Tests for LearningArtifact dataclass."""

    def test_create_minimal(self):
        artifact = LearningArtifact(
            title='Test Learning',
            content='Synthesized insight',
            tags=['tag1'],
            confidence=0.8,
            supporting_item_ids=['e1', 'e2'],
        )
        assert artifact.title == 'Test Learning'
        assert artifact.confidence == 0.8
        assert artifact.recommended_actions == []
        assert artifact.counterexamples == []
        assert artifact.scope is None
        assert artifact.when_not_to_apply is None

    def test_create_full(self):
        artifact = LearningArtifact(
            title='Test Learning',
            content='Full content',
            tags=['tag1', 'tag2'],
            confidence=0.95,
            supporting_item_ids=['e1', 'e2', 'e3'],
            recommended_actions=['Fix the checkout flow'],
            counterexamples=['e4'],
            scope='Checkout page',
            when_not_to_apply='Guest users',
        )
        assert len(artifact.recommended_actions) == 1
        assert artifact.scope == 'Checkout page'


class TestProvenance:
    """Tests for Provenance dataclass."""

    def test_create_defaults(self):
        prov = Provenance()
        assert prov.source_ref is None
        assert prov.clustering_method is None
        assert prov.total_analyzed == 0
        assert prov.metadata == {}

    def test_create_full(self):
        prov = Provenance(
            source_ref='conv_123',
            clustering_method='llm',
            total_analyzed=50,
            supporting_count=5,
            cluster_category='Missing Context',
            timestamp='2026-01-01T00:00:00Z',
            metadata={'run_id': 'abc'},
        )
        assert prov.clustering_method == 'llm'
        assert prov.total_analyzed == 50


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_create(self):
        cr = PatternDiscoveryResult(
            patterns=[],
            uncategorized=[],
            total_analyzed=10,
            method=ClusteringMethod.LLM,
        )
        result = PipelineResult(
            clustering_result=cr,
            learnings=[],
            filtered_count=2,
            deduplicated_count=1,
            validation_repairs=3,
            sink_ids=['mem_1'],
        )
        assert result.filtered_count == 2
        assert result.deduplicated_count == 1
        assert result.validation_repairs == 3
        assert result.sink_ids == ['mem_1']
        assert result.metadata == {}


class TestEvidenceDictKeyMismatch:
    """Test that dict key != item.id raises ValueError."""

    def test_dict_key_mismatch_raises(self):
        evidence = {
            'wrong_key': EvidenceItem(id='e1', text='text'),
        }
        with pytest.raises(ValueError, match='does not match item.id'):
            PatternDiscovery._to_evidence_dict(evidence)

    def test_dict_key_match_ok(self):
        evidence = {
            'e1': EvidenceItem(id='e1', text='text'),
            'e2': EvidenceItem(id='e2', text='more text'),
        }
        result = PatternDiscovery._to_evidence_dict(evidence)
        assert 'e1' in result
        assert 'e2' in result

    def test_sequence_input(self):
        items = [
            EvidenceItem(id='e1', text='text'),
            EvidenceItem(id='e2', text='more text'),
        ]
        result = PatternDiscovery._to_evidence_dict(items)
        assert set(result.keys()) == {'e1', 'e2'}
