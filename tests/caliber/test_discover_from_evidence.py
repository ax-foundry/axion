import pytest

from axion.caliber.pattern_discovery.discovery import PatternDiscovery
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    EvidenceItem,
)


class TestDiscoverFromEvidence:
    """Tests for PatternDiscovery.discover_from_evidence edge cases."""

    @pytest.mark.asyncio
    async def test_discover_from_evidence_no_text(self):
        """Items with no text should produce empty result."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        evidence = [
            EvidenceItem(id='e1', text=''),
            EvidenceItem(id='e2', text=''),
        ]
        result = await discovery.discover_from_evidence(evidence)
        assert result.patterns == []
        assert result.method == ClusteringMethod.LLM

    @pytest.mark.asyncio
    async def test_discover_from_evidence_sequence_input(self):
        """Sequence input should be accepted and normalized."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        evidence = [
            EvidenceItem(id='e1', text=''),
            EvidenceItem(id='e2', text=''),
        ]
        # Should not raise
        result = await discovery.discover_from_evidence(evidence)
        assert result.total_analyzed == 2

    @pytest.mark.asyncio
    async def test_discover_from_evidence_dict_input(self):
        """Dict input should be accepted when keys match item.id."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        evidence = {
            'e1': EvidenceItem(id='e1', text=''),
            'e2': EvidenceItem(id='e2', text=''),
        }
        result = await discovery.discover_from_evidence(evidence)
        assert result.total_analyzed == 2

    @pytest.mark.asyncio
    async def test_discover_from_evidence_dict_key_mismatch(self):
        """Dict input with mismatched keys should raise ValueError."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        evidence = {
            'wrong': EvidenceItem(id='e1', text='text'),
        }
        with pytest.raises(ValueError, match='does not match item.id'):
            await discovery.discover_from_evidence(evidence)

    @pytest.mark.asyncio
    async def test_discover_from_evidence_invalid_method(self):
        """Invalid method should raise ValueError."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        evidence = [EvidenceItem(id='e1', text='text')]
        with pytest.raises(ValueError, match='Unknown clustering method'):
            await discovery.discover_from_evidence(evidence, method='bad')

    def test_to_evidence_dict_from_sequence(self):
        items = [
            EvidenceItem(id='a', text='t1'),
            EvidenceItem(id='b', text='t2'),
        ]
        result = PatternDiscovery._to_evidence_dict(items)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'a', 'b'}

    def test_to_evidence_dict_from_dict(self):
        items = {
            'a': EvidenceItem(id='a', text='t1'),
            'b': EvidenceItem(id='b', text='t2'),
        }
        result = PatternDiscovery._to_evidence_dict(items)
        assert result is items  # same reference for valid dict

    def test_to_evidence_dict_strict_validation(self):
        items = {
            'wrong_key': EvidenceItem(id='e1', text='t'),
        }
        with pytest.raises(ValueError):
            PatternDiscovery._to_evidence_dict(items)

    def test_init_with_metadata_config(self):
        from axion.caliber.pattern_discovery._utils import MetadataConfig

        cfg = MetadataConfig(
            include_in_clustering=True,
            max_keys=3,
        )
        discovery = PatternDiscovery(metadata_config=cfg)
        assert discovery._metadata_config.include_in_clustering is True
        assert discovery._metadata_config.max_keys == 3

    def test_init_with_seed(self):
        discovery = PatternDiscovery(seed=42)
        assert discovery._seed == 42

    def test_init_defaults_metadata_config(self):
        discovery = PatternDiscovery()
        assert discovery._metadata_config is not None
        assert discovery._metadata_config.include_in_clustering is False
