from axion.caliber.pattern_discovery._utils import (
    MetadataConfig,
    aggregate_cluster_metadata,
    check_recurrence,
    default_tag_normalizer,
    deterministic_sample,
    format_metadata_header,
    validate_learning,
)
from axion.caliber.pattern_discovery.handlers import LearningArtifactOutput
from axion.caliber.pattern_discovery.models import EvidenceItem


class TestMetadataConfig:
    """Tests for MetadataConfig and metadata formatting."""

    def test_default_denied_keys(self):
        cfg = MetadataConfig()
        assert 'email' in cfg.denied_keys
        assert 'password' in cfg.denied_keys
        assert 'ssn' in cfg.denied_keys

    def test_denied_keys_filtered(self):
        cfg = MetadataConfig()
        header = format_metadata_header(
            {'email': 'user@test.com', 'sentiment': 'negative'}, cfg
        )
        assert header is not None
        assert 'email' not in header
        assert 'sentiment' in header

    def test_allowed_keys_filter(self):
        cfg = MetadataConfig(allowed_keys={'sentiment', 'topic'})
        header = format_metadata_header(
            {'sentiment': 'negative', 'topic': 'billing', 'extra': 'val'},
            cfg,
        )
        assert header is not None
        assert 'extra' not in header
        assert 'sentiment' in header

    def test_max_keys_cap(self):
        cfg = MetadataConfig(max_keys=2)
        header = format_metadata_header({'a': '1', 'b': '2', 'c': '3', 'd': '4'}, cfg)
        assert header is not None
        # Should only have 2 key=val pairs
        assert header.count('=') == 2

    def test_max_value_length_truncation(self):
        cfg = MetadataConfig(max_value_length=5)
        header = format_metadata_header(
            {'key': 'a very long value that should be truncated'}, cfg
        )
        assert header is not None
        assert 'a ver' in header

    def test_max_header_chars_cap(self):
        cfg = MetadataConfig(max_header_chars=30)
        header = format_metadata_header(
            {'long_key': 'long_value', 'another': 'value'}, cfg
        )
        assert header is not None
        assert len(header) <= 30

    def test_empty_metadata_returns_none(self):
        cfg = MetadataConfig()
        header = format_metadata_header({}, cfg)
        assert header is None

    def test_all_keys_denied_returns_none(self):
        cfg = MetadataConfig()
        header = format_metadata_header({'email': 'user@test.com'}, cfg)
        assert header is None

    def test_include_in_clustering_default_false(self):
        cfg = MetadataConfig()
        assert cfg.include_in_clustering is False

    def test_include_in_distillation_default_true(self):
        cfg = MetadataConfig()
        assert cfg.include_in_distillation is True


class TestAggregateClusterMetadata:
    """Tests for aggregate_cluster_metadata."""

    def test_basic_aggregation(self):
        items = {
            'e1': EvidenceItem(id='e1', text='t', metadata={'step': 'checkout'}),
            'e2': EvidenceItem(id='e2', text='t', metadata={'step': 'checkout'}),
            'e3': EvidenceItem(id='e3', text='t', metadata={'step': 'billing'}),
        }
        cfg = MetadataConfig()
        result = aggregate_cluster_metadata(items, ['e1', 'e2', 'e3'], cfg)
        assert result is not None
        assert 'step' in result
        assert 'checkout' in result

    def test_empty_item_ids(self):
        items = {'e1': EvidenceItem(id='e1', text='t', metadata={'a': '1'})}
        cfg = MetadataConfig()
        result = aggregate_cluster_metadata(items, [], cfg)
        assert result is None


class TestValidateLearning:
    """Tests for validate_learning function."""

    def _make_learning(self, **overrides):
        defaults = {
            'title': 'Test',
            'content': 'Content',
            'tags': ['tag'],
            'confidence': 0.8,
            'supporting_item_ids': ['e1', 'e2'],
            'recommended_actions': ['Do something'],
            'counterexamples': [],
            'scope': None,
            'when_not_to_apply': None,
        }
        defaults.update(overrides)
        return LearningArtifactOutput(**defaults)

    def test_valid_learning_passes(self):
        learning = self._make_learning()
        cluster_ids = {'e1', 'e2', 'e3'}
        artifact, repairs = validate_learning(learning, cluster_ids)
        assert artifact is not None
        assert repairs == 0
        assert artifact.supporting_item_ids == ['e1', 'e2']

    def test_invalid_supporting_ids_removed(self):
        learning = self._make_learning(supporting_item_ids=['e1', 'e2', 'hallucinated'])
        cluster_ids = {'e1', 'e2'}
        artifact, repairs = validate_learning(learning, cluster_ids)
        assert artifact is not None
        assert repairs == 1
        assert 'hallucinated' not in artifact.supporting_item_ids

    def test_all_supporting_ids_invalid_returns_none(self):
        learning = self._make_learning(
            supporting_item_ids=['hallucinated1', 'hallucinated2']
        )
        cluster_ids = {'e1', 'e2'}
        artifact, repairs = validate_learning(learning, cluster_ids)
        assert artifact is None
        assert repairs == 2

    def test_cross_cluster_ids_rejected(self):
        """IDs valid globally but not in this cluster should be rejected."""
        learning = self._make_learning(
            supporting_item_ids=['e1', 'e5']  # e5 is from another cluster
        )
        cluster_ids = {'e1', 'e2', 'e3'}
        artifact, repairs = validate_learning(learning, cluster_ids)
        assert artifact is not None
        assert repairs == 1
        assert artifact.supporting_item_ids == ['e1']

    def test_invalid_counterexamples_removed(self):
        learning = self._make_learning(counterexamples=['e1', 'bad_id'])
        cluster_ids = {'e1', 'e2'}
        artifact, repairs = validate_learning(learning, cluster_ids)
        assert artifact is not None
        assert repairs == 1
        assert artifact.counterexamples == ['e1']

    def test_quality_gate_demotes_confidence(self):
        """confidence >= 0.7 with no recommended_actions => demoted."""
        learning = self._make_learning(confidence=0.85, recommended_actions=[])
        cluster_ids = {'e1', 'e2'}
        artifact, repairs = validate_learning(learning, cluster_ids)
        assert artifact is not None
        assert artifact.confidence == 0.69
        assert repairs == 1

    def test_quality_gate_not_triggered_below_threshold(self):
        learning = self._make_learning(confidence=0.5, recommended_actions=[])
        cluster_ids = {'e1', 'e2'}
        artifact, repairs = validate_learning(learning, cluster_ids)
        assert artifact is not None
        assert artifact.confidence == 0.5
        assert repairs == 0

    def test_quality_gate_not_triggered_with_actions(self):
        learning = self._make_learning(confidence=0.9, recommended_actions=['action1'])
        cluster_ids = {'e1', 'e2'}
        artifact, repairs = validate_learning(learning, cluster_ids)
        assert artifact is not None
        assert artifact.confidence == 0.9
        assert repairs == 0


class TestCheckRecurrence:
    """Tests for check_recurrence with custom key functions."""

    def _make_evidence(self):
        return {
            'e1': EvidenceItem(id='e1', text='t', source_ref='conv_1'),
            'e2': EvidenceItem(id='e2', text='t', source_ref='conv_1'),
            'e3': EvidenceItem(id='e3', text='t', source_ref='conv_2'),
        }

    def test_default_key_counts_items(self):
        evidence = self._make_evidence()
        assert check_recurrence(['e1', 'e2'], evidence, threshold=2) is True
        assert check_recurrence(['e1'], evidence, threshold=2) is False

    def test_custom_key_deduplicates_by_source(self):
        """2 items from same source_ref count as 1 with custom key."""
        evidence = self._make_evidence()
        key_fn = lambda item: item.source_ref or item.id

        # e1, e2 share conv_1 → only 1 unique key
        assert (
            check_recurrence(['e1', 'e2'], evidence, threshold=2, key_fn=key_fn)
            is False
        )
        # e1 (conv_1), e3 (conv_2) → 2 unique keys
        assert (
            check_recurrence(['e1', 'e3'], evidence, threshold=2, key_fn=key_fn) is True
        )

    def test_missing_ids_skipped(self):
        evidence = self._make_evidence()
        assert check_recurrence(['e1', 'missing'], evidence, threshold=2) is False


class TestDeterministicSample:
    """Tests for deterministic_sample."""

    def test_returns_all_when_below_max(self):
        items = {f'e{i}': EvidenceItem(id=f'e{i}', text=f't{i}') for i in range(3)}
        result = deterministic_sample(items, max_items=5)
        assert len(result) == 3

    def test_samples_to_max(self):
        items = {f'e{i}': EvidenceItem(id=f'e{i}', text=f't{i}') for i in range(10)}
        result = deterministic_sample(items, max_items=5)
        assert len(result) == 5

    def test_no_seed_is_deterministic(self):
        items = {f'e{i}': EvidenceItem(id=f'e{i}', text=f't{i}') for i in range(10)}
        r1 = deterministic_sample(items, max_items=5)
        r2 = deterministic_sample(items, max_items=5)
        assert list(r1.keys()) == list(r2.keys())

    def test_seed_reproducible(self):
        items = {f'e{i}': EvidenceItem(id=f'e{i}', text=f't{i}') for i in range(10)}
        r1 = deterministic_sample(items, max_items=5, seed=42)
        r2 = deterministic_sample(items, max_items=5, seed=42)
        assert set(r1.keys()) == set(r2.keys())

    def test_different_seeds_different_results(self):
        items = {f'e{i}': EvidenceItem(id=f'e{i}', text=f't{i}') for i in range(20)}
        r1 = deterministic_sample(items, max_items=5, seed=42)
        r2 = deterministic_sample(items, max_items=5, seed=99)
        # With 20 items and 5 selected, different seeds should differ
        assert set(r1.keys()) != set(r2.keys())


class TestDefaultTagNormalizer:
    """Tests for default_tag_normalizer."""

    def test_basic_normalization(self):
        tags = ['  Missing Context  ', 'Bug Fix', 'missing context']
        result = default_tag_normalizer(tags)
        assert result == ['missing_context', 'bug_fix']

    def test_empty_tags_dropped(self):
        tags = ['', '  ', 'valid']
        result = default_tag_normalizer(tags)
        assert result == ['valid']

    def test_long_tags_dropped(self):
        long_tag = 'a' * 51
        tags = ['short', long_tag]
        result = default_tag_normalizer(tags)
        assert result == ['short']

    def test_max_10_tags(self):
        tags = [f'tag{i}' for i in range(15)]
        result = default_tag_normalizer(tags)
        assert len(result) == 10

    def test_preserves_order(self):
        tags = ['zeta', 'alpha', 'beta']
        result = default_tag_normalizer(tags)
        assert result == ['zeta', 'alpha', 'beta']
