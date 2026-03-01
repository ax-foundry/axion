"""Tests for pattern discovery display functions (console output)."""

from __future__ import annotations

import pytest

from axion.caliber.pattern_discovery.display import (
    display_learnings,
    display_patterns,
    display_pipeline_result,
)
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    LearningArtifact,
    PatternDiscoveryResult,
    PipelineResult,
)
from axion.caliber.pattern_discovery.pipeline import EvidencePipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_patterns() -> list[DiscoveredPattern]:
    return [
        DiscoveredPattern(
            category='Timeout Errors',
            description='Requests failing due to upstream timeouts',
            count=12,
            record_ids=['r1', 'r2'],
            examples=['Connection timed out after 30s', 'Gateway timeout'],
            confidence=0.85,
        ),
        DiscoveredPattern(
            category='Auth Failures',
            description='Token validation issues',
            count=7,
            record_ids=['r3'],
            examples=[],
            confidence=None,
        ),
    ]


@pytest.fixture()
def sample_discovery(
    sample_patterns: list[DiscoveredPattern],
) -> PatternDiscoveryResult:
    return PatternDiscoveryResult(
        patterns=sample_patterns,
        uncategorized=['r99'],
        total_analyzed=20,
        method=ClusteringMethod.LLM,
    )


@pytest.fixture()
def sample_learnings() -> list[LearningArtifact]:
    return [
        LearningArtifact(
            title='Increase timeout budget',
            content='Upstream calls need longer deadlines during peak.',
            tags=['reliability', 'timeouts'],
            confidence=0.9,
            supporting_item_ids=['r1', 'r2'],
            recommended_actions=['Raise timeout to 60s', 'Add retry with backoff'],
            counterexamples=['Batch jobs already have 120s timeout'],
            scope='API gateway',
            when_not_to_apply='Offline batch processing',
        ),
        LearningArtifact(
            title='Rotate signing keys',
            content='Expired keys cause silent auth failures.',
            tags=['security'],
            confidence=0.75,
            supporting_item_ids=['r3'],
        ),
    ]


@pytest.fixture()
def sample_pipeline(
    sample_discovery: PatternDiscoveryResult,
    sample_learnings: list[LearningArtifact],
) -> PipelineResult:
    return PipelineResult(
        clustering_result=sample_discovery,
        learnings=sample_learnings,
        filtered_count=3,
        deduplicated_count=1,
        validation_repairs=2,
        sink_ids=['s1', 's2'],
    )


# ---------------------------------------------------------------------------
# display_patterns
# ---------------------------------------------------------------------------


class TestDisplayPatterns:
    def test_shows_summary(self, capsys, sample_discovery):
        display_patterns(sample_discovery)
        out = capsys.readouterr().out
        assert 'Pattern Discovery Results' in out
        assert 'llm' in out
        assert '20' in out  # total_analyzed
        assert '2' in out  # pattern count

    def test_shows_pattern_details(self, capsys, sample_discovery):
        display_patterns(sample_discovery)
        out = capsys.readouterr().out
        assert 'Timeout Errors' in out
        assert '85%' in out  # confidence
        assert 'Connection timed out' in out  # example
        assert 'Auth Failures' in out

    def test_empty_patterns(self, capsys):
        result = PatternDiscoveryResult(
            patterns=[],
            uncategorized=[],
            total_analyzed=5,
            method=ClusteringMethod.BERTOPIC,
        )
        display_patterns(result)
        out = capsys.readouterr().out
        assert 'No patterns discovered' in out

    def test_pattern_without_confidence(self, capsys, sample_patterns):
        result = PatternDiscoveryResult(
            patterns=[sample_patterns[1]],  # Auth Failures, confidence=None
            uncategorized=[],
            total_analyzed=7,
            method=ClusteringMethod.LLM,
        )
        display_patterns(result)
        out = capsys.readouterr().out
        assert 'Auth Failures' in out
        # Should not crash and should not show "None"
        assert 'None' not in out


# ---------------------------------------------------------------------------
# display_learnings
# ---------------------------------------------------------------------------


class TestDisplayLearnings:
    def test_shows_learning_details(self, capsys, sample_learnings):
        display_learnings(sample_learnings)
        out = capsys.readouterr().out
        assert 'Learning Artifacts' in out
        assert 'Increase timeout budget' in out
        assert '90%' in out  # confidence
        assert 'reliability' in out  # tag
        assert 'Raise timeout to 60s' in out  # action
        assert 'Batch jobs already' in out  # counterexample
        assert 'API gateway' in out  # scope
        assert 'Offline batch processing' in out  # when_not_to_apply

    def test_minimal_learning(self, capsys):
        la = LearningArtifact(
            title='Simple insight',
            content='Something learned.',
            tags=['misc'],
            confidence=0.5,
            supporting_item_ids=['x1'],
        )
        display_learnings([la])
        out = capsys.readouterr().out
        assert 'Simple insight' in out
        assert '50%' in out

    def test_empty_learnings(self, capsys):
        display_learnings([])
        out = capsys.readouterr().out
        assert 'No learnings generated' in out


# ---------------------------------------------------------------------------
# display_pipeline_result
# ---------------------------------------------------------------------------


class TestDisplayPipelineResult:
    def test_shows_pipeline_summary(self, capsys, sample_pipeline):
        display_pipeline_result(sample_pipeline)
        out = capsys.readouterr().out
        assert 'Evidence Pipeline Results' in out
        assert 'Filtered: 3' in out
        assert 'Deduplicated: 1' in out
        assert 'Validation repairs: 2' in out
        assert 'Artifacts written: 2' in out

    def test_includes_patterns_and_learnings(self, capsys, sample_pipeline):
        display_pipeline_result(sample_pipeline)
        out = capsys.readouterr().out
        assert 'Pattern Discovery Results' in out
        assert 'Learning Artifacts' in out
        assert 'Timeout Errors' in out
        assert 'Increase timeout budget' in out

    def test_no_sink_ids(self, capsys, sample_discovery, sample_learnings):
        result = PipelineResult(
            clustering_result=sample_discovery,
            learnings=sample_learnings,
        )
        display_pipeline_result(result)
        out = capsys.readouterr().out
        assert 'Artifacts written' not in out


# ---------------------------------------------------------------------------
# EvidencePipeline.display
# ---------------------------------------------------------------------------


class TestPipelineDisplayMethod:
    def test_pipeline_display_delegates(self, capsys, sample_pipeline):
        pipeline = EvidencePipeline.__new__(EvidencePipeline)
        pipeline.display(sample_pipeline)
        out = capsys.readouterr().out
        assert 'Evidence Pipeline Results' in out
        assert 'Timeout Errors' in out
        assert 'Increase timeout budget' in out
