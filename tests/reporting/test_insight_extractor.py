from unittest.mock import AsyncMock, patch

import pytest

from axion.caliber.pattern_discovery.handlers import LearningArtifactOutput
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    EvidenceItem,
    PatternDiscoveryResult,
    PipelineResult,
)
from axion.caliber.pattern_discovery.pipeline import EvidencePipeline
from axion.reporting.insight_extractor import (
    InsightExtractor,
    InsightPattern,
    InsightResult,
    _issue_to_evidence,
)
from axion.reporting.issue_extractor import ExtractedIssue, IssueExtractionResult


def _make_issue(
    test_case_id='tc_1',
    metric_name='faithfulness',
    signal_group='claim_0',
    signal_name='faithfulness_verdict',
    value='CONTRADICTORY',
    score=0.0,
    reasoning='The claim contradicts the context.',
    query='What is X?',
    source_path='results[0].score_results[0].signals.claim_0',
):
    return ExtractedIssue(
        test_case_id=test_case_id,
        metric_name=metric_name,
        signal_group=signal_group,
        signal_name=signal_name,
        value=value,
        score=score,
        reasoning=reasoning,
        item_context={'query': query} if query else {},
        source_path=source_path,
    )


def _make_extraction_result(
    issues,
    evaluation_name='test_eval',
    total_test_cases=10,
):
    issues_by_metric = {}
    for issue in issues:
        issues_by_metric.setdefault(issue.metric_name, []).append(issue)
    issues_by_type = {}
    for issue in issues:
        key = f'{issue.metric_name}:{issue.signal_name}'
        issues_by_type.setdefault(key, []).append(issue)

    return IssueExtractionResult(
        run_id='run_1',
        evaluation_name=evaluation_name,
        total_test_cases=total_test_cases,
        total_signals_analyzed=len(issues) * 3,
        issues_found=len(issues),
        issues_by_metric=issues_by_metric,
        issues_by_type=issues_by_type,
        all_issues=issues,
    )


def _make_clustering_result(evidence_ids, n_clusters=2):
    per_cluster = max(1, len(evidence_ids) // n_clusters)
    patterns = []
    for c in range(n_clusters):
        start = c * per_cluster
        end = start + per_cluster if c < n_clusters - 1 else len(evidence_ids)
        rids = evidence_ids[start:end]
        if rids:
            patterns.append(
                DiscoveredPattern(
                    category=f'Cluster {c}',
                    description=f'Description for cluster {c}',
                    count=len(rids),
                    record_ids=rids,
                    examples=[f'example {c}'],
                    confidence=0.85,
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
        tags=['tag1'],
        confidence=confidence,
        supporting_item_ids=item_ids,
        recommended_actions=['Do something'],
        counterexamples=[],
    )


def _make_pipeline_result(evidence_dict, n_clusters=2):
    """Build a PipelineResult from evidence for mocking."""
    eids = list(evidence_dict.keys())
    clustering_result = _make_clustering_result(eids, n_clusters)
    return PipelineResult(
        clustering_result=clustering_result,
        learnings=[],
    )


# ── Adapter tests ──


class TestIssueToEvidence:
    def test_issue_to_evidence_mapping(self):
        """Verify id (hash-based), text, metadata, source_ref for a full issue."""
        issue = _make_issue()
        item = _issue_to_evidence(issue)

        assert item is not None
        assert len(item.id) == 16  # sha256 hex prefix
        assert item.source_ref == 'tc_1'
        assert 'The claim contradicts the context.' in item.text
        assert '[faithfulness / faithfulness_verdict: CONTRADICTORY]' in item.text
        assert 'Query: What is X?' in item.text
        assert item.metadata['metric_name'] == 'faithfulness'
        assert item.metadata['signal_name'] == 'faithfulness_verdict'
        assert item.metadata['signal_group'] == 'claim_0'
        assert item.metadata['value'] == 'CONTRADICTORY'
        assert item.metadata['score'] == 0.0
        assert item.metadata['source_path'] == 'results[0].score_results[0].signals.claim_0'

    def test_issue_to_evidence_no_text_returns_none(self):
        """Issue with no reasoning and no query -> None."""
        issue = _make_issue(reasoning=None, query=None)
        item = _issue_to_evidence(issue)
        assert item is None

    def test_issue_to_evidence_reasoning_only(self):
        """Issue with reasoning but no query -> valid."""
        issue = _make_issue(reasoning='Some reasoning', query=None)
        item = _issue_to_evidence(issue)
        assert item is not None
        assert 'Some reasoning' in item.text
        assert 'Query:' not in item.text

    def test_issue_to_evidence_query_only(self):
        """Issue with no reasoning but has query -> valid."""
        issue = _make_issue(reasoning=None, query='What is Y?')
        item = _issue_to_evidence(issue)
        assert item is not None
        assert 'Query: What is Y?' in item.text

    def test_issue_to_evidence_id_stability(self):
        """Same issue fields produce same hash ID regardless of creation order."""
        issue_a = _make_issue(test_case_id='tc_1', metric_name='faithfulness')
        issue_b = _make_issue(test_case_id='tc_1', metric_name='faithfulness')

        item_a = _issue_to_evidence(issue_a)
        item_b = _issue_to_evidence(issue_b)

        assert item_a.id == item_b.id


# ── Pipeline integration tests ──


def _build_mocked_pipeline(evidence_dict, n_clusters=2, recurrence_threshold=1):
    """Create an InsightExtractor with fully mocked pipeline internals."""
    eids = list(evidence_dict.keys())
    clustering_result = _make_clustering_result(eids, n_clusters)

    mock_clusterer = AsyncMock()
    mock_clusterer.cluster = AsyncMock(return_value=clustering_result)

    mock_writer = AsyncMock()
    mock_writer.distill = AsyncMock(
        side_effect=lambda cluster, ctx: [
            _make_learning_output(cluster.item_ids[:2])
        ]
    )

    pipeline = EvidencePipeline(
        clusterer=mock_clusterer,
        writer=mock_writer,
        recurrence_threshold=recurrence_threshold,
    )
    return pipeline, mock_clusterer, mock_writer


class TestInsightExtractorAnalyze:
    @pytest.mark.asyncio
    async def test_basic_analyze(self):
        """6 issues across 2 metrics -> patterns + learnings returned."""
        issues = [
            _make_issue(test_case_id=f'tc_{i}', metric_name='faithfulness')
            for i in range(3)
        ] + [
            _make_issue(test_case_id=f'tc_{i+3}', metric_name='contextual_recall')
            for i in range(3)
        ]
        extraction = _make_extraction_result(issues)

        # Convert to get the evidence IDs that will be generated
        evidence_items = [_issue_to_evidence(i) for i in issues]
        evidence_dict = {e.id: e for e in evidence_items if e is not None}

        pipeline, mock_clusterer, mock_writer = _build_mocked_pipeline(
            evidence_dict, n_clusters=2, recurrence_threshold=1
        )
        extractor = InsightExtractor(pipeline=pipeline, method=ClusteringMethod.LLM)

        result = await extractor.analyze(extraction)

        assert isinstance(result, InsightResult)
        assert result.total_issues_analyzed == 6
        assert len(result.patterns) == 2
        assert len(result.learnings) >= 1
        assert result.clustering_method == ClusteringMethod.LLM
        assert result.pipeline_result is not None

    @pytest.mark.asyncio
    async def test_cross_metric_detection(self):
        """Cluster containing issues from 2+ metrics -> is_cross_metric=True."""
        issues = [
            _make_issue(test_case_id='tc_1', metric_name='faithfulness'),
            _make_issue(test_case_id='tc_2', metric_name='contextual_recall'),
            _make_issue(test_case_id='tc_3', metric_name='faithfulness'),
            _make_issue(test_case_id='tc_4', metric_name='contextual_recall'),
        ]
        extraction = _make_extraction_result(issues)

        evidence_items = [_issue_to_evidence(i) for i in issues]
        evidence_dict = {e.id: e for e in evidence_items if e is not None}
        eids = list(evidence_dict.keys())

        # Single cluster with all items -> cross-metric
        clustering_result = PatternDiscoveryResult(
            patterns=[
                DiscoveredPattern(
                    category='Retrieval Issues',
                    description='Problems with retrieval',
                    count=len(eids),
                    record_ids=eids,
                    examples=['example'],
                    confidence=0.9,
                )
            ],
            uncategorized=[],
            total_analyzed=len(eids),
            method=ClusteringMethod.LLM,
        )

        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=clustering_result)
        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            side_effect=lambda cluster, ctx: [
                _make_learning_output(cluster.item_ids[:2])
            ]
        )

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=1,
        )
        extractor = InsightExtractor(pipeline=pipeline, method=ClusteringMethod.LLM)
        result = await extractor.analyze(extraction)

        assert len(result.patterns) == 1
        pattern = result.patterns[0]
        assert pattern.is_cross_metric is True
        assert len(pattern.metrics_involved) >= 2
        assert 'faithfulness' in pattern.metrics_involved
        assert 'contextual_recall' in pattern.metrics_involved
        assert pattern.distinct_test_cases == 4

    @pytest.mark.asyncio
    async def test_single_metric_cluster(self):
        """Cluster from 1 metric -> is_cross_metric=False."""
        issues = [
            _make_issue(test_case_id=f'tc_{i}', metric_name='faithfulness')
            for i in range(4)
        ]
        extraction = _make_extraction_result(issues)

        evidence_items = [_issue_to_evidence(i) for i in issues]
        evidence_dict = {e.id: e for e in evidence_items if e is not None}
        eids = list(evidence_dict.keys())

        clustering_result = PatternDiscoveryResult(
            patterns=[
                DiscoveredPattern(
                    category='Faith Issues',
                    description='Faithfulness failures',
                    count=len(eids),
                    record_ids=eids,
                    examples=['example'],
                )
            ],
            uncategorized=[],
            total_analyzed=len(eids),
            method=ClusteringMethod.LLM,
        )

        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=clustering_result)
        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            side_effect=lambda cluster, ctx: [
                _make_learning_output(cluster.item_ids[:2])
            ]
        )

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=1,
        )
        extractor = InsightExtractor(pipeline=pipeline, method=ClusteringMethod.LLM)
        result = await extractor.analyze(extraction)

        assert len(result.patterns) == 1
        assert result.patterns[0].is_cross_metric is False
        assert result.patterns[0].metrics_involved == ['faithfulness']

    @pytest.mark.asyncio
    async def test_recurrence_by_test_case(self):
        """Multiple signals from same test_case_id count as 1 occurrence."""
        # 3 issues from same test case
        issues = [
            _make_issue(
                test_case_id='tc_1',
                metric_name='faithfulness',
                signal_group=f'claim_{i}',
            )
            for i in range(3)
        ]
        extraction = _make_extraction_result(issues)

        evidence_items = [_issue_to_evidence(i) for i in issues]
        evidence_dict = {e.id: e for e in evidence_items if e is not None}
        eids = list(evidence_dict.keys())

        clustering_result = PatternDiscoveryResult(
            patterns=[
                DiscoveredPattern(
                    category='Single Source',
                    description='D',
                    count=len(eids),
                    record_ids=eids,
                )
            ],
            uncategorized=[],
            total_analyzed=len(eids),
            method=ClusteringMethod.LLM,
        )

        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=clustering_result)
        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            side_effect=lambda cluster, ctx: [
                _make_learning_output(cluster.item_ids)
            ]
        )

        # recurrence_threshold=2, but all items have source_ref='tc_1'
        # so the recurrence key fn groups them as 1 occurrence
        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=2,
            recurrence_key_fn=lambda item: item.source_ref or item.id,
        )
        extractor = InsightExtractor(pipeline=pipeline, method=ClusteringMethod.LLM)
        result = await extractor.analyze(extraction)

        # Only 1 unique source_ref -> below threshold 2 -> no learnings
        assert len(result.learnings) == 0

    @pytest.mark.asyncio
    async def test_empty_issues(self):
        """Empty all_issues -> empty InsightResult, pipeline NOT called."""
        extraction = _make_extraction_result([])

        mock_clusterer = AsyncMock()
        mock_writer = AsyncMock()
        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
        )
        extractor = InsightExtractor(pipeline=pipeline, method=ClusteringMethod.LLM)
        result = await extractor.analyze(extraction)

        assert result.patterns == []
        assert result.learnings == []
        assert result.total_issues_analyzed == 0
        mock_clusterer.cluster.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_issues_filtered_out(self):
        """Non-empty input, but all _issue_to_evidence return None -> empty, pipeline NOT called."""
        issues = [
            _make_issue(reasoning=None, query=None),
            _make_issue(reasoning=None, query=None, test_case_id='tc_2'),
        ]
        extraction = _make_extraction_result(issues)

        mock_clusterer = AsyncMock()
        mock_writer = AsyncMock()
        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
        )
        extractor = InsightExtractor(pipeline=pipeline, method=ClusteringMethod.LLM)
        result = await extractor.analyze(extraction)

        assert result.patterns == []
        assert result.learnings == []
        assert result.total_issues_analyzed == 0
        mock_clusterer.cluster.assert_not_called()

    def test_analyze_sync(self):
        """Verify sync wrapper works via run_async_function."""
        extraction = _make_extraction_result([])

        mock_clusterer = AsyncMock()
        mock_writer = AsyncMock()
        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
        )
        extractor = InsightExtractor(pipeline=pipeline, method=ClusteringMethod.LLM)
        result = extractor.analyze_sync(extraction)

        assert isinstance(result, InsightResult)
        assert result.patterns == []
        assert result.learnings == []

    def test_custom_pipeline_override(self):
        """Passing pipeline= uses it instead of building default."""
        mock_clusterer = AsyncMock()
        mock_writer = AsyncMock()
        custom_pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
        )
        extractor = InsightExtractor(pipeline=custom_pipeline)
        assert extractor._pipeline is custom_pipeline

    def test_pipeline_kwargs_conflict(self):
        """Passing both pipeline= and kwargs -> ValueError."""
        mock_clusterer = AsyncMock()
        mock_writer = AsyncMock()
        custom_pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
        )
        with pytest.raises(ValueError, match='Cannot pass pipeline_kwargs'):
            InsightExtractor(
                pipeline=custom_pipeline,
                seed=42,
            )

    @pytest.mark.asyncio
    async def test_domain_context_with_name(self):
        """Verify generated context string includes evaluation name."""
        issues = [
            _make_issue(test_case_id='tc_1', metric_name='faithfulness'),
            _make_issue(test_case_id='tc_2', metric_name='contextual_recall'),
        ]
        extraction = _make_extraction_result(
            issues, evaluation_name='my_eval', total_test_cases=5
        )

        evidence_items = [_issue_to_evidence(i) for i in issues]
        evidence_dict = {e.id: e for e in evidence_items if e is not None}
        eids = list(evidence_dict.keys())

        clustering_result = _make_clustering_result(eids, n_clusters=1)
        mock_clusterer = AsyncMock()
        mock_clusterer.cluster = AsyncMock(return_value=clustering_result)
        mock_writer = AsyncMock()
        mock_writer.distill = AsyncMock(
            side_effect=lambda cluster, ctx: [
                _make_learning_output(cluster.item_ids[:2])
            ]
        )

        pipeline = EvidencePipeline(
            clusterer=mock_clusterer,
            writer=mock_writer,
            recurrence_threshold=1,
        )
        extractor = InsightExtractor(pipeline=pipeline, method=ClusteringMethod.LLM)

        # Check domain context directly
        ctx = extractor._build_domain_context(extraction)
        assert "Evaluation 'my_eval'" in ctx
        assert '5 test cases' in ctx
        assert '2 issues' in ctx
        assert 'faithfulness' in ctx
        assert 'contextual_recall' in ctx

    @pytest.mark.asyncio
    async def test_domain_context_without_name(self):
        """Verify generated context string works when name is None."""
        issues = [_make_issue(test_case_id='tc_1', metric_name='faithfulness')]
        extraction = _make_extraction_result(
            issues, evaluation_name=None, total_test_cases=3
        )

        extractor = InsightExtractor(
            pipeline=EvidencePipeline(
                clusterer=AsyncMock(), writer=AsyncMock()
            ),
            method=ClusteringMethod.LLM,
        )
        ctx = extractor._build_domain_context(extraction)
        assert ctx.startswith('Evaluation:')
        assert "'" not in ctx
        assert '3 test cases' in ctx

    @pytest.mark.asyncio
    async def test_metadata_config_passthrough(self):
        """Verify allowed_keys influence the MetadataConfig passed to pipeline."""
        # When no custom pipeline is provided, InsightExtractor builds one
        # with specific MetadataConfig
        extractor = InsightExtractor(
            # Use a mock LLM to avoid needing real LLM
            llm=AsyncMock(),
            method=ClusteringMethod.LLM,
        )
        meta_cfg = extractor._pipeline._metadata_config
        assert meta_cfg.include_in_clustering is True
        assert meta_cfg.allowed_keys == {'metric_name', 'signal_name', 'value', 'score'}
