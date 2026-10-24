"""Shared fixtures for pattern-discovery tests.

The grouped-evidence fixture models the common shape where several evidence
items hang off one parent entity (chunks of a conversation, findings on a
ticket) and recurrence must be counted across parents, not items.
"""

from typing import Callable, Dict, List, Optional

import pytest

from axion.caliber.pattern_discovery.handlers import (
    ClusteringOutput,
    LearningArtifactOutput,
    PatternCategory,
)
from axion.caliber.pattern_discovery.models import (
    ClusteringMethod,
    DiscoveredPattern,
    EvidenceItem,
    PatternDiscoveryResult,
)


def parent_key(item: EvidenceItem) -> str:
    """Recurrence key: the parent entity carried in ``source_ref``."""
    return item.source_ref or item.id


def make_grouped_evidence(
    n_parents: int = 4, per_parent: int = 5
) -> Dict[str, EvidenceItem]:
    """``n_parents`` parents, ``per_parent`` items each, ids ``p{P}_i{I}``."""
    evidence: Dict[str, EvidenceItem] = {}
    for p in range(n_parents):
        for i in range(per_parent):
            eid = f'p{p}_i{i}'
            evidence[eid] = EvidenceItem(
                id=eid,
                text=f'Parent {p} observation {i}: the handoff step was skipped',
                metadata={'parent': f'parent_{p}', 'slot': str(i)},
                source_ref=f'parent_{p}',
            )
    return evidence


@pytest.fixture
def grouped_evidence() -> Dict[str, EvidenceItem]:
    return make_grouped_evidence()


class ScriptedClusterer:
    """An ``EvidenceClusterer`` that returns exactly the clusters it is given."""

    def __init__(
        self,
        clusters: List[List[str]],
        uncategorized: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._clusters = clusters
        self._uncategorized = uncategorized or []
        self._metadata = metadata or {}
        self.calls: List[Dict[str, EvidenceItem]] = []

    async def cluster(
        self, evidence: Dict[str, EvidenceItem], method: ClusteringMethod
    ) -> PatternDiscoveryResult:
        self.calls.append(evidence)
        patterns = [
            DiscoveredPattern(
                category=f'Cluster {i}',
                description=f'Description {i}',
                count=len(ids),
                record_ids=list(ids),
            )
            for i, ids in enumerate(self._clusters)
        ]
        return PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=list(self._uncategorized),
            total_analyzed=len(evidence),
            method=method,
            metadata=dict(self._metadata),
        )


class ScriptedWriter:
    """An ``ArtifactWriter`` whose supporting ids are chosen by ``pick``.

    ``pick`` receives the ``ClusterForDistillation`` and returns the ids to
    cite; by default it cites every example the writer was shown.
    """

    def __init__(
        self,
        pick: Optional[Callable] = None,
        confidence: float = 0.8,
        title: str = 'Learning',
    ) -> None:
        self._pick = pick or (lambda c: [e.item_id for e in c.examples])
        self._confidence = confidence
        self._title = title
        self.clusters: list = []

    async def distill(self, cluster, domain_context):
        self.clusters.append(cluster)
        return [
            LearningArtifactOutput(
                title=f'{self._title}: {cluster.category}',
                content='content',
                tags=['t'],
                confidence=self._confidence,
                supporting_item_ids=list(self._pick(cluster)),
                recommended_actions=['act'],
            )
        ]


class ScriptedClusteringHandler:
    """Stands in for ``EvidenceClusteringHandler``: returns a fixed output."""

    model_name = 'scripted'
    llm_provider = 'scripted'

    def __init__(self, output: ClusteringOutput) -> None:
        self._output = output
        self.inputs: list = []

    async def execute(self, input_data):
        self.inputs.append(input_data)
        return self._output


def clustering_output(
    patterns: List[List[str]], uncategorized: Optional[List[str]] = None
) -> ClusteringOutput:
    return ClusteringOutput(
        patterns=[
            PatternCategory(
                category=f'Cluster {i}', record_ids=ids, description=f'D{i}'
            )
            for i, ids in enumerate(patterns)
        ],
        uncategorized=uncategorized or [],
    )
