import pytest

from axion.caliber.pattern_discovery.handlers import (
    ClusterForDistillation,
    DistillationInput,
    DistillationOutput,
    EvidenceClusteringInput,
    EvidenceNote,
    LearningArtifactOutput,
)


class TestEvidenceNote:
    def test_create(self):
        note = EvidenceNote(item_id='e1', text='some text')
        assert note.item_id == 'e1'
        assert note.text == 'some text'
        assert note.context is None

    def test_with_context(self):
        note = EvidenceNote(
            item_id='e1',
            text='text',
            context='[meta: step=checkout]',
        )
        assert note.context == '[meta: step=checkout]'


class TestEvidenceClusteringInput:
    def test_create(self):
        inp = EvidenceClusteringInput(
            items=[
                EvidenceNote(item_id='e1', text='note 1'),
                EvidenceNote(item_id='e2', text='note 2'),
            ]
        )
        assert len(inp.items) == 2


class TestClusterForDistillation:
    def test_create(self):
        cluster = ClusterForDistillation(
            category='Missing Context',
            description='Items missing context',
            item_ids=['e1', 'e2'],
            example_texts=['example 1', 'example 2'],
        )
        assert cluster.category == 'Missing Context'
        assert cluster.metadata_summary is None

    def test_with_metadata_summary(self):
        cluster = ClusterForDistillation(
            category='Bug Reports',
            description='Bug-related items',
            item_ids=['e1'],
            example_texts=['text'],
            metadata_summary='severity: high (80%), medium (20%)',
        )
        assert cluster.metadata_summary is not None


class TestDistillationInput:
    def test_create(self):
        cluster = ClusterForDistillation(
            category='Test',
            description='Desc',
            item_ids=['e1'],
            example_texts=['text'],
        )
        inp = DistillationInput(cluster=cluster)
        assert inp.cluster.category == 'Test'
        assert inp.domain_context is None

    def test_with_domain_context(self):
        cluster = ClusterForDistillation(
            category='Test',
            description='Desc',
            item_ids=['e1'],
            example_texts=['text'],
        )
        inp = DistillationInput(
            cluster=cluster,
            domain_context='E-commerce checkout flow',
        )
        assert inp.domain_context == 'E-commerce checkout flow'


class TestLearningArtifactOutput:
    def test_create_minimal(self):
        out = LearningArtifactOutput(
            title='Test',
            content='Content',
            tags=['tag'],
            confidence=0.8,
            supporting_item_ids=['e1'],
        )
        assert out.recommended_actions == []
        assert out.counterexamples == []
        assert out.scope is None

    def test_create_full(self):
        out = LearningArtifactOutput(
            title='Test',
            content='Content',
            tags=['tag1', 'tag2'],
            confidence=0.9,
            supporting_item_ids=['e1', 'e2'],
            recommended_actions=['Fix checkout'],
            counterexamples=['e3'],
            scope='Checkout flow',
            when_not_to_apply='Mobile',
        )
        assert len(out.recommended_actions) == 1
        assert len(out.counterexamples) == 1


class TestDistillationOutput:
    def test_create(self):
        out = DistillationOutput(
            learnings=[
                LearningArtifactOutput(
                    title='L1',
                    content='C1',
                    tags=['t'],
                    confidence=0.8,
                    supporting_item_ids=['e1'],
                ),
                LearningArtifactOutput(
                    title='L2',
                    content='C2',
                    tags=['t2'],
                    confidence=0.6,
                    supporting_item_ids=['e2'],
                ),
            ]
        )
        assert len(out.learnings) == 2

    def test_empty_learnings(self):
        out = DistillationOutput(learnings=[])
        assert out.learnings == []
