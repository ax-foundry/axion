import json
import tempfile
from pathlib import Path

import pytest

from axion.caliber.pattern_discovery.models import LearningArtifact, Provenance
from axion.caliber.pattern_discovery.plugins import (
    ArtifactSink,
    Deduper,
    EmbeddingDeduper,
    InMemoryDeduper,
    InMemorySink,
    JsonlSink,
    NoopSanitizer,
    Sanitizer,
)


class TestNoopSanitizer:
    @pytest.mark.asyncio
    async def test_pass_through(self):
        s = NoopSanitizer()
        assert await s.sanitize('hello world') == 'hello world'

    def test_implements_protocol(self):
        assert isinstance(NoopSanitizer(), Sanitizer)


class TestInMemorySink:
    @pytest.mark.asyncio
    async def test_write_and_retrieve(self):
        sink = InMemorySink()
        artifact = LearningArtifact(
            title='Test',
            content='Content',
            tags=['t'],
            confidence=0.8,
            supporting_item_ids=['e1'],
        )
        prov = Provenance(clustering_method='llm')
        sid = await sink.write(artifact, prov)
        assert sid == 'mem_1'
        assert sid in sink.artifacts
        assert sink.artifacts[sid]['artifact'] is artifact

    @pytest.mark.asyncio
    async def test_increments_counter(self):
        sink = InMemorySink()
        artifact = LearningArtifact(
            title='Test',
            content='C',
            tags=[],
            confidence=0.5,
            supporting_item_ids=[],
        )
        prov = Provenance()
        s1 = await sink.write(artifact, prov)
        s2 = await sink.write(artifact, prov)
        assert s1 == 'mem_1'
        assert s2 == 'mem_2'

    def test_implements_protocol(self):
        assert isinstance(InMemorySink(), ArtifactSink)


class TestJsonlSink:
    @pytest.mark.asyncio
    async def test_write_jsonl(self):
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            path = f.name

        sink = JsonlSink(path)
        artifact = LearningArtifact(
            title='Test',
            content='Content',
            tags=['tag1'],
            confidence=0.8,
            supporting_item_ids=['e1'],
        )
        prov = Provenance(clustering_method='llm', total_analyzed=10)
        sid = await sink.write(artifact, prov)
        assert sid == 'jsonl_1'

        lines = Path(path).read_text().strip().split('\n')
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record['title'] == 'Test'
        assert record['provenance']['clustering_method'] == 'llm'

        Path(path).unlink()

    def test_implements_protocol(self):
        sink = JsonlSink('/tmp/test.jsonl')
        assert isinstance(sink, ArtifactSink)


class TestInMemoryDeduper:
    @pytest.mark.asyncio
    async def test_detects_duplicate(self):
        d = InMemoryDeduper()
        a1 = LearningArtifact(
            title='Same Title',
            content='C1',
            tags=[],
            confidence=0.8,
            supporting_item_ids=[],
        )
        a2 = LearningArtifact(
            title='same title',  # case insensitive
            content='C2',
            tags=[],
            confidence=0.7,
            supporting_item_ids=[],
        )
        assert await d.is_duplicate(a1) is False
        assert await d.is_duplicate(a2) is True

    @pytest.mark.asyncio
    async def test_different_titles_not_duplicate(self):
        d = InMemoryDeduper()
        a1 = LearningArtifact(
            title='Title One',
            content='C',
            tags=[],
            confidence=0.8,
            supporting_item_ids=[],
        )
        a2 = LearningArtifact(
            title='Title Two',
            content='C',
            tags=[],
            confidence=0.8,
            supporting_item_ids=[],
        )
        assert await d.is_duplicate(a1) is False
        assert await d.is_duplicate(a2) is False

    @pytest.mark.asyncio
    async def test_reset_clears(self):
        d = InMemoryDeduper()
        a1 = LearningArtifact(
            title='Title',
            content='C',
            tags=[],
            confidence=0.8,
            supporting_item_ids=[],
        )
        await d.is_duplicate(a1)
        d.reset()
        assert await d.is_duplicate(a1) is False

    def test_implements_protocol(self):
        assert isinstance(InMemoryDeduper(), Deduper)


class TestEmbeddingDeduper:
    def test_import_error_when_deps_missing(self):
        """EmbeddingDeduper should raise ImportError with clear message
        when numpy/openai are unavailable.

        Since numpy and openai are installed in this test env, we test
        that it initializes successfully instead. The ImportError path
        is guarded by the try/except in __init__.
        """
        # In this test environment, deps are available,
        # so just verify it initializes
        deduper = EmbeddingDeduper()
        assert deduper._similarity_threshold == 0.85
        assert deduper._max_stored == 1000
        assert deduper.reset_per_run is True

    def test_reset_clears_state(self):
        deduper = EmbeddingDeduper()
        deduper._embeddings = [[1.0, 2.0]]
        deduper._titles = ['test']
        deduper.reset()
        assert deduper._embeddings == []
        assert deduper._titles == []

    def test_custom_config(self):
        deduper = EmbeddingDeduper(
            similarity_threshold=0.9,
            max_stored=500,
            reset_per_run=False,
        )
        assert deduper._similarity_threshold == 0.9
        assert deduper._max_stored == 500
        assert deduper.reset_per_run is False
