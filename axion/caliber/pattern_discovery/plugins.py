import json
from pathlib import Path
from typing import Dict, List, Protocol, runtime_checkable

from axion.caliber.pattern_discovery.models import LearningArtifact, Provenance


@runtime_checkable
class Sanitizer(Protocol):
    async def sanitize(self, text: str) -> str: ...


@runtime_checkable
class ArtifactSink(Protocol):
    async def write(
        self, artifact: LearningArtifact, provenance: Provenance
    ) -> str: ...


@runtime_checkable
class Deduper(Protocol):
    async def is_duplicate(self, artifact: LearningArtifact) -> bool: ...



class NoopSanitizer:
    """Async pass-through sanitizer (default)."""

    async def sanitize(self, text: str) -> str:
        return text


class InMemorySink:
    """Dict-based in-memory sink for testing/prototyping."""

    def __init__(self) -> None:
        self._counter = 0
        self.artifacts: Dict[str, Dict] = {}

    async def write(
        self, artifact: LearningArtifact, provenance: Provenance
    ) -> str:
        self._counter += 1
        sink_id = f'mem_{self._counter}'
        self.artifacts[sink_id] = {
            'artifact': artifact,
            'provenance': provenance,
        }
        return sink_id


class JsonlSink:
    """Appends learning artifacts as JSON lines to a file."""

    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        self._counter = 0

    async def write(
        self, artifact: LearningArtifact, provenance: Provenance
    ) -> str:
        self._counter += 1
        sink_id = f'jsonl_{self._counter}'
        record = {
            'sink_id': sink_id,
            'title': artifact.title,
            'content': artifact.content,
            'tags': artifact.tags,
            'confidence': artifact.confidence,
            'supporting_item_ids': artifact.supporting_item_ids,
            'recommended_actions': artifact.recommended_actions,
            'counterexamples': artifact.counterexamples,
            'scope': artifact.scope,
            'when_not_to_apply': artifact.when_not_to_apply,
            'provenance': {
                'source_ref': provenance.source_ref,
                'clustering_method': provenance.clustering_method,
                'total_analyzed': provenance.total_analyzed,
                'supporting_count': provenance.supporting_count,
                'cluster_category': provenance.cluster_category,
                'timestamp': provenance.timestamp,
                'metadata': provenance.metadata,
            },
        }
        with self._path.open('a') as f:
            f.write(json.dumps(record) + '\n')
        return sink_id


class InMemoryDeduper:
    """Title-based case-insensitive deduplication for testing."""

    def __init__(self) -> None:
        self._seen: set = set()

    async def is_duplicate(self, artifact: LearningArtifact) -> bool:
        key = artifact.title.lower().strip()
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    def reset(self) -> None:
        self._seen.clear()


class EmbeddingDeduper:
    """
    Embedding-based cosine similarity deduplication.

    Requires ``axion[embeddings]`` extra. Raises ``ImportError`` with a clear
    message if dependencies are unavailable (NO silent fallback).
    """

    def __init__(
        self,
        embed_model=None,
        embed_model_name: str = 'text-embedding-3-small',
        similarity_threshold: float = 0.85,
        max_stored: int = 1000,
        reset_per_run: bool = True,
    ) -> None:
        try:
            import numpy as np  # noqa: F401
            from openai import OpenAI  # noqa: F401
        except ImportError:
            raise ImportError(
                'EmbeddingDeduper requires axion[embeddings]. '
                'Install with: pip install axion[embeddings]'
            )

        self._embed_model = embed_model
        self._embed_model_name = embed_model_name
        self._similarity_threshold = similarity_threshold
        self._max_stored = max_stored
        self.reset_per_run = reset_per_run

        self._embeddings: List[list] = []
        self._titles: List[str] = []

    async def is_duplicate(self, artifact: LearningArtifact) -> bool:
        import numpy as np

        text = artifact.title + ' ' + artifact.content
        embedding = await self._embed(text)

        for stored in self._embeddings:
            similarity = self._cosine_similarity(embedding, stored)
            if similarity >= self._similarity_threshold:
                return True

        self._embeddings.append(embedding)
        self._titles.append(artifact.title)

        # LRU eviction
        if len(self._embeddings) > self._max_stored:
            self._embeddings.pop(0)
            self._titles.pop(0)

        return False

    async def _embed(self, text: str) -> list:
        if self._embed_model is not None:
            result = await self._embed_model.aembed_query(text)
            return result

        from openai import OpenAI

        client = OpenAI()
        response = client.embeddings.create(
            model=self._embed_model_name, input=text
        )
        return response.data[0].embedding

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        import numpy as np

        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def reset(self) -> None:
        self._embeddings.clear()
        self._titles.clear()
