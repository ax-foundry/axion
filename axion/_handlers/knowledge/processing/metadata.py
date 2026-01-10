from typing import Any, List

from axion._handlers.knowledge.processing.transformations import (
    BaseTransformComponent,
)
from llama_index.core.schema import Node


class MetadataOptimizer(BaseTransformComponent):
    """
    Optimizes nodes by excluding specified metadata keys from embedding and LLM calls.
    This helps reduce embedding cost/noise and focuses the LLM on relevant context.
    """

    embed_exclude_keys: List[str] = []
    llm_exclude_keys: List[str] = []

    @classmethod
    def class_name(cls) -> str:
        return 'MetadataOptimizer'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        for node in nodes:
            # Use sets to avoid adding duplicate keys
            node.excluded_embed_metadata_keys = list(
                set(node.excluded_embed_metadata_keys) | set(self.embed_exclude_keys)
            )
            node.excluded_llm_metadata_keys = list(
                set(node.excluded_llm_metadata_keys) | set(self.llm_exclude_keys)
            )
        print(f'Optimized metadata for {len(nodes)} nodes.')
        return nodes
