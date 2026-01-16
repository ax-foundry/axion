import asyncio
from typing import List, Optional

from llama_index.core import Document

from axion._core.logging import get_logger
from axion._handlers.knowledge.loaders.base import BaseDocumentLoader

logger = get_logger(__name__)


class Ingestion:
    def __init__(
        self,
        loaders: Optional[List[BaseDocumentLoader]] = None,
    ):
        self.loaders = loaders or []

    def add_loader(self, loader: BaseDocumentLoader) -> None:
        """Add loader."""
        self.loaders.append(loader)

    async def execute(self) -> List[Document]:
        if not self.loaders:
            raise ValueError('No loaders configured')

        logger.info(f'Loading documents from {len(self.loaders)} loaders')
        tasks = [loader.execute() for loader in self.loaders]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f'Loader {i} failed: {result}')
                continue
            all_documents.extend(result)

        logger.info(f'Successfully loaded {len(all_documents)} total documents')
        return all_documents
