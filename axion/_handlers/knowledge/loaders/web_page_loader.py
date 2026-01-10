from typing import List

from axion._core.logging import get_logger
from axion._handlers.knowledge.loaders.base import BaseDocumentLoader
from llama_index.core import Document

logger = get_logger(__name__)


class WebPageLoader(BaseDocumentLoader):
    def __init__(self, urls: List[str], **kwargs):
        super().__init__(kwargs)
        self.urls = urls

    async def execute(self) -> List[Document]:
        from llama_index.readers.web import SimpleWebPageReader

        def _load_sync():
            return SimpleWebPageReader().load_data(self.urls)

        documents = await self.executor.run(_load_sync)
        logger.info(f'Loaded {len(documents)} documents from {len(self.urls)} URLs')
        return self._add_loader_metadata(documents)
