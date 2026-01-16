from typing import Dict, List

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

from axion._core.logging import get_logger
from axion._handlers.knowledge.loaders.base import BaseDocumentLoader

logger = get_logger(__name__)


class CustomReaderLoader(BaseDocumentLoader):
    def __init__(self, reader: BaseReader, load_kwargs: Dict = None, **kwargs):
        super().__init__(kwargs)
        self.reader = reader
        self.load_kwargs = load_kwargs or {}

    async def execute(self) -> List[Document]:
        def _load_sync():
            return self.reader.load_data(**self.load_kwargs)

        documents = await self.executor.run(_load_sync)
        logger.info(
            f'Loaded {len(documents)} documents using {self.reader.__class__.__name__}'
        )
        return self._add_loader_metadata(documents)
