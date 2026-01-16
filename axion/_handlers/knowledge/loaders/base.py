from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.core import Document

from axion._core.asyncio import SemaphoreExecutor


class BaseDocumentLoader(ABC):
    """Abstract base class for document loading strategies."""

    def __init__(
        self,
        loader_config: Optional[Dict[str, Any]] = None,
    ):
        self.loader_config = loader_config or {}
        self.executor = SemaphoreExecutor()

    @abstractmethod
    async def execute(self) -> List[Document]:
        pass

    def _add_loader_metadata(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            doc.metadata.update(
                {
                    'loader_type': self.__class__.__name__,
                }
            )
        return documents
