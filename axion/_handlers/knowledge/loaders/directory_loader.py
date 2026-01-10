from typing import Any, Dict, List, Optional

from axion._core.logging import get_logger
from axion._handlers.knowledge.loaders.base import BaseDocumentLoader
from llama_index.core import Document, SimpleDirectoryReader

logger = get_logger(__name__)


class DirectoryLoader(BaseDocumentLoader):
    """
    Loads documents from local directories or S3 paths.

    Examples:
        # Basic usage
        loader = DirectoryLoader("./documents")
        documents = await loader.execute()

        # With LlamaParse
        loader = DirectoryLoader(
            "./documents",
            reader_type="docling",
            api_key="your-key"
        )

        # S3 path
        loader = DirectoryLoader("s3://bucket/docs")
    """

    def __init__(
        self,
        directory_path: str,
        reader_type: str = 'docling',
        api_key: Optional[str] = None,
        file_extractor: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the DirectoryLoader.

        Args:
            directory_path: Path to local directory or S3 URI
            reader_type: Type of reader ("docling", "llama_parse", "unstructured")
            api_key: API key for readers that require it
            file_extractor: Custom file extractor mapping
            **kwargs: Additional configuration for SimpleDirectoryReader
        """
        super().__init__(kwargs)

        self.directory_path = directory_path
        self.reader_type = reader_type
        self.api_key = api_key

        if file_extractor:
            self.file_extractor = file_extractor
        else:
            self.file_extractor = self._create_file_extractor()

    def _create_reader_instance(self) -> Any:
        """Create reader instance based on type."""

        if self.reader_type == 'docling':
            from llama_index.readers.docling import DoclingReader

            return DoclingReader()
        else:
            raise ValueError(
                f'Unsupported reader type: {self.reader_type}. '
                'Currently only docling supported. Stay tuned'
            )

    def _create_file_extractor(self) -> Dict[str, Any]:
        """Create file extractor for common file types."""
        reader = self._create_reader_instance()
        return {
            '.pdf': reader,
            '.docx': reader,
            '.html': reader,
            '.jpg': reader,
            '.png': reader,
        }

    async def execute(self) -> List[Document]:
        """Execute the document loading process."""
        logger.info(f'Loading documents from: {self.directory_path}')

        # Configure reader arguments
        reader_kwargs = {
            'input_dir': self.directory_path,
            'file_extractor': self.file_extractor,
            **self.loader_config,
        }

        # S3 filesystem support
        if self.directory_path.startswith('s3://'):
            import s3fs

            reader_kwargs['fs'] = s3fs.S3FileSystem()

        # Load documents
        reader = SimpleDirectoryReader(**reader_kwargs)
        documents = await reader.aload_data()

        logger.info(f'Loaded {len(documents)} documents')
        return self._add_loader_metadata(documents)
