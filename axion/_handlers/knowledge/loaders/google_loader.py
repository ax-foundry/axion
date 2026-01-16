from typing import List, Optional

from llama_index.core import Document

from axion._core.environment import settings
from axion._core.logging import get_logger
from axion._handlers.knowledge.loaders.base import BaseDocumentLoader

logger = get_logger(__name__)


class BaseGoogleLoader(BaseDocumentLoader):
    """Base class for all Google API loaders with common credential handling."""

    def __init__(self, credentials_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.credentials = self._load_google_credentials(credentials_path)

    @staticmethod
    def _load_google_credentials(credentials_path: Optional[str] = None):
        """
        Load Google service account credentials from a JSON file.
        """
        from google.oauth2 import service_account

        if not credentials_path:
            credentials_path = settings.google_credentials_path

        if not credentials_path:
            raise ValueError(
                'credentials_path is required for Google API authentication. '
                'Provide it as a parameter or set GOOGLE_CREDENTIALS_PATH environment variable'
            )

        return service_account.Credentials.from_service_account_file(credentials_path)

    def _set_document_ids(self, docs: List[Document]):
        """
        Set document IDs based on metadata keys in order of preference.
        """
        for doc in docs:
            # Try each metadata key in order until we find one that exists
            for key in self.id_metadata_keys:
                if key in doc.metadata and doc.metadata[key]:
                    doc.id_ = doc.metadata[key]
                    break


class GoogleDriveLoader(BaseGoogleLoader):
    """
    Loader for Google Drive files and folders.
    """

    id_metadata_keys = ['file_name', 'file_id', 'title']

    def __init__(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        credentials_path: Optional[str] = None,
        id_metadata_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize Google Drive loader.

        Args:
            folder_id (str, optional): Google Drive folder ID to load all files from
            file_ids (List[str], optional): List of specific Google Drive file IDs to load
            credentials_path (str, optional): Path to service account JSON file
            id_metadata_keys (List[str], optional): Metadata keys to use for document IDs
        """
        self.folder_id = folder_id
        self.file_ids = file_ids or []
        self.id_metadata_keys = id_metadata_keys or self.id_metadata_keys
        super().__init__(credentials_path=credentials_path, **kwargs)

    async def execute(self) -> List[Document]:
        """Load documents from Google Drive."""
        from llama_index.readers.google import GoogleDriveReader

        loader = GoogleDriveReader(credentials=self.credentials)
        docs = (
            await loader.aload_data(folder_id=self.folder_id)
            if self.folder_id
            else await loader.aload_data(file_ids=self.file_ids)
        )
        self._set_document_ids(docs)

        logger.info(f'Loaded {len(docs)} documents from Google Drive')
        return self._add_loader_metadata(docs)


class GoogleDocLoader(BaseGoogleLoader):
    """
    Loader for Google Docs documents.
    """

    id_metadata_keys = ['document_id', 'doc_id', 'title']

    def __init__(
        self,
        document_ids: Optional[List[str]] = None,
        credentials_path: Optional[str] = None,
        id_metadata_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize Google Docs loader.

        Args:
            document_ids (List[str], optional): List of Google Docs document IDs to load
            credentials_path (str, optional): Path to service account JSON file
            id_metadata_keys (List[str], optional): Metadata keys to use for document IDs
        """
        self.document_ids = document_ids or []
        self.id_metadata_keys = id_metadata_keys or self.id_metadata_keys
        super().__init__(credentials_path=credentials_path, **kwargs)

    async def execute(self) -> List[Document]:
        """Load documents from Google Docs."""
        from llama_index.readers.google import GoogleDocsReader

        loader = GoogleDocsReader(credentials=self.credentials)
        docs = await loader.aload_data(document_ids=self.document_ids)
        self._set_document_ids(docs)

        logger.info(f'Loaded {len(docs)} documents from Google Docs')
        return self._add_loader_metadata(docs)


class GoogleSheetsLoader(BaseGoogleLoader):
    """
    Loader for Google Sheets spreadsheets.
    """

    id_metadata_keys = ['sheet_id', 'spreadsheet_id', 'title']

    def __init__(
        self,
        sheet_ids: Optional[List[str]] = None,
        credentials_path: Optional[str] = None,
        id_metadata_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize Google Sheets loader.

        Args:
            sheet_ids (List[str], optional): List of Google Sheets spreadsheet IDs to load
            credentials_path (str, optional): Path to service account JSON file
            id_metadata_keys (List[str], optional): Metadata keys to use for document IDs
        """
        self.sheet_ids = sheet_ids or []
        self.id_metadata_keys = id_metadata_keys or self.id_metadata_keys
        super().__init__(credentials_path=credentials_path, **kwargs)

    async def execute(self) -> List[Document]:
        """Load documents from Google Sheets."""
        from llama_index.readers.google import GoogleSheetsReader

        loader = GoogleSheetsReader(credentials=self.credentials)
        docs = await loader.aload_data(self.sheet_ids)
        self._set_document_ids(docs)

        logger.info(f'Loaded {len(docs)} documents from Google Sheets')
        return self._add_loader_metadata(docs)
