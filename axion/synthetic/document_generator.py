import asyncio
from typing import Any, Dict, List, Optional, Union

from axion._core.asyncio import SemaphoreExecutor
from axion._core.logging import configure_logging, get_logger
from axion._core.metadata.schema import ToolMetadata
from axion._core.schema import EmbeddingRunnable, LLMRunnable
from axion._core.tracing import init_tracer, trace
from axion._core.tracing.handlers import BaseTraceHandler
from axion.synthetic.schema import GenerationParams
from axion.synthetic.workflow import QAWorkflowGraph
from axion._handlers.knowledge.ingestion import Ingestion
from axion._handlers.knowledge.loaders import DirectoryLoader
from llama_index.core import Document

# Set logging for error for bulk
logger = get_logger(__name__)
configure_logging(level='ERROR', use_rich=False)


class DocumentQAGenerator:
    """Orchestrates QA pair generation from multiple documents concurrently."""

    def __init__(
        self,
        llm: LLMRunnable,
        params: GenerationParams,
        embed_model: EmbeddingRunnable = None,
        max_concurrent: int = 5,
        show_progress: bool = True,
        tracer: Optional[BaseTraceHandler] = None,
        **kwargs,
    ):
        """
        Initialize DocumentQAGenerator

        Args:
            llm: The language model to use for generation.
            params: A GenerationParams object with all configuration.
            embed_model: An embedding model used for semantic parsing.
            max_concurrent: Max concurrent retrievers
            show_progress: Whether to show progress bars using tqdm
        """
        self.llm = llm
        self.params = params
        self.embed_model = embed_model
        self.executor = SemaphoreExecutor(max_concurrent=max_concurrent)
        self.show_progress = show_progress
        self.all_qa_pairs: List[Dict[str, Any]] = []
        self.all_statements: List[str] = []
        self.tracer = init_tracer('llm', self._get_tool_metadata(), tracer)
        self.workflow = QAWorkflowGraph(
            llm=self.llm, embed_model=embed_model, tracer=self.tracer, **kwargs
        )

    def _get_tool_metadata(self):
        return ToolMetadata(
            name=self.__class__.__name__,
            description='Orchestrates QA pair generation from multiple documents concurrently.',
            owner='AI Toolkit',
            version='1.0.0',
        )

    @trace(name='process_single_document')
    async def _process_single_document(
        self, content: Union[str, Document], params: GenerationParams
    ) -> Optional[Dict[str, Any]]:
        """Runs the full QA workflow for a single document's content."""
        if isinstance(content, Document):
            file_name = content.metadata.get('file_name', 'unknown_file')
        else:
            file_name = 'unknown_file'

        logger.info(f'Starting QA generation for document: {file_name}')
        try:
            # Convert Pydantic model to dict for config
            config_kwargs = params.model_dump()
            final_state = await self.workflow.run_from_documents(
                content, **config_kwargs
            )

            if final_state.get('processing_errors'):
                logger.error(
                    f"Errors occurred while processing {file_name}: {final_state['processing_errors']}"
                )
                return None

            qa_pairs = final_state.get('validated_qa_pairs', [])
            for pair in qa_pairs:
                pair['source_document'] = file_name

            logger.info(
                f'Successfully generated {len(qa_pairs)} QA pairs from {file_name}.'
            )
            return final_state
        except Exception as e:
            logger.error(f'Failed to process document {file_name}: {e}', exc_info=True)
            return None

    @trace(name='generate_from_documents')
    async def _generate_from_documents(
        self,
        documents: Union[List[Document], List[str]],
    ) -> List[Dict[str, Any]]:
        """Process a list of documents asynchronously using an executor."""

        # TODO -- should pass on failures
        tasks = [
            self.executor.run(self._process_single_document, doc, self.params)
            for doc in documents
        ]

        async def run_with_progress() -> List[Dict[str, Any]]:
            try:
                from tqdm.asyncio import tqdm

                return await tqdm.gather(
                    *tasks,
                    desc='🚀 Processing documents for QA generation',
                    total=len(tasks),
                )
            except ImportError:
                logger.warning(
                    'tqdm not available, falling back to asyncio.gather without progress bar'
                )
                return await asyncio.gather(*tasks)

        return (
            await run_with_progress()
            if self.show_progress
            else await asyncio.gather(*tasks)
        )

    async def generate_from_directory(
        self, directory_path: str
    ) -> List[Dict[str, Any]]:
        """
        Main entry point. Loads docs from a directory and generates QA pairs.
        Args:
            directory_path: The path to the directory containing documents.
        Returns:
            A list of all generated QA pairs.
        """
        async with self.tracer.async_span('generate_from_directory') as span:
            span.set_attribute('parameters', self.params.to_dict())

            ingestion = Ingestion(
                loaders=[DirectoryLoader(directory_path=directory_path)]
            )

            # Add progress for document loading if desired
            if self.show_progress:
                print(f"📁 Loading documents from '{directory_path}'...")

            documents = await ingestion.execute()
            logger.info(
                f"Loaded {len(documents)} documents from '{directory_path}'. Starting generation."
            )

            if self.show_progress:
                print(
                    f'✅ Loaded {len(documents)} documents. Starting QA generation...'
                )

            return await self._generate_from_documents(documents)

    def to_items(self, results: List[Any]) -> List:
        """
        Converts a list of QA evaluation results into a List of `DatasetItems`.

        Args:
            results (List[Any]): A list of result dictionaries, each containing a 'qa_pairs' list.

        Returns:
            List: A list of DatasetItem objects.
        """

        from axion.dataset import DatasetItem
        from axion._core.types import FieldNames

        items: List[DatasetItem] = []

        for result in results:
            statements = result.get('statements', [])
            content = result.get('content', '')

            for pair in result.get(self.workflow.PAIRS_NAME, []):
                indices = pair.get('statement_indices', [])
                expected_reference = [
                    statements[i] for i in indices if i < len(statements)
                ]

                item_data = {
                    FieldNames.QUERY: pair.get(self.workflow.QUESTION_NAME, ''),
                    FieldNames.EXPECTED_OUTPUT: pair.get(self.workflow.ANSWER_NAME, ''),
                    FieldNames.EXPECTED_REFERENCE: expected_reference,
                    FieldNames.DOCUMENT_TEXT: content,
                }
                items.append(DatasetItem(**item_data))

        return items

    def to_dataset(self, results: List[Any], dataset_name: str):
        """
        Converts a list of QA evaluation results into a structured `Dataset` object.

        The function extracts these pairs, renames the fields to match internal
        `FieldNames` standards, and wraps each into a `DatasetItem`. These are then
        collected into a `Dataset` for downstream evaluation or analysis.

        Args:
            results (List[Any]): A list of result dictionaries, each containing a 'qa_pairs' list.
            dataset_name (str): The name to assign to the resulting Dataset.

        Returns:
            Dataset: A structured Dataset containing DatasetItems with standardized field names.
        """
        from axion.dataset import Dataset

        items = self.to_items(results)
        return Dataset(items=items, name=dataset_name)
