from typing import List, Literal, Optional, Union

from llama_index.core import Document
from llama_index.core.node_parser import (
    NodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)

from axion._core.logging import get_logger
from axion._core.schema import EmbeddingRunnable
from axion._core.tracing import init_tracer, trace
from axion._core.tracing.handlers import BaseTraceHandler
from axion._handlers.knowledge.processing.transformations import (
    DocumentTextPreprocessor,
)

logger = get_logger(__name__)


class DocumentProcessor:
    """Splits documents into semantically coherent chunks using a LlamaIndex node parser and preprocessing pipeline."""

    def __init__(
        self,
        splitter_type: Literal['semantic', 'sentence'] = 'sentence',
        embed_model: Optional[EmbeddingRunnable] = None,
        chunk_size: int = 4000,
        tracer: Optional[BaseTraceHandler] = None,
        **kwargs,
    ):
        """
        Initialize the DocumentChunker with a splitter and optional preprocessing configuration.

        Args:
            splitter_type (str): Type of splitter to use: "semantic" or "sentence".
            embed_model (Optional[Any]): Required if using semantic splitter.
            chunk_size (int): Chunk size for sentence splitter.
            **kwargs: Additional arguments passed to the splitter if needed.
        """
        self.splitter: NodeParser
        self.tracer = init_tracer('llm', tracer=tracer)

        if splitter_type == 'semantic':
            if embed_model is None:
                raise ValueError(
                    'An `embed_model` must be provided for semantic splitting.'
                )
            self.splitter = SemanticSplitterNodeParser(
                buffer_size=kwargs.get('buffer_size', 1),
                breakpoint_percentile_threshold=kwargs.get(
                    'breakpoint_percentile_threshold', 95
                ),
                embed_model=embed_model,
            )
        elif splitter_type == 'sentence':
            self.splitter = SentenceSplitter(
                chunk_size=chunk_size, chunk_overlap=kwargs.get('chunk_overlap', 200)
            )
        else:
            raise ValueError(
                f"Unsupported splitter_type: {splitter_type}. Use 'semantic' or 'sentence'."
            )

        self.text_processor = DocumentTextPreprocessor()

    @trace
    async def process(self, document: Union[str, Document]) -> List[str]:
        """
        Preprocess and split a document into semantically meaningful chunks.

        Args:
            document (Union[str, Document]): Raw string or Document instance.

        Returns:
            List[str]: A list of cleaned and split text chunks.
        """
        if not document or (isinstance(document, str) and not document.strip()):
            return []

        # Wrap raw string into Document
        doc_obj = (
            document if isinstance(document, Document) else Document(text=document)
        )

        docs = await self.text_processor.execute([doc_obj])
        nodes = await self.splitter.aget_nodes_from_documents(docs, show_progress=False)
        return [node.text for node in nodes]
