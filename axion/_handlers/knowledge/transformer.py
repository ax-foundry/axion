import asyncio
from typing import Any, Dict, List, Optional, Type, Union

from axion._core.error import InvalidConfig
from axion._core.logging import get_logger
from axion._core.utils import Timer
from axion._handlers.knowledge.processing.metadata import MetadataManagementMixin
from axion._handlers.knowledge.processing.strategies import PipelineStrategy
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import MetadataMode, Node, TextNode, TransformComponent

logger = get_logger(__name__)


class DocumentTransformer(MetadataManagementMixin):
    """
    High-performance document processing transformer with metadata management.

    This class provides a clean, extensible way to process documents using different strategies
    for different content types. It includes metadata management through the MetadataManagementMixin.

    Examples:
        # Basic usage with metadata management
        transformer = DocumentTransformer(
            embed_model=embed_model,
            transformations=[SentenceSplitter(chunk_size=512)],
            normalize_metadata=True,
            excluded_embed_metadata_keys=['file_path']
        )
        nodes = await transformer.execute(documents)

        # Method chaining
        transformer = (DocumentTransformer(embed_model=embed_model)
                      .set_transformations([SentenceSplitter(chunk_size=512)])
                      .enable_metadata_normalization(True)
                      .exclude_metadata_for_embedding(['file_path'])
                      .execute(documents))
    """

    _metrics: Dict
    DEFAULT_BATCH_SIZE = 100

    # Enhancements
    #  - metadata normalization should be option for pre and post
    #  - metadata must all be same type (so string doesn't work for all)

    def __init__(
        self,
        transformations: Optional[List[TransformComponent]] = None,
        strategy: Optional[PipelineStrategy] = None,
        num_workers: Optional[int] = None,
        llm=None,
        embed_model=None,
        enable_cache: bool = True,
        request_metrics: bool = True,
        excluded_llm_metadata_keys: List = None,
        excluded_embed_metadata_keys: List = None,
        include_llm_metadata_keys: List = None,
        include_embed_metadata_keys: List = None,
        text_template: str = None,
        normalize_metadata: bool = False,
        metadata_fill_value: str = '',
        **kwargs,
    ):
        """
        Initialize DocumentTransformer with metadata management.

        Args:
            transformations: Optional list of transformations to use directly
            strategy: Optional pipeline strategy
            llm: Language model for extractors
            embed_model: Embedding model for vectorization
            num_workers: Number of parallel workers
            enable_cache: Whether to enable IngestionPipeline caching
            request_metrics: Request performance metrics along with nodes

            # Metadata management parameters
            excluded_llm_metadata_keys: List of metadata keys to exclude for LLM
            excluded_embed_metadata_keys: List of metadata keys to exclude for embedding
            include_llm_metadata_keys: List of metadata keys to include for LLM (exclusion of others)
            include_embed_metadata_keys: List of metadata keys to include for embedding (exclusion of others)
            text_template: Custom text template for documents
            normalize_metadata: Whether to normalize metadata keys across all nodes
            metadata_fill_value: Value to use for missing metadata keys
        """
        MetadataManagementMixin.__init__(
            self,
            excluded_llm_metadata_keys=excluded_llm_metadata_keys,
            excluded_embed_metadata_keys=excluded_embed_metadata_keys,
            include_llm_metadata_keys=include_llm_metadata_keys,
            include_embed_metadata_keys=include_embed_metadata_keys,
            text_template=text_template,
            normalize_metadata=normalize_metadata,
            metadata_fill_value=metadata_fill_value,
        )

        if strategy is not None and transformations is not None:
            raise ValueError(
                "Cannot provide both 'strategy' and 'transformations'. Choose one."
            )

        self.llm = llm
        self.embed_model = embed_model
        self.pipeline_strategy = strategy
        self.transformations = transformations
        self._pipeline = None
        self.num_workers = num_workers
        self.enable_cache = enable_cache
        self.request_metrics = request_metrics

        if transformations:
            self._create_pipeline_from_transformations(
                transformations, 'custom transformations'
            )
        elif strategy:
            strategy_transformations = strategy.create_pipeline()
            self._create_pipeline_from_transformations(
                strategy_transformations, f'strategy: {strategy.get_strategy_name()}'
            )

    def _create_pipeline_from_transformations(
        self, transformations: List[TransformComponent], source: str = 'transformations'
    ) -> None:
        """Helper function to create pipeline from transformations with embed_model handling."""
        final_transformations = self._ensure_embedding_last(transformations)
        self._pipeline = IngestionPipeline(transformations=final_transformations)
        self._configure_pipeline()
        logger.info(
            f'Created pipeline from {source} ({len(final_transformations)} transformations)'
        )

    def _ensure_embedding_last(
        self, transformations: List[TransformComponent]
    ) -> List[TransformComponent]:
        """Ensure that the embed_model is the last transformation if provided."""
        if not self.embed_model:
            return transformations
        cleaned = [
            t for t in transformations if not isinstance(t, type(self.embed_model))
        ]
        return cleaned + [self.embed_model]

    def _configure_pipeline(self):
        """Configure pipeline settings for optimal performance."""
        if self._pipeline:
            self._pipeline.disable_cache = not self.enable_cache
            logger.info(
                f"Configured pipeline - Cache: {'enabled' if self.enable_cache else 'disabled'}"
            )

    def set_strategy(
        self, strategy: Union[PipelineStrategy, str, Type[PipelineStrategy]], **kwargs
    ) -> 'DocumentTransformer':
        """Set the pipeline strategy and build the pipeline from transformations."""
        if isinstance(strategy, PipelineStrategy):
            self.pipeline_strategy = strategy
        elif isinstance(strategy, type) and issubclass(strategy, PipelineStrategy):
            self.pipeline_strategy = strategy(
                llm=self.llm, embed_model=self.embed_model
            )
        else:
            raise ValueError(
                f'Expected PipelineStrategy instance or class, got {type(strategy)}'
            )

        transformations = self.pipeline_strategy.create_pipeline(**kwargs)
        self._create_pipeline_from_transformations(
            transformations, f'strategy: {self.pipeline_strategy.get_strategy_name()}'
        )
        return self

    def set_transformations(
        self, transformations: List[TransformComponent]
    ) -> 'DocumentTransformer':
        """Set transformations directly and build pipeline."""
        self.transformations = transformations
        self.pipeline_strategy = None
        self._create_pipeline_from_transformations(
            transformations, 'custom transformations'
        )
        return self

    def use_pipeline(self, pipeline: IngestionPipeline) -> 'DocumentTransformer':
        """Use a pre-built pipeline directly."""
        self._pipeline = pipeline
        self.pipeline_strategy = None
        self.transformations = None
        self._configure_pipeline()
        logger.info('Using custom pre-built pipeline')
        return self

    def set_parallel_processing(
        self, num_workers: Optional[int]
    ) -> 'DocumentTransformer':
        """Configure parallel processing settings."""
        self.num_workers = num_workers
        logger.info(
            f"Set parallel processing: {num_workers if num_workers else 'sequential'}"
        )
        return self

    def set_caching(self, enable_cache: bool) -> 'DocumentTransformer':
        """Configure pipeline caching."""
        self.enable_cache = enable_cache
        if self._pipeline:
            self._pipeline.disable_cache = not enable_cache
        logger.info(f"Set caching: {'enabled' if enable_cache else 'disabled'}")
        return self

    async def execute(
        self,
        documents: List[Document],
        show_progress: bool = True,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[Node]:
        """
        Process documents using the configured pipeline strategy.

        Processing Order:
        1. Pre-processing transformations (chunking, extraction, etc.) - sees all metadata
        2. Apply metadata normalization - ensures consistent metadata structure
        3. Apply metadata exclusions - filter what models see
        4. LLM/Embedding operations - only see filtered metadata

        Args:
            documents: List of documents to process
            show_progress: Display progress bar during processing
            num_workers: Number of workers for parallel execution
            batch_size: Batch size for processing

        Returns:
            List of processed Nodes
        """
        if not self._pipeline:
            raise InvalidConfig('No pipeline set.')

        workers = num_workers or self.num_workers
        batch = batch_size or self.DEFAULT_BATCH_SIZE
        use_batching = len(documents) > batch

        logger.info(
            f"Processing {len(documents)} documents "
            f"[workers: {workers or 'sequential'}, "
            f"batching: {'yes' if use_batching else 'no'}, "
            f""
            f"metadata_config: {self._has_metadata_config()}]"
        )

        with Timer() as timer:
            mem_before = self._get_memory_usage() if self.request_metrics else 0

            if self._has_metadata_config():
                if use_batching:
                    nodes = await self._process_with_batching_and_metadata(
                        documents, workers, batch, show_progress
                    )
                else:
                    nodes = await self._process_direct_with_metadata(
                        documents, workers, show_progress
                    )
            else:
                if use_batching:
                    nodes = await self._process_with_batching(
                        documents, workers, batch, show_progress
                    )
                else:
                    nodes = await self._process_direct(
                        documents, workers, show_progress
                    )

            mem_after = self._get_memory_usage() if self.request_metrics else 0

        duration = timer.elapsed_time
        docs_per_sec = len(documents) / duration if duration > 0 else 0

        if self.request_metrics:
            self._metrics = {
                'documents_processed': len(documents),
                'nodes_created': len(nodes),
                'processing_time_seconds': round(duration, 2),
                'documents_per_second': round(docs_per_sec, 2),
                'memory_used_mb': round(mem_after - mem_before, 2),
                'workers_used': workers,
                'batching_used': use_batching,
                'batch_size': batch if use_batching else None,
                'strategy': (
                    self.pipeline_strategy.get_strategy_name()
                    if self.pipeline_strategy
                    else 'Custom'
                ),
                'metadata_config': self.get_metadata_config(),
            }
            logger.info(
                f"Processing complete: {self.metrics['documents_per_second']:.1f} docs/sec, "
                f""
                f"{self.metrics['memory_used_mb']:.1f}MB memory used"
            )
        else:
            logger.info(
                f'Processed {len(documents)} → {len(nodes)} nodes in {duration:.2f}s '
                f'({docs_per_sec:.1f} docs/sec)'
            )

        return nodes

    async def _process_direct(
        self, documents: List[Document], workers: Optional[int], show_progress: bool
    ) -> List[Node]:
        """Direct processing without batching (no metadata config)."""
        return await self._pipeline.arun(
            documents=documents, show_progress=show_progress, num_workers=workers
        )

    async def _process_with_batching(
        self,
        documents: List[Document],
        workers: Optional[int],
        batch_size: int,
        show_progress: bool,
    ) -> List[Node]:
        """Process documents concurrently in batches (no metadata config)."""
        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]
        logger.info(
            f'Processing {len(documents)} documents in '
            f'{len(batches)} batches (~{batch_size} each)'
        )

        results = await asyncio.gather(
            *[self._process_direct(batch, workers, show_progress) for batch in batches]
        )
        return [node for batch in results for node in batch]

    async def _process_direct_with_metadata(
        self, documents: List[Document], workers: Optional[int], show_progress: bool
    ) -> List[Node]:
        """Process documents with metadata configuration applied after transformations."""
        try:
            return await self._process_with_metadata_management(
                documents, self._pipeline, workers, show_progress
            )
        except RuntimeError:
            raise RuntimeError(
                "Your model or tensors are on GPU, which isn't supported. "
                'Either enforce CPU embeddings or set num_workers=1'
            )

    async def _process_with_batching_and_metadata(
        self,
        documents: List[Document],
        workers: Optional[int],
        batch_size: int,
        show_progress: bool,
    ) -> List[Node]:
        """Process documents in batches with metadata configuration."""
        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]
        logger.info(
            f'Processing {len(documents)} documents in {len(batches)} '
            f'batches (~{batch_size} each)'
        )

        results = await asyncio.gather(
            *[
                self._process_direct_with_metadata(batch, workers, show_progress)
                for batch in batches
            ]
        )
        return [node for batch in results for node in batch]

    @staticmethod
    def show_model_content(
        node: TextNode, metadata_mode: Union[str, MetadataMode] = MetadataMode.EMBED
    ) -> str:
        """
        Retrieve the content of a TextNode with the specified metadata visibility.

        Args:
            node: The TextNode to inspect.
            metadata_mode: One of 'none', 'llm', 'embed', or 'all', or a MetadataMode instance.

        Returns:
            The string content of the node as seen by the model.
        """
        if isinstance(metadata_mode, str):
            metadata_mode = metadata_mode.lower()
            try:
                metadata_mode = MetadataMode(metadata_mode)
            except ValueError:
                raise ValueError(
                    f"Invalid metadata_mode: '{metadata_mode}'. Must be one of: "
                    f"{', '.join(m.value for m in MetadataMode)}"
                )

        return node.get_content(metadata_mode=metadata_mode)

    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    @property
    def pipeline(self) -> Optional[IngestionPipeline]:
        """Get the current pipeline."""
        return self._pipeline

    @property
    def strategy(self) -> Optional[PipelineStrategy]:
        """Get the current strategy."""
        return self.pipeline_strategy

    @property
    def current_transformations(self) -> Optional[List[TransformComponent]]:
        """Get the current transformations."""
        if self._pipeline:
            return self._pipeline.transformations
        return None

    @property
    def metrics(self) -> Optional[Dict]:
        """Performance Metrics"""
        return self._metrics

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get detailed information about the current pipeline and strategy."""
        if not self._pipeline:
            return {'status': 'No pipeline set'}

        source = 'Unknown'
        if self.pipeline_strategy:
            source = 'Strategy'
        elif self.transformations:
            source = 'Custom Transformations'

        info = {
            'status': 'Pipeline ready',
            'source': source,
            'strategy': {
                'name': (
                    self.pipeline_strategy.get_strategy_name()
                    if self.pipeline_strategy
                    else 'Custom'
                ),
                'class': (
                    self.pipeline_strategy.__class__.__name__
                    if self.pipeline_strategy
                    else 'N/A'
                ),
            },
            'pipeline': {
                'transformations': [
                    type(t).__name__ for t in self._pipeline.transformations
                ],
                'num_transformations': len(self._pipeline.transformations),
                'cache_enabled': not getattr(self._pipeline, 'disable_cache', False),
                'cache_collection': getattr(self._pipeline, 'cache_collection', None),
            },
            'configuration': {
                'has_llm': self.llm is not None,
                'has_embed_model': self.embed_model is not None,
                'num_workers': self.num_workers,
                'parallel_processing': self.num_workers is not None
                and self.num_workers > 1,
                'cache_enabled': self.enable_cache,
            },
            'metadata': self.get_metadata_config(),
        }
        return info
