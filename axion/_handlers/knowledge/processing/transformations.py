import asyncio
import html
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from axion._core.logging import get_logger
from axion._core.schema import RichEnum
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    NodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import Document, Node, TransformComponent
from pydantic import Field
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger(__name__)


class BaseTransformComponent(TransformComponent, ABC):
    """
    An abstract base class for creating custom transform components.

    This base class handles the boilerplate for both synchronous and asynchronous
    execution, and includes a built-in logger instance. Subclasses only need
    to implement the core async logic in the `execute` method.
    """

    def __init__(self, **data: Any):
        super().__init__(**data)

    @abstractmethod
    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        """
        Asynchronously transform a list of nodes.
        Implement the core transformation logic in this method.
        """
        pass

    def __call__(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        """
        Synchronous wrapper that runs the async `execute` method.
        """
        logger.info(
            f'Running (Sync wrapper) on {len(nodes)} for {self.class_name} nodes...'
        )
        return asyncio.run(self.execute(nodes, **kwargs))

    async def acall(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        """
        Asynchronous entry point for the pipeline.
        """
        logger.info(f'Running (Async) on {len(nodes)} for {self.class_name()} nodes...')
        return await self.execute(nodes, **kwargs)


class WhitespaceNormalizer(BaseTransformComponent):
    """Normalizes whitespace in a node's content."""

    @classmethod
    def class_name(cls) -> str:
        return 'WhitespaceNormalizer'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        for node in nodes:
            original_text = node.get_content()
            normalized_text = re.sub(r'\s+', ' ', original_text).strip()
            node.set_content(normalized_text)
        return nodes


class TextCleaner(BaseTransformComponent):
    """Cleans text by removing unwanted characters."""

    pattern: str = Field(default=r'[^0-9A-Za-z\s.,!?-]|-{3,}')
    replacement: str = Field(default='')

    @classmethod
    def class_name(cls) -> str:
        return 'TextCleaner'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        for node in nodes:
            original_text = node.get_content()
            cleaned_text = re.sub(self.pattern, self.replacement, original_text)
            node.set_content(cleaned_text)
        return nodes


class DocumentTextPreprocessor(BaseTransformComponent):
    """Preprocesses Document text before node parsing."""

    clean_text: bool = Field(default=True)
    normalize_whitespace: bool = Field(default=True)
    text_pattern: str = Field(default=r'[^0-9A-Za-z\s.,!?-]|-{3,}')
    text_replacement: str = Field(default='')

    @classmethod
    def class_name(cls) -> str:
        return 'DocumentTextPreprocessor'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        for node in nodes:
            processed_text = node.get_content()
            if self.normalize_whitespace:
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            if self.clean_text:
                processed_text = re.sub(
                    self.text_pattern, self.text_replacement, processed_text
                )
            node.set_content(processed_text)
        return nodes


class MetadataEnricher(BaseTransformComponent):
    """Adds custom metadata to nodes."""

    metadata_to_add: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def class_name(cls) -> str:
        return 'MetadataEnricher'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        for node in nodes:
            node.metadata.update(self.metadata_to_add)
        return nodes


class MetadataNormalizer(BaseTransformComponent):
    """
    Ensures all nodes have a consistent set of metadata keys.
    """

    fill_value: Any = Field(default='')

    @classmethod
    def class_name(cls) -> str:
        return 'MetadataNormalizer'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        if not nodes:
            return nodes

        # Collect all unique metadata keys from all nodes
        all_keys = set()
        for node in nodes:
            all_keys.update(node.metadata.keys())

        if not all_keys:
            return nodes

        # Ensure each node has all collected keys, filling missing ones
        for node in nodes:
            for key in all_keys:
                if key not in node.metadata:
                    node.metadata[key] = self.fill_value

        logger.info(
            f'Normalized metadata for {len(nodes)} nodes with {len(all_keys)} keys'
        )
        return nodes


class ContentFilter(BaseTransformComponent):
    """Filters nodes based on content criteria."""

    min_length: int = Field(default=10)
    max_length: Optional[int] = Field(default=None)
    required_keywords: List[str] = Field(default_factory=list)
    excluded_keywords: List[str] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return 'ContentFilter'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        filtered_nodes = []
        for node in nodes:
            text = node.get_content().lower()
            text_len = len(text)

            if text_len < self.min_length:
                continue
            if self.max_length and text_len > self.max_length:
                continue
            if self.required_keywords and not any(
                kw.lower() in text for kw in self.required_keywords
            ):
                continue
            if self.excluded_keywords and any(
                kw.lower() in text for kw in self.excluded_keywords
            ):
                continue
            filtered_nodes.append(node)

        logger.info(f'Filtered {len(nodes)} nodes to {len(filtered_nodes)} nodes')
        return filtered_nodes


class DeduplicationMethod(str, RichEnum):
    """Enumeration of available deduplication methods."""

    EXACT = 'exact'
    EMBEDDING = 'embedding'


class DuplicateRemover(BaseTransformComponent):
    """
    Removes duplicate nodes based on either exact text content match or embedding similarity.

    Supports two deduplication methods:
    1. exact: Uses exact text matching (hash-based, fast)
    2. embedding: Uses cosine similarity of embeddings (more flexible, slower)
    """

    method: DeduplicationMethod = Field(
        default=DeduplicationMethod.EXACT,
        description="Method to use for duplicate detection: 'exact' or 'embedding'",
    )
    embed_model: Optional[BaseEmbedding] = Field(
        default=None,
        description="Embedding model to use when method='embedding'. Required for embedding-based deduplication.",
    )
    similarity_threshold: float = Field(
        default=0.95,
        description='Cosine similarity threshold for embedding-based deduplication (0.0 to 1.0). Higher values are more strict.',
    )
    batch_size: int = Field(
        default=100,
        description='Batch size for embedding computation to optimize memory usage',
    )

    @classmethod
    def class_name(cls) -> str:
        return 'DuplicateRemover'

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.method == DeduplicationMethod.EMBEDDING and self.embed_model is None:
            raise ValueError(
                'embed_model is required when using embedding-based deduplication'
            )

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError('similarity_threshold must be between 0.0 and 1.0')

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        if not nodes:
            return nodes

        if self.method == DeduplicationMethod.EXACT:
            return await self._exact_deduplication(nodes)
        else:
            return await self._embedding_deduplication(nodes)

    async def _exact_deduplication(self, nodes: List[Node]) -> List[Node]:
        """Fast exact text matching using hash-based deduplication."""
        seen_texts = set()
        unique_nodes = []

        for node in nodes:
            text_hash = hash(node.get_content().strip().lower())
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_nodes.append(node)

        removed_count = len(nodes) - len(unique_nodes)
        if removed_count > 0:
            logger.info(f'Removed {removed_count} duplicate nodes using exact matching')

        return unique_nodes

    async def _embedding_deduplication(self, nodes: List[Node]) -> List[Node]:
        """Embedding-based deduplication using cosine similarity."""
        if len(nodes) <= 1:
            return nodes

        # Extract texts and compute embeddings in batches
        texts = [node.get_content().strip() for node in nodes]
        embeddings = await self._compute_embeddings_batched(texts)

        # Convert to numpy array for efficient similarity computation
        embeddings_array = np.array(embeddings)

        # Compute cosine similarity matrix efficiently
        similarity_matrix = cosine_similarity(embeddings_array)

        # Find duplicates using vectorized operations
        unique_indices = self._find_unique_indices(similarity_matrix)
        unique_nodes = [nodes[i] for i in unique_indices]

        removed_count = len(nodes) - len(unique_nodes)
        if removed_count > 0:
            logger.info(
                f'Removed {removed_count} duplicate nodes using embedding similarity '
                f'(threshold: {self.similarity_threshold})'
            )

        return unique_nodes

    async def _compute_embeddings_batched(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings in batches for memory efficiency."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = await self.embed_model.aget_text_embedding_batch(
                batch_texts
            )
            all_embeddings.extend(batch_embeddings)

            if len(batch_texts) == self.batch_size:
                logger.debug(
                    f'Processed embedding batch {i // self.batch_size + 1}/{(len(texts) - 1) // self.batch_size + 1}'
                )

        return all_embeddings

    def _find_unique_indices(self, similarity_matrix: np.ndarray) -> List[int]:
        """
        Efficiently find unique node indices using similarity matrix.

        Uses a greedy approach: for each node, if it's similar to any previously
        selected node above the threshold, skip it. Otherwise, include it.
        """
        n = similarity_matrix.shape[0]
        unique_indices = []
        is_duplicate = np.zeros(n, dtype=bool)

        for i in range(n):
            if is_duplicate[i]:
                continue

            # Check similarity with all previously selected unique nodes
            is_similar_to_unique = False
            for unique_idx in unique_indices:
                if similarity_matrix[i, unique_idx] >= self.similarity_threshold:
                    is_similar_to_unique = True
                    break

            if not is_similar_to_unique:
                unique_indices.append(i)
                # Mark all nodes similar to this one as duplicates
                similar_indices = np.where(
                    similarity_matrix[i] >= self.similarity_threshold
                )[0]
                is_duplicate[similar_indices] = True
                is_duplicate[i] = False  # Keep the current node as unique

        return unique_indices


class HTMLCleaner(BaseTransformComponent):
    """Removes HTML tags and cleans web content."""

    remove_scripts: bool = Field(default=True)
    remove_styles: bool = Field(default=True)

    @classmethod
    def class_name(cls) -> str:
        return 'HTMLCleaner'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        for node in nodes:
            cleaned_text = node.get_content()
            if self.remove_scripts:
                cleaned_text = re.sub(
                    r'<script[^>]*>.*?</script>',
                    '',
                    cleaned_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
            if self.remove_styles:
                cleaned_text = re.sub(
                    r'<style[^>]*>.*?</style>',
                    '',
                    cleaned_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
            cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            node.set_content(cleaned_text)
        return nodes


class CodeBlockExtractor(BaseTransformComponent):
    """Extracts and specially handles code blocks."""

    mark_as_code: bool = Field(default=True)

    @classmethod
    def class_name(cls) -> str:
        return 'CodeBlockExtractor'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        for node in nodes:
            if '```' in node.get_content() or '    ' in node.get_content():
                if self.mark_as_code:
                    node.metadata['content_type'] = 'code'
        return nodes


class BeautifulSoupCleaner(BaseTransformComponent):
    """A robust HTML cleaner using BeautifulSoup."""

    @classmethod
    def class_name(cls) -> str:
        return 'BeautifulSoupCleaner'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError('BeautifulSoup4 not install.')

        for node in nodes:
            soup = BeautifulSoup(node.get_content(), 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            main_content = soup.get_text(separator='\n', strip=True)
            node.set_content(main_content)
        return nodes


class TurnBasedSplitter(BaseTransformComponent):
    """A custom transformer to split chat logs by speaker turn."""

    @classmethod
    def class_name(cls) -> str:
        return 'TurnBasedSplitter'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        new_nodes = []
        for node in nodes:
            turns = re.split(r'\n(?=[A-Za-z0-9_ ]+:\s)', node.get_content().strip())
            for turn in turns:
                if not turn.strip():
                    continue
                speaker_match = re.match(r'([A-Za-z0-9_ ]+):(.*)', turn, re.DOTALL)
                if speaker_match:
                    speaker, content = speaker_match.groups()
                    new_nodes.append(
                        Document(
                            text=content.strip(),
                            metadata={
                                'speaker': speaker.strip(),
                                'source_doc_id': node.id_,
                            },
                        )
                    )
        return new_nodes


class GoogleDocParser(BaseTransformComponent):
    """
    An advanced, production-grade parser for Google Docs. This version includes
    a sophisticated table repair engine, list preservation, and robust artifact
    and character cleaning to handle real-world document messiness.
    """

    min_node_length: int = Field(
        default=25, description="Minimum content length to keep a node's text."
    )
    prepend_title: bool = Field(
        default=True, description='Prepend document title to each node for RAG context.'
    )
    remove_urls: bool = Field(
        default=False, description='Remove URLs from content using heuristic detection.'
    )

    # Pre-compiled regex patterns for performance and accuracy
    _NUMBERED_HEADER_PATTERN = re.compile(r'^\s*(\d+(\.\d+)*)\s+(.+)')
    _EXCESSIVE_NEWLINES_PATTERN = re.compile(r'\n\s*\n{2,}')
    _EMPTY_HEADER_PATTERN = re.compile(r'^\s*#{2,}\s*$', re.MULTILINE)
    _FOOTER_TABLE_PATTERN = re.compile(r'\|\s*Authored By\s*\|.*', re.DOTALL)
    _BROKEN_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(\.\)')
    _GOOGLE_ARTIFACTS_PATTERN = re.compile(
        r'Image\s*\d*\s*of\s*\d*|'
        r'\[Image:.*?\]|\[image:.*?\]|'
        r'Page\s+\d+\s+of\s+\d+',
        re.IGNORECASE,
    )

    # URL detection patterns - ordered by specificity
    _URL_PATTERNS = [
        # Markdown links: [text](url)
        re.compile(r'\[([^\]]+)\]\((?:https?://|www\.)[^\)]+\)', re.IGNORECASE),
        # Full URLs with protocol
        re.compile(r'https?://[^\s\)\]\}\>]+', re.IGNORECASE),
        # www. URLs without protocol
        re.compile(r'www\.[a-zA-Z0-9][-a-zA-Z0-9]*\.[^\s\)\]\}\>]+', re.IGNORECASE),
        # Common TLDs without www
        re.compile(
            r'\b[a-zA-Z0-9][-a-zA-Z0-9]*\.(?:com|org|net|edu|gov|io|co|ai|dev|app)\b[^\s\)\]\}\>]*',
            re.IGNORECASE,
        ),
        # Email addresses (often unwanted in RAG context)
        re.compile(
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', re.IGNORECASE
        ),
    ]

    @classmethod
    def class_name(cls) -> str:
        return 'GoogleDocParser'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        """
        Asynchronously processes all nodes through the cleaning pipeline.
        This is the required entry point for the transformation component.
        """
        cleaned_nodes = []
        for node in nodes:
            try:
                cleaned_node = self._process_node(node)
                if cleaned_node.get_content().strip():  # Only keep non-empty nodes
                    cleaned_nodes.append(cleaned_node)
            except Exception as e:
                logger.warning(
                    f"Failed to clean node from doc '{node.metadata.get('doc_id')}': {e}. Content preserved."
                )
                cleaned_nodes.append(node)

        logger.info(
            f'Processed {len(nodes)} nodes, returning {len(cleaned_nodes)} cleaned nodes.'
        )
        return cleaned_nodes

    def _process_node(self, node: Node) -> Node:
        """Processes a single node in a stateless and robust manner."""
        content = node.get_content()

        if self.prepend_title:
            title = node.metadata.get('doc_title') or node.metadata.get('title')
            if title:
                content = self._add_title_if_needed(content, title)

        cleaned_content = self._clean_content(content)

        if len(cleaned_content.strip()) >= self.min_node_length:
            node.set_content(cleaned_content)
        else:
            node.set_content('')  # Prune nodes that become too short

        return node

    def _add_title_if_needed(self, content: str, title: str) -> str:
        """Adds a title only if the content does not already begin with any form of header."""
        first_line = content.lstrip().split('\n', 1)[0].strip()
        if (
            first_line.startswith('#')
            or self._NUMBERED_HEADER_PATTERN.match(first_line)
            or self._is_likely_header(first_line)
        ):
            return content
        return f'# {title.strip()}\n\n{content}'

    def _is_likely_header(self, line: str) -> bool:
        """Helper to detect non-markdown headers (e.g., all-caps lines)."""
        words = line.split()
        return (
            1 < len(words) < 10
            and line.isupper()
            and not line.endswith(('.', ',', ';'))
        )

    def _clean_content(self, content: str) -> str:
        """Runs the improved, multi-stage cleaning pipeline."""
        content = self._initial_text_cleanup(content)
        content = self._remove_artifacts(content)
        if self.remove_urls:
            content = self._remove_urls_from_content(content)
        content = self._repair_tables(content)
        content = self._format_structure(content)
        content = self._clean_links(content)
        content = self._normalize_whitespace(content)
        return content.strip()

    def _initial_text_cleanup(self, content: str) -> str:
        """Handles low-level text issues like escaped chars and HTML entities."""
        content = html.unescape(content)
        content = content.replace("\\'", "'").replace('\\"', '"').replace('\\n', '\n')
        return content

    def _remove_artifacts(self, content: str) -> str:
        """Removes high-level structural noise like footers and empty headers."""
        content = self._GOOGLE_ARTIFACTS_PATTERN.sub('', content)
        content = self._EMPTY_HEADER_PATTERN.sub('', content)
        content = self._FOOTER_TABLE_PATTERN.sub('', content)
        return content

    def _remove_urls_from_content(self, content: str) -> str:
        """
        Removes URLs using heuristic-based detection patterns.
        Preserves link text from markdown links while removing the URL.
        """
        # First, handle markdown links specially - keep the link text
        content = self._URL_PATTERNS[0].sub(r'\1', content)

        # Then remove all other URL patterns
        for pattern in self._URL_PATTERNS[1:]:
            content = pattern.sub('', content)

        return content

    def _format_structure(self, content: str) -> str:
        """Improved to preserve existing markdown and correctly handle lists."""
        lines = content.split('\n')
        processed_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                processed_lines.append('')
                continue

            # Preserve existing markdown headers and bulleted lists
            if stripped.startswith('#') or stripped.startswith(('-', '*')):
                processed_lines.append(line)
                continue

            # Convert numbered list items that are used as headers
            numbered_match = self._NUMBERED_HEADER_PATTERN.match(stripped)
            if numbered_match:
                level_str = numbered_match.group(1)
                header_text = numbered_match.group(3).strip()
                level = min(level_str.count('.') + 2, 6)  # Start at H2
                processed_lines.append(f"{'#' * level} {header_text}")
                continue

            processed_lines.append(line)
        return '\n'.join(processed_lines)

    def _repair_tables(self, content: str) -> str:
        """
        A robust table processor that adds missing markdown separators,
        removes empty/malformed rows, and cleans up tables used for layout.
        """
        # Find all table-like blocks (consecutive lines starting with a pipe)
        table_blocks = re.findall(r'((?:^\s*\|.*\|\s*\n)+)', content, re.MULTILINE)

        for block in table_blocks:
            lines = block.strip().split('\n')
            valid_rows = []

            for line in lines:
                cells = [c.strip() for c in line.split('|')]
                # Filter out separator lines or rows where all cells are empty
                if re.match(r'^[|\s-]+$', line) or not any(
                    c for c in cells if c.strip()
                ):
                    continue

                # For layout tables, remove empty spacer columns
                meaningful_cells = [c for c in cells if c]
                if len(meaningful_cells) < len(cells) - 1:
                    # Re-add the outer empty cells required for markdown table format
                    valid_row_cells = [''] + meaningful_cells + ['']
                else:
                    valid_row_cells = cells

                valid_rows.append(' | '.join(valid_row_cells))

            if not valid_rows:
                # This entire table block was junk, so remove it from the content
                content = content.replace(block, '')
                continue

            # Inject the markdown separator line after the first valid row (the header)
            if len(valid_rows) > 0:
                num_cols = valid_rows[0].count('|') - 1
                if num_cols > 0:
                    separator = '|' + ' --- |' * num_cols
                    valid_rows.insert(1, separator)

            # Replace the old, malformed block with the cleaned, valid markdown table
            cleaned_table_str = '\n'.join(valid_rows)
            content = content.replace(
                block, cleaned_table_str + '\n\n'
            )  # Add newlines for spacing

        return content

    def _clean_links(self, content: str) -> str:
        """Removes broken markdown links."""
        # Replaces links like [text](.) with just "text"
        return self._BROKEN_LINK_PATTERN.sub(r'\1', content)

    def _normalize_whitespace(self, content: str) -> str:
        """Collapses excessive newlines and removes trailing spaces."""
        content = self._EXCESSIVE_NEWLINES_PATTERN.sub('\n\n', content)
        return re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)


class FileTypeDispatchingTransformer(BaseTransformComponent):
    """
    Applies a specific TransformComponent to nodes based on their source file type.

    This component is designed for the refinement stage of an ingestion pipeline,
    applying specialized cleaning or enrichment logic after nodes have been created.
    It uses lazy-loading to initialize transformers only when they are first needed.

    Example:
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(),  // Creates nodes
                FileTypeDispatchingTransformer( // Cleans nodes based on source
                    docx_remove_urls=True,
                    custom_transformers={
                        '.json': MyJsonCleaner(),
                    }
                )
            ]
        )
    """

    custom_transformers: Dict[str, Any] = Field(
        default_factory=dict,
        description='Override or add custom transformers per file extension. '
        'Can be instances, classes, or (class, params) tuples.',
    )
    # --- Parameters for specific default transformers ---
    docx_min_node_length: int = Field(
        25, description='Min content length for .docx nodes.'
    )
    docx_prepend_title: bool = Field(
        True, description='Prepend doc title for .docx nodes.'
    )
    docx_remove_urls: bool = Field(False, description='Remove URLs from .docx nodes.')

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Cache for instantiated transformers
        self._instance_cache: Dict[str, TransformComponent] = {}
        # Merged configuration from defaults and custom user input
        self._transformer_configs: Dict[str, Any] = self._get_default_transformers()
        self._transformer_configs.update(self.custom_transformers)

    def _get_default_transformers(self) -> Dict[str, Any]:
        """
        Defines the default, best-practice transformers for common file types.
        Users can override these via the `custom_transformers` argument.
        """
        return {
            # For Microsoft Word documents, apply the advanced Google Doc cleaner
            '.docx': (
                GoogleDocParser,
                {
                    'min_node_length': self.docx_min_node_length,
                    'prepend_title': self.docx_prepend_title,
                    'remove_urls': self.docx_remove_urls,
                },
            ),
            # For HTML, use the robust BeautifulSoup cleaner
            '.html': BeautifulSoupCleaner,
            '.htm': BeautifulSoupCleaner,
            '.xhtml': BeautifulSoupCleaner,
            # For Markdown, tag code blocks for special handling downstream
            '.md': CodeBlockExtractor,
            '.markdown': CodeBlockExtractor,
            # For common code files, also tag them as code
            '.py': CodeBlockExtractor,
            '.js': CodeBlockExtractor,
            '.ts': CodeBlockExtractor,
            '.java': CodeBlockExtractor,
            '.cs': CodeBlockExtractor,
            '.go': CodeBlockExtractor,
            '.rb': CodeBlockExtractor,
            '.php': CodeBlockExtractor,
            # For plain text, a simple whitespace normalization is sufficient
            '.txt': WhitespaceNormalizer,
        }

    def _create_transformer_instance(self, spec: Any) -> TransformComponent:
        """
        Creates a transformer instance from its configuration specification.
        The spec can be an instance, a class, or a (class, params_dict) tuple.
        """
        if isinstance(spec, TransformComponent):
            return spec
        if isinstance(spec, type) and issubclass(spec, TransformComponent):
            return spec()
        if (
            isinstance(spec, tuple)
            and len(spec) == 2
            and isinstance(spec[0], type)
            and issubclass(spec[0], TransformComponent)
            and isinstance(spec[1], dict)
        ):
            cls, params = spec
            return cls(**params)
        raise ValueError(f'Invalid transformer specification: {spec}')

    @staticmethod
    def _get_file_extension(node: Node) -> str:
        """Extracts the file extension from a node's metadata."""
        file_path = node.metadata.get('file_path', '')
        if not file_path:
            return ''
        return os.path.splitext(file_path)[1].lower()

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        """
        Asynchronously processes nodes by dispatching them to the correct transformer
        based on their file extension.
        """
        # Group nodes by file extension to process them in efficient batches.
        nodes_by_ext: Dict[str, List[Node]] = {}
        unhandled_nodes: List[Node] = []

        for node in nodes:
            ext = self._get_file_extension(node)
            if ext in self._transformer_configs:
                if ext not in nodes_by_ext:
                    nodes_by_ext[ext] = []
                nodes_by_ext[ext].append(node)
            else:
                # Collect nodes that don't have a specific transformer
                unhandled_nodes.append(node)

        # Asynchronously process each batch with its corresponding transformer.
        tasks = []
        for ext, batch in nodes_by_ext.items():
            if ext not in self._instance_cache:
                # Lazy-load: create and cache the transformer instance on first use.
                try:
                    spec = self._transformer_configs[ext]
                    self._instance_cache[ext] = self._create_transformer_instance(spec)
                    logger.info(
                        f"Initialized transformer for '{ext}': {self._instance_cache[ext].class_name()}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to create transformer for '{ext}': {e}. Skipping transformation for this type."
                    )
                    unhandled_nodes.extend(
                        batch
                    )  # Add batch to unhandled if creation fails
                    continue

            transformer = self._instance_cache[ext]
            tasks.append(transformer.acall(batch, **kwargs))

        # Gather results from all transformation tasks.
        processed_batches = await asyncio.gather(*tasks)

        # Flatten the list of processed batches and add back the unhandled nodes.
        final_nodes = [node for batch in processed_batches for node in batch]
        final_nodes.extend(unhandled_nodes)

        return final_nodes


class FileTypeDispatchingNodeParser(BaseTransformComponent):
    """
    Dispatches node parsing based on file type from metadata with user-customizable parsers.

    Examples:
        parser = FileTypeDispatchingNodeParser(
            custom_parsers={
                ".pdf": DoclingNodeParser(),
                ".docx": GoogleDocParser(remove_urls=True),
                ".html": HTMLNodeParser(strip_tags=False),
                ".jpg": ImageCaptionParser(model="gpt-4-vision"),
            }
        )
    """

    chunk_size: int = Field(
        default=1024, description='Default chunk size for non-specialized parsers.'
    )
    chunk_overlap: int = Field(
        default=20, description='Default chunk overlap for non-specialized parsers.'
    )
    custom_parsers: Dict[str, Any] = Field(
        default_factory=dict,
        description='Custom parsers per file extension. Can be instances, classes, or (class, params) tuples.',
    )
    min_node_length: int = Field(
        default=25, description="Minimum content length to keep a node's text."
    )
    prepend_title: bool = Field(
        default=True, description='Prepend document title to each node for RAG context.'
    )
    remove_urls: bool = Field(
        default=False, description='Remove URLs from content using heuristic detection.'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parser_cache: Dict[str, NodeParser] = {}
        self._initialize_parsers()

    @classmethod
    def class_name(cls) -> str:
        return 'FileTypeDispatchingNodeParser'

    def _get_default_parsers(self) -> Dict[str, Any]:
        """Get default parser configuration for common file types."""
        return {
            '.md': MarkdownNodeParser,
            '.markdown': MarkdownNodeParser,
            '.txt': (
                SentenceSplitter,
                {'chunk_size': self.chunk_size, 'chunk_overlap': self.chunk_overlap},
            ),
            '.rtf': (
                SentenceSplitter,
                {'chunk_size': self.chunk_size, 'chunk_overlap': self.chunk_overlap},
            ),
            '.docx': (
                GoogleDocParser,
                {
                    'min_node_length': self.min_node_length,
                    'prepend_title': self.prepend_title,
                    'remove_urls': self.remove_urls,
                },
            ),
        }

    def _initialize_parsers(self):
        """Initialize parser instances from configuration."""
        parser_config = self._get_default_parsers()
        # Override with user-provided parsers
        parser_config.update(self.custom_parsers)

        # Convert configuration to actual parser instances
        for ext, parser_spec in parser_config.items():
            try:
                self._parser_cache[ext] = self._create_parser_instance(parser_spec)
                logger.debug(
                    f'Initialized parser for {ext}: {type(self._parser_cache[ext]).__name__}'
                )
            except Exception as e:
                logger.warning(f'Failed to initialize parser for {ext}: {e}')

    def _create_parser_instance(self, parser_spec: Any) -> NodeParser:
        """Create a parser instance from an instance, class, or (class, params) tuple."""

        if isinstance(parser_spec, NodeParser):
            return parser_spec

        if isinstance(parser_spec, type) and issubclass(parser_spec, NodeParser):
            return parser_spec()

        if (
            isinstance(parser_spec, tuple)
            and len(parser_spec) == 2
            and isinstance(parser_spec[0], type)
            and issubclass(parser_spec[0], NodeParser)
        ):
            parser_class, params = parser_spec
            return parser_class(**params)

        raise ValueError(f'Invalid parser specification: {parser_spec}')

    @staticmethod
    def _get_file_extension(file_path: str) -> str:
        """Extract file extension from file path."""
        _, ext = os.path.splitext(file_path)
        return ext.lower()

    def _get_parser_for_extension(self, extension: str) -> NodeParser:
        """Get the appropriate parser for a file extension."""

        # Check if we have a specific parser for this extension
        if extension in self._parser_cache:
            print(extension, self._parser_cache)
            return self._parser_cache[extension]

        # Fallback to default sentence splitter
        logger.info(
            f'No specific parser for {extension}, using default SentenceSplitter'
        )
        return SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        """Asynchronously transform documents by dispatching to appropriate parsers."""
        all_result_nodes = []

        for doc in nodes:
            file_path = doc.metadata.get('file_path', '')
            logger.debug(f"Processing document: '{file_path}'")

            extension = self._get_file_extension(file_path)

            parser = self._get_parser_for_extension(extension)
            print(file_path, extension)

            try:
                parsed_nodes = await parser.aget_nodes_from_documents([doc])
                all_result_nodes.extend(parsed_nodes)
                logger.debug(f'Parsed {len(parsed_nodes)} nodes from {file_path}')
            except Exception:
                # Fallback to default splitter on error
                logger.debug(f'Falling back to SentenceSplitter for {file_path}')
                fallback_parser = SentenceSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                parsed_nodes = await fallback_parser.aget_nodes_from_documents([doc])
                all_result_nodes.extend(parsed_nodes)

        return all_result_nodes


class ResponseScrubber(BaseTransformComponent):
    """
    A component that scrubs noisy or undesired text patterns from LLM responses.
    Works with both LlamaIndex Nodes and raw strings.
    """

    remove_intro_phrases: bool = Field(
        default=True, description='Remove canned intro phrases.'
    )
    remove_ascii_art: bool = Field(
        default=True, description='Remove non-ASCII characters.'
    )
    remove_urls: bool = Field(default=True, description='Remove URLs from the text.')
    remove_markdown_links: bool = Field(
        default=True, description='Strip markdown-style links.'
    )
    normalize_whitespace: bool = Field(
        default=True, description='Normalize whitespace.'
    )
    remove_phrases: List[str] = Field(
        default_factory=lambda: [],
        description='List of intro phrases to strip if enabled.',
    )

    @classmethod
    def class_name(cls) -> str:
        return 'ResponseScrubber'

    async def execute(self, nodes: List[Node], **kwargs: Any) -> List[Node]:
        for node in nodes:
            original_text = node.get_content()
            cleaned_text = self._clean_text(original_text)
            node.set_content(cleaned_text)
        return nodes

    def clean_string(self, text: str) -> str:
        """Apply all enabled transformations to a plain string."""
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """Main cleaning pipeline for either Node content or raw strings."""
        if self.remove_intro_phrases:
            text = self._remove_intro_phrases(text)
        if self.remove_markdown_links:
            text = self._remove_markdown_links(text)
        if self.remove_urls:
            text = self._remove_urls(text)
        if self.remove_ascii_art:
            text = self._remove_ascii_art(text)
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        return text

    def _remove_intro_phrases(self, text: str) -> str:
        lowered = text.strip().lower()
        for phrase in self.remove_phrases:
            phrase_lower = phrase.strip().lower()
            if lowered.startswith(phrase_lower):
                return text[len(phrase) :].strip()
        return text

    @staticmethod
    def _remove_ascii_art(text: str) -> str:
        return re.sub(r'[^\x00-\x7F]+', '', text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        return re.sub(r'http\S+|www\.\S+', '', text)

    @staticmethod
    def _remove_markdown_links(text: str) -> str:
        return re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
