from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.core.extractors import (
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.node_parser import (
    CodeSplitter,
    HierarchicalNodeParser,
    MarkdownNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import MetadataMode, TransformComponent

from axion._core.logging import get_logger
from axion._handlers.knowledge.processing.transformations import (
    CodeBlockExtractor,
    ContentFilter,
    DocumentTextPreprocessor,
    DuplicateRemover,
    HTMLCleaner,
    MetadataEnricher,
    WhitespaceNormalizer,
)
from axion._handlers.validation import Validation

logger = get_logger(__name__)


class PipelineStrategy(ABC):
    """
    Strategy interface for pipeline creation.

    This module contains all pipeline strategy implementations using the Strategy pattern.
    Each strategy defines how to build a specific type of document processing pipeline.
    """

    def __init__(self, embed_model, llm=None):
        self.embed_model = embed_model
        Validation.validate_embed_model(self.embed_model)
        self.llm = llm
        if self.llm:
            Validation.validate_llm_model(self.llm)

    @abstractmethod
    def create_pipeline(self, **kwargs) -> List[TransformComponent]:
        """Create and return the list of transformations."""
        pass

    def get_strategy_name(self) -> str:
        """Get human-readable strategy name."""
        return self.__class__.__name__.replace('Strategy', '').replace('Pipeline', '')


class SimplePipelineStrategy(PipelineStrategy):
    """Strategy for default document processing."""

    def create_pipeline(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        enable_embeddings: bool = True,
        custom_transformations: Optional[List[TransformComponent]] = None,
        **kwargs,
    ) -> List[TransformComponent]:
        """Create default processing pipeline."""
        transformations = [
            # 1. Document preprocessing
            DocumentTextPreprocessor(clean_text=True, normalize_whitespace=True)
        ]

        # 2. Custom transformations (before parsing)
        if custom_transformations:
            transformations.extend(custom_transformations)

        # 3. Node parsing
        transformations.append(
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )

        # 4. Embeddings
        if enable_embeddings:
            transformations.append(self.embed_model)

        logger.info(
            f'Built default pipeline with {len(transformations)} transformations'
        )
        return transformations


class EnrichedPipelineStrategy(PipelineStrategy):
    """Strategy for default document processing."""

    def create_pipeline(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        enable_extractors: bool = True,
        enable_embeddings: bool = True,
        custom_transformations: Optional[List[TransformComponent]] = None,
        **kwargs,
    ) -> List[TransformComponent]:
        """Create default processing pipeline."""
        transformations = [
            # 1. Document preprocessing
            DocumentTextPreprocessor(clean_text=True, normalize_whitespace=True)
        ]

        # 2. Custom transformations (before parsing)
        if custom_transformations:
            transformations.extend(custom_transformations)

        # 3. Node parsing
        transformations.append(
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )

        # 4. Content extraction
        if enable_extractors and self.llm:
            transformations.extend(
                [
                    TitleExtractor(llm=self.llm, metadata_mode=MetadataMode.EMBED),
                    SummaryExtractor(llm=self.llm, metadata_mode=MetadataMode.EMBED),
                    KeywordExtractor(llm=self.llm, keywords=10),
                ]
            )

        # 5. Embeddings
        if enable_embeddings:
            transformations.append(self.embed_model)

        logger.info(
            f'Built default pipeline with {len(transformations)} transformations'
        )
        return transformations


class EnrichedHierarchicalPipelineStrategy(PipelineStrategy):
    """Strategy for default document processing."""

    def create_pipeline(
        self,
        chunk_sizes: List = None,
        enable_extractors: bool = True,
        enable_embeddings: bool = True,
        custom_transformations: Optional[List[TransformComponent]] = None,
        **kwargs,
    ) -> List[TransformComponent]:
        """Create default processing pipeline."""

        if chunk_sizes is None:
            chunk_sizes = [2048, 512, 128]  # Parent, child, grandchild chunks

        transformations = [
            # 1. Document preprocessing
            DocumentTextPreprocessor(clean_text=True, normalize_whitespace=True)
        ]

        # 2. Custom transformations (before parsing)
        if custom_transformations:
            transformations.extend(custom_transformations)

        # 3. Node parsing
        transformations.append(
            HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes),
        )

        # 4. Content extraction
        if enable_extractors and self.llm:
            transformations.extend(
                [
                    TitleExtractor(llm=self.llm, metadata_mode=MetadataMode.EMBED),
                    SummaryExtractor(llm=self.llm, metadata_mode=MetadataMode.EMBED),
                    KeywordExtractor(llm=self.llm, keywords=10),
                ]
            )

        # 5. Embeddings
        if enable_embeddings:
            transformations.append(self.embed_model)

        logger.info(
            f'Built default pipeline with {len(transformations)} transformations'
        )
        return transformations


class MarkdownPipelineStrategy(PipelineStrategy):
    """Strategy for Markdown document processing."""

    def create_pipeline(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        enable_extractors: bool = True,
        enable_embeddings: bool = True,
        preserve_headers: bool = True,
        **kwargs,
    ) -> List[TransformComponent]:
        """Create Markdown-optimized processing pipeline."""
        transformations = [
            # 1. Light preprocessing (preserve Markdown structure)
            WhitespaceNormalizer(),
            # 2. Markdown-aware parsing
            MarkdownNodeParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            # 3. Markdown-specific metadata
            MetadataEnricher(
                metadata_to_add={
                    'content_type': 'markdown',
                    'preserve_structure': preserve_headers,
                }
            ),
        ]

        # 4. Content extraction (optimized for structured content)
        if enable_extractors and self.llm:
            transformations.extend(
                [TitleExtractor(llm=self.llm), SummaryExtractor(llm=self.llm)]
            )

        # 5. Embeddings
        if enable_embeddings:
            transformations.append(self.embed_model)

        logger.info(
            f'Built Markdown pipeline with {len(transformations)} transformations'
        )
        return transformations


class CodePipelineStrategy(PipelineStrategy):
    """Strategy for code document processing."""

    def create_pipeline(
        self,
        language: str = 'python',
        chunk_lines: int = 40,
        chunk_lines_overlap: int = 5,
        enable_embeddings: bool = True,
        preserve_syntax: bool = True,
        **kwargs,
    ) -> List[TransformComponent]:
        """Create code-optimized processing pipeline."""

        transformations = [
            # 1. Code-specific preprocessing
            CodeBlockExtractor(mark_as_code=True),
            WhitespaceNormalizer(),
            # 2. Code-aware splitting
            CodeSplitter(
                language=language,
                chunk_lines=chunk_lines,
                chunk_lines_overlap=chunk_lines_overlap,
            ),
            # 3. Code-specific metadata
            MetadataEnricher(metadata_to_add={'content_type': 'code'}),
        ]

        # 4. Embeddings (code-optimized)
        if enable_embeddings:
            transformations.append(self.embed_model)

        logger.info(
            f'Built code pipeline for {language} with {len(transformations)} transformations'
        )
        return transformations


class ResearchPipelineStrategy(PipelineStrategy):
    """Strategy for research document processing."""

    def create_pipeline(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 50,
        enable_extractors: bool = True,
        enable_embeddings: bool = True,
        min_content_length: int = 50,
        required_keywords: Optional[List[str]] = None,
        extract_questions: int = 5,
        extract_keywords: int = 15,
        **kwargs,
    ) -> List[TransformComponent]:
        """Create research-optimized processing pipeline."""

        transformations = [
            # Base research transformations
            DocumentTextPreprocessor(clean_text=True, normalize_whitespace=True),
            MetadataEnricher(metadata_to_add={'content_domain': 'research'}),
            # Additional filtering
            ContentFilter(
                min_length=min_content_length,
                required_keywords=required_keywords
                or ['research', 'study', 'analysis'],
            ),
            # Chunking
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        ]

        # 4. Enhanced extraction for research content
        if enable_extractors and self.llm:
            transformations.extend(
                [
                    TitleExtractor(llm=self.llm, metadata_mode=MetadataMode.EMBED),
                    SummaryExtractor(llm=self.llm, metadata_mode=MetadataMode.EMBED),
                    QuestionsAnsweredExtractor(
                        llm=self.llm, questions=extract_questions
                    ),
                    KeywordExtractor(llm=self.llm, keywords=extract_keywords),
                ]
            )

        # 5. Research-specific metadata
        transformations.append(
            MetadataEnricher(
                metadata_to_add={
                    'content_domain': 'research',
                    'extraction_level': 'enhanced',
                    'chunk_overlap_ratio': chunk_overlap / chunk_size,
                }
            )
        )

        # 6. Embeddings
        if enable_embeddings:
            transformations.append(self.embed_model)

        logger.info(
            f'Built research pipeline with {len(transformations)} transformations'
        )
        return transformations


class WebContentPipelineStrategy(PipelineStrategy):
    """Strategy for web content processing."""

    def create_pipeline(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 20,
        enable_extractors: bool = False,
        enable_embeddings: bool = True,
        aggressive_cleaning: bool = True,
        remove_duplicates: bool = True,
        **kwargs,
    ) -> List[TransformComponent]:
        """Create web-content-optimized processing pipeline."""
        transformations = []

        # 1. Aggressive web content cleaning
        if aggressive_cleaning:
            transformations.extend(
                [
                    HTMLCleaner(remove_scripts=True, remove_styles=True),
                    WhitespaceNormalizer(),
                    DuplicateRemover(),
                ]
            )
        else:
            transformations.extend(
                [
                    HTMLCleaner(remove_scripts=True, remove_styles=True),
                    WhitespaceNormalizer(),
                ]
            )

        # 2. Additional deduplication for web content
        if remove_duplicates:
            transformations.append(DuplicateRemover())

        # 3. Web-optimized chunking (smaller chunks for noisy content)
        transformations.append(
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )

        # 4. Minimal extraction (web content is often noisy)
        if enable_extractors and self.llm:
            transformations.extend(
                [
                    TitleExtractor(llm=self.llm),
                    KeywordExtractor(llm=self.llm, keywords=5),
                ]
            )

        # 5. Web-specific metadata
        transformations.append(
            MetadataEnricher(
                metadata_to_add={
                    'content_type': 'web',
                    'cleaned_aggressively': aggressive_cleaning,
                    'deduplication_applied': remove_duplicates,
                }
            )
        )

        # 6. Embeddings
        if enable_embeddings:
            transformations.append(self.embed_model)

        logger.info(
            f'Built web content pipeline with {len(transformations)} transformations'
        )
        return transformations


class ConversationalStrategy(PipelineStrategy):
    """Strategy for chat/conversation processing."""

    def create_pipeline(
        self,
        chunk_size: int = 256,  # Smaller chunks for conversations
        chunk_overlap: int = 50,
        enable_extractors: bool = False,
        enable_embeddings: bool = True,
        preserve_speaker_info: bool = True,
        **kwargs,
    ) -> List[TransformComponent]:
        """Create conversation-optimized processing pipeline."""
        transformations = [
            # 1. Light preprocessing (preserve conversational tone)
            WhitespaceNormalizer(),
            # 2. Small chunking for conversational context
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            # 3. Conversation-specific metadata
            MetadataEnricher(
                metadata_to_add={
                    'content_type': 'conversation',
                    'preserve_speakers': preserve_speaker_info,
                    'chunk_size_small': True,
                }
            ),
        ]

        # 4. Minimal extraction (preserve natural language)
        if enable_extractors and self.llm:
            transformations.append(KeywordExtractor(llm=self.llm, keywords=5))

        # 5. Embeddings
        if enable_embeddings:
            transformations.append(self.embed_model)

        logger.info(
            f'Built conversational pipeline with {len(transformations)} transformations'
        )
        return transformations
