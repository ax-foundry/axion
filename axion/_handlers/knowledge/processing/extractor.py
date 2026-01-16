import asyncio
from abc import ABC
from typing import Any, Dict, List, Optional, Sequence, Union

from llama_index.core.extractors import (
    BaseExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.schema import BaseNode
from pydantic import Field, PrivateAttr

from axion._core.asyncio import SemaphoreExecutor
from axion._core.logging import get_logger

logger = get_logger(__name__)


class ThrottledBaseExtractor(BaseExtractor, ABC):
    """Base class for throttled extractors that limits LLM calls."""

    max_concurrent: int = Field(default=3, description='Maximum concurrent LLM calls')

    # Use PrivateAttr for non-serializable attributes
    _semaphore_executor: Optional[SemaphoreExecutor] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model creation."""
        super().model_post_init(__context)
        self._semaphore_executor = SemaphoreExecutor(max_concurrent=self.max_concurrent)

    async def aextract(
        self, nodes: Sequence[BaseNode]
    ) -> Union[List[Dict], BaseException]:
        """Extract metadata with throttling - async version."""
        extractor_name = self.class_name()
        logger.info(
            f'Running {extractor_name} for {len(nodes)} nodes with '
            f'max_concurrent={self.max_concurrent}'
        )

        async def extract_single_node(node: BaseNode) -> Dict:
            """Extract metadata for a single node."""
            if self._semaphore_executor:
                # Run the parent's aextract method with throttling
                result_metadata = await self._semaphore_executor.run(
                    super(ThrottledBaseExtractor, self).aextract, [node]
                )
                return result_metadata[0] if result_metadata else {}
            else:
                # Fallback if semaphore not initialized
                result_metadata = await super().aextract([node])
                return result_metadata[0] if result_metadata else {}

        # Process all nodes concurrently with throttling
        tasks = [extract_single_node(node) for node in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        metadata_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f'{extractor_name} failed for node {i}: {result}')
                metadata_list.append({})  # Return empty metadata on error
            else:
                metadata_list.append(result)
        return metadata_list


class ThrottledTitleExtractor(ThrottledBaseExtractor, TitleExtractor):
    """Throttled version of TitleExtractor that limits LLM calls."""

    @classmethod
    def class_name(cls) -> str:
        return 'ThrottledTitleExtractor'


class ThrottledSummaryExtractor(ThrottledBaseExtractor, SummaryExtractor):
    """Throttled version of SummaryExtractor that limits LLM calls."""

    @classmethod
    def class_name(cls) -> str:
        return 'ThrottledSummaryExtractor'


class ThrottledQuestionsAnsweredExtractor(
    ThrottledBaseExtractor, QuestionsAnsweredExtractor
):
    """Throttled version of QuestionsAnsweredExtractor that limits LLM calls."""

    @classmethod
    def class_name(cls) -> str:
        return 'ThrottledQuestionsAnsweredExtractor'


class ThrottledKeywordExtractor(ThrottledBaseExtractor, KeywordExtractor):
    """Throttled version of KeywordExtractor that limits LLM calls."""

    @classmethod
    def class_name(cls) -> str:
        return 'ThrottledKeywordExtractor'


class ThrottledTopicExtractor(ThrottledBaseExtractor):
    """Extract main topics and themes from text."""

    num_topics: int = Field(default=3, description='Number of topics to extract')

    @classmethod
    def class_name(cls) -> str:
        return 'ThrottledTopicExtractor'

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extract topics with throttling."""
        logger.info(f'Extracting topics for {len(nodes)} nodes')

        async def extract_topics(node: BaseNode) -> Dict:
            text = node.get_content(metadata_mode=self.metadata_mode)
            prompt = f"""Analyze the following text and identify the {self.num_topics} main topics or themes.

            Text: {text}

            Return only the topics as a comma-separated list."""

            if self._semaphore_executor and hasattr(self, 'llm'):
                response = await self._semaphore_executor.run(
                    self.llm.acomplete, prompt
                )
                topics = [topic.strip() for topic in response.text.split(',')]
                return {'topics': topics[: self.num_topics]}
            return {'topics': []}

        tasks = [extract_topics(node) for node in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            result if not isinstance(result, Exception) else {} for result in results
        ]
