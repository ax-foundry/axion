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


class ThrottledSalesforceProductExtractor(ThrottledBaseExtractor):
    """Extract Salesforce products and services mentioned in text."""

    # Comprehensive list of Salesforce products and services
    salesforce_products: List[str] = Field(
        default=[],
        description='List of Salesforce products to detect',
    )

    include_versions: bool = Field(
        default=True,
        description='Whether to extract version information (e.g., API versions, release versions)',
    )

    include_features: bool = Field(
        default=True, description='Whether to extract specific features within products'
    )

    @classmethod
    def class_name(cls) -> str:
        return 'ThrottledSalesforceProductExtractor'

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extract Salesforce products with throttling."""
        logger.info(f'Extracting Salesforce products for {len(nodes)} nodes')

        async def extract_salesforce_products(node: BaseNode) -> Dict:
            text = node.get_content(metadata_mode=self.metadata_mode)

            # First do simple keyword matching for known products
            found_products = []
            text_lower = text.lower()

            for product in self.salesforce_products:
                if product.lower() in text_lower:
                    found_products.append(product)

            # Use LLM for more sophisticated extraction
            if self._semaphore_executor and hasattr(self, 'llm'):
                prompt = f"""Analyze the following text and identify all Salesforce products, services, APIs, and features mentioned.

Text: {text}

Look for:
1. Salesforce products (Sales Cloud, Service Cloud, Einstein, etc.)
2. Specific features within products
3. API versions or endpoints
4. Development tools and platforms
5. Industry-specific solutions
6. Any product abbreviations or nicknames

Known products include: {', '.join(self.salesforce_products[:20])}... and many others.

Return in JSON format:
{{
    "primary_products": ["Sales Cloud", "Einstein"],
    "features": ["Lead Scoring", "Opportunity Management"],
    "apis": ["REST API v52.0", "Metadata API"],
    "tools": ["Salesforce CLI", "VS Code"],
    "confidence": 0.9,
    "context": "Brief explanation of what the text discusses about these products"
}}"""

                try:
                    response = await self._semaphore_executor.run(
                        self.llm.acomplete, prompt
                    )

                    import json

                    llm_results = json.loads(response.text)

                    # Combine keyword matching with LLM results
                    all_products = list(
                        set(found_products + llm_results.get('primary_products', []))
                    )

                    return {
                        'salesforce_products': all_products,
                        'salesforce_features': llm_results.get('features', []),
                        'salesforce_apis': (
                            llm_results.get('apis', []) if self.include_versions else []
                        ),
                        'salesforce_tools': llm_results.get('tools', []),
                        'salesforce_confidence': llm_results.get('confidence', 0.5),
                        'salesforce_context': llm_results.get('context', ''),
                        'salesforce_extraction_method': 'hybrid',  # keyword + LLM
                    }
                except Exception as e:
                    logger.warning(
                        f'LLM extraction failed, using keyword matching: {e}'
                    )

            # Fallback to keyword matching only
            return {
                'salesforce_products': found_products,
                'salesforce_features': [],
                'salesforce_apis': [],
                'salesforce_tools': [],
                'salesforce_confidence': 0.8 if found_products else 0.1,
                'salesforce_context': f'Found {len(found_products)} products via keyword matching',
                'salesforce_extraction_method': 'keyword_only',
            }

        tasks = [extract_salesforce_products(node) for node in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            result if not isinstance(result, Exception) else {} for result in results
        ]
