import asyncio
import re
from typing import List, Optional

from axion._core.logging import get_logger
from axion._core.schema import LLMRunnable, RichBaseModel
from axion._core.tracing.handlers import BaseTraceHandler
from axion._handlers.llm.handler import LLMHandler
from pydantic import Field

logger = get_logger(__name__)


class StatementGeneratorInput(RichBaseModel):
    content: str = Field(description='The document content')
    max_statements: Optional[int] = Field(
        default=None, description='Maximum number of statements to extract'
    )


class StatementGeneratorOutput(RichBaseModel):
    statements: List[str] = Field(description='The generated factual statements')


class StatementGenerator(LLMHandler[StatementGeneratorInput, StatementGeneratorOutput]):
    instruction = (
        'Extract all distinct, standalone factual statements from the provided content. '
        'Ensure each statement is clear, self-contained, and does not use pronouns. '
        'Each statement should be a single piece of information that can stand on its own. '
        'If a maximum number of statements is specified, rank all statements by importance and return only the most important ones up to that limit. '
        'Prioritize statements that contain key facts, main concepts, or essential information.'
    )
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    description = 'Statement Extractor'
    owner = 'QASystem'
    examples = [
        (
            StatementGeneratorInput(
                content=(
                    'Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest '
                    'and most influential physicists of all time. He was best known for developing the theory of relativity, '
                    'he also made important contributions to the development of the theory of quantum mechanics.'
                ),
            ),
            StatementGeneratorOutput(
                statements=[
                    'Albert Einstein was a German-born theoretical physicist.',
                    'Albert Einstein is widely acknowledged as one of the greatest and most influential physicists of all time.',
                    'Albert Einstein was best known for developing the theory of relativity.',
                    'Albert Einstein made important contributions to the development of the theory of quantum mechanics.',
                ]
            ),
        ),
        (
            StatementGeneratorInput(
                content=(
                    'The Great Wall of China is a series of fortifications built across the historical northern borders of China. '
                    'It was constructed to protect Chinese states and empires against various nomadic groups from the Eurasian Steppe.'
                ),
            ),
            StatementGeneratorOutput(
                statements=[
                    'The Great Wall of China is a series of fortifications.',
                    'The Great Wall of China was built across the historical northern borders of China.',
                    'The Great Wall of China was constructed to protect Chinese states and empires.',
                    'The Great Wall of China was built to defend against various nomadic groups from the Eurasian Steppe.',
                ]
            ),
        ),
        (
            StatementGeneratorInput(
                content=(
                    'Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. '
                    'Photosynthesis generally involves the green pigment chlorophyll and generates oxygen as a by-product.'
                ),
                max_statements=2,
            ),
            StatementGeneratorOutput(
                statements=[
                    'Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water.',
                    'Photosynthesis generally involves the green pigment chlorophyll and generates oxygen as a by-product.',
                ]
            ),
        ),
    ]


class StatementExtractor:
    """Extracts factual statements from documents using LLMs."""

    def __init__(self, llm: LLMRunnable, tracer: Optional[BaseTraceHandler] = None):
        self.llm = llm
        self.tracer = tracer

    async def extract_statements(
        self, content: str, max_statements: Optional[int] = None
    ) -> List[str]:
        """
        Extract factual statements from a single content block.
        """
        generator = StatementGenerator(llm=self.llm, tracer=self.tracer)
        input_data = StatementGeneratorInput(
            content=content, max_statements=max_statements
        )
        result = await generator.execute(input_data)

        statements = result.statements
        if max_statements and len(statements) > max_statements:
            logger.debug(
                f'Trimming {len(statements)} statements to {max_statements} limit.'
            )
            return statements[:max_statements]

        return statements

    async def extract_statements_from_chunks(
        self, chunks: List[str], statements_per_chunk: int = 5
    ) -> List[str]:
        """
        Extract and deduplicate important factual statements from multiple content chunks.
        """
        logger.info(
            f'Processing {len(chunks)} chunks (top {statements_per_chunk} per chunk)'
        )

        tasks = [
            self._extract(chunk, i + 1, len(chunks), statements_per_chunk)
            for i, chunk in enumerate(chunks)
        ]
        chunk_results = await asyncio.gather(*tasks)

        all_statements = [stmt for chunk in chunk_results for stmt in chunk]
        total_before_dedup = len(all_statements)
        unique_statements = self._deduplicate_statements(all_statements)

        logger.info(
            f'Total extracted: {total_before_dedup}, Unique: {len(unique_statements)}, '
            f'Duplicates removed: {total_before_dedup - len(unique_statements)} '
            f'({(1 - len(unique_statements)/total_before_dedup) * 100:.1f}% dedup rate)'
        )

        return unique_statements

    async def _extract(
        self, chunk: str, index: int, total: int, max_statements: int
    ) -> List[str]:
        logger.info(
            f'Chunk {index}/{total} → extracting top {max_statements} statements'
        )
        statements = await self.extract_statements(chunk, max_statements=max_statements)
        logger.debug(f'Chunk {index} returned {len(statements)} statements')
        return statements

    @staticmethod
    def _deduplicate_statements(statements: List[str]) -> List[str]:
        seen, unique = set(), []
        for stmt in statements:
            normalized = re.sub(r'[^\w\s]', '', stmt.lower())
            if normalized not in seen:
                seen.add(normalized)
                unique.append(stmt)
        return unique
