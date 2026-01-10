import asyncio
from typing import Dict, List, Optional

from axion._core.logging import get_logger
from axion._core.schema import LLMRunnable, RichBaseModel
from axion._core.tracing.handlers import BaseTraceHandler
from axion._handlers.llm.handler import LLMHandler
from pydantic import Field

logger = get_logger(__name__)


ENHANCEMENT_FEEDBACK_TEMPLATE = """The previous generation attempt for QA pairs achieved an average quality score of {average_quality:.2f}.
This was below the required threshold of {validation_threshold:.2f}.

{num_low_quality} pairs were identified as low-quality. Please improve them.
Common issues included:
{feedback_examples}

In the next iteration, please focus on:
1.  Ensuring questions are directly and unambiguously answerable from the provided statements.
2.  Refining the clarity and conciseness of both questions and answers.
3.  Adhering to the specified difficulty and answer length.
"""


class ValidationInput(RichBaseModel):
    question: str = Field(description='The question text')
    answer: str = Field(description='The answer text')
    statements: List[str] = Field(description='The source statements')
    supporting_statement_indices: List[int] = Field(
        description='Indices of statements supporting the answer'
    )


class ValidationOutput(RichBaseModel):
    is_valid: bool = Field(description='Whether the QA pair is valid')
    feedback: str = Field(description='Feedback about the QA pair quality')
    improved_answer: Optional[str] = Field(
        description='Improved answer if needed', default=None
    )
    quality_score: float = Field(
        description='Quality score (0-1) based on comprehensive evaluation', default=0.0
    )


class Validation(LLMHandler[ValidationInput, ValidationOutput]):
    instruction = (
        'Evaluate the quality and factual accuracy of the given question-answer pair based on the source statements. '
        'Analyze the following dimensions:\n'
        '1. Accuracy: Is the answer fully supported by the source statements?\n'
        '2. Completeness: Does the answer address all aspects of the question?\n'
        '3. Relevance: Does the answer directly address the question?\n'
        '4. Clarity: Is the answer clearly articulated?\n'
        '5. Factual Integrity: Does the answer avoid introducing information not in the statements?\n\n'
        'First provide specific feedback on each dimension. Then assign a quality score from 0.0 to 1.0, '
        'where 0.0 is completely unacceptable and 1.0 is perfect. '
        'If there are issues, provide an improved answer.'
    )
    input_model = ValidationInput
    output_model = ValidationOutput
    description = 'QA Validator'
    owner = 'QASystem'
    examples = [
        (
            ValidationInput(
                question='How does Python support multiple programming paradigms?',
                answer='Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.',
                statements=[
                    'Python is a high-level, interpreted programming language.',
                    'Python was created by Guido van Rossum in 1991.',
                    'Python supports multiple programming paradigms including procedural, object-oriented, and functional.',
                    "Python's design philosophy emphasizes code readability.",
                    'Python has a comprehensive standard library.',
                ],
                supporting_statement_indices=[2],
            ),
            ValidationOutput(
                is_valid=True,
                feedback="Evaluation:\n1. Accuracy: The answer is fully supported by statement #2.\n2. Completeness: The answer completely addresses the question by listing all the programming paradigms mentioned in the statements.\n3. Relevance: The answer directly addresses the question about Python's support for multiple paradigms.\n4. Clarity: The answer is clearly articulated.\n5. Factual Integrity: The answer does not introduce any information not present in the facts.",
                quality_score=0.95,
            ),
        ),
        (
            ValidationInput(
                question='What are the key features of Python?',
                answer='Python is known for its simplicity, readability, and versatility in web development and data science.',
                statements=[
                    'Python is a high-level, interpreted programming language.',
                    'Python was created by Guido van Rossum in 1991.',
                    'Python supports multiple programming paradigms including procedural, object-oriented, and functional.',
                    "Python's design philosophy emphasizes code readability.",
                    'Python has a comprehensive standard library.',
                ],
                supporting_statement_indices=[0, 3, 4],
            ),
            ValidationOutput(
                is_valid=False,
                feedback="Evaluation:\n1. Accuracy: The answer mentions 'readability' which is supported by fact #3, but 'simplicity' is not explicitly mentioned in the statements.\n2. Completeness: The answer misses key features mentioned in the facts such as being high-level, interpreted, and having a comprehensive standard library.\n3. Relevance: The answer addresses the question but includes unsupported information.\n4. Clarity: The answer is clear but incomplete.\n5. Factual Integrity: The answer introduces information not present in the facts - specifically 'simplicity' and applications in 'web development and data science'.",
                improved_answer='Python is a high-level, interpreted programming language that emphasizes code readability. It supports multiple programming paradigms and has a comprehensive standard library.',
                quality_score=0.4,
            ),
        ),
    ]


class QAValidator:
    """Validates generated QA pairs for quality and factual accuracy"""

    BATCH_SIZE = 5

    def __init__(self, llm: LLMRunnable, tracer: Optional[BaseTraceHandler] = None):
        self.llm = llm
        self.tracer = tracer

    async def validate_qa_pair(self, qa_pair: Dict, statements: List[str]) -> Dict:
        """Validate a single QA pair against source statements"""

        model = Validation(llm=self.llm, tracer=self.tracer)
        input_data = ValidationInput(
            question=qa_pair['question'],
            answer=qa_pair['answer'],
            statements=statements,
            supporting_statement_indices=qa_pair['statement_indices'],
        )
        result = await model.execute(input_data)

        validated_qa_pair = qa_pair.copy()
        validated_qa_pair['is_valid'] = result.is_valid
        validated_qa_pair['validation_feedback'] = result.feedback
        validated_qa_pair['quality_score'] = result.quality_score

        if not result.is_valid and result.improved_answer:
            validated_qa_pair['original_answer'] = qa_pair['answer']
            validated_qa_pair['answer'] = result.improved_answer

        return validated_qa_pair

    async def validate_qa_pairs_batch(
        self, qa_pairs: List[Dict], statements: List[str]
    ) -> List[Dict]:
        """
        Validate multiple QA pairs in parallel.

        Args:
            qa_pairs: List of QA pairs to validate
            statements: List of statements to validate against

        Returns:
            List of validated QA pairs
        """
        # Process QA pairs in batches to avoid overwhelming the API
        all_validated_pairs = []

        for i in range(0, len(qa_pairs), self.BATCH_SIZE):
            batch_pairs = qa_pairs[i : i + self.BATCH_SIZE]

            # Process batch in parallel
            tasks = []
            for pair in batch_pairs:
                tasks.append(self.validate_qa_pair(pair, statements))

            # Wait for all pairs in batch to be processed
            batch_validated_pairs = await asyncio.gather(*tasks)
            all_validated_pairs.extend(batch_validated_pairs)

            logger.info(
                f'Validated {len(all_validated_pairs)}/{len(qa_pairs)} QA pairs'
            )

        return all_validated_pairs
