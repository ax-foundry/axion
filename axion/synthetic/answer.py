import asyncio
from typing import Dict, List, Optional

from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import LLMRunnable, RichBaseModel
from axion._core.tracing.handlers import BaseTraceHandler
from axion._handlers.llm.handler import LLMHandler

logger = get_logger(__name__)


class AnswerInput(RichBaseModel):
    question_text: str = Field(description='The question to answer')
    statements: List[str] = Field(
        description='Statements to use in generating the answer'
    )
    statement_indices: List[int] = Field(
        description='Indices of statements most relevant to the question'
    )
    answer_length: str = Field(
        description='Desired length of the answer (short, medium, long)'
    )
    custom_guidelines: Optional[str] = Field(
        default=None, description='Custom guidelines for answer generation'
    )


class AnswerOutput(RichBaseModel):
    answer_text: str = Field(description='The generated answer')
    supporting_statement_indices: List[int] = Field(
        description='Indices of statements that support this answer'
    )


class Answer(LLMHandler[AnswerInput, AnswerOutput]):
    instruction = (
        'Generate a comprehensive, accurate answer to the question using ONLY the provided statements. '
        'The answer should be clear, concise, and directly address the question. '
        'Include ONLY information from the statements in your answer. '
        'Do not introduce any information not contained in the statements. '
        'Specify which statements support your answer by including their indices.'
    )
    input_model = AnswerInput
    output_model = AnswerOutput
    description = 'Answer Generator'
    owner = 'QASystem'
    examples = [
        (
            AnswerInput(
                question_text='Under what game conditions does the infield fly rule apply?',
                statements=[
                    'The infield fly rule applies when there are fewer than two outs with runners on first and second or bases loaded.',
                    'The rule was introduced in 1895 to prevent infielders from deliberately dropping easy catches.',
                    'When the infield fly rule is invoked, the batter is automatically out.',
                    'Runners may advance at their own risk when the ball is caught or dropped.',
                    'The umpire must declare "infield fly" while the ball is in the air.',
                    'The infield fly rule only applies to fair fly balls that can be caught with ordinary effort.',
                    'The rule protects baserunners from being subjected to an unfair double play.',
                ],
                statement_indices=[0, 5],
                answer_length='medium',
            ),
            AnswerOutput(
                answer_text='The infield fly rule applies under specific game conditions: there must be fewer than two outs with runners on first and second base, or bases loaded. Additionally, the rule only applies to fair fly balls that can be caught with ordinary effort by an infielder. These conditions ensure the rule is invoked only in situations where the defense could exploit an easy catch for a double play.',
                supporting_statement_indices=[0, 5],
            ),
        ),
        (
            AnswerInput(
                question_text='How does the infield fly rule protect baserunners?',
                statements=[
                    'The infield fly rule applies when there are fewer than two outs with runners on first and second or bases loaded.',
                    'The rule was introduced in 1895 to prevent infielders from deliberately dropping easy catches.',
                    'When the infield fly rule is invoked, the batter is automatically out.',
                    'Runners may advance at their own risk when the ball is caught or dropped.',
                    'The umpire must declare "infield fly" while the ball is in the air.',
                    'The infield fly rule only applies to fair fly balls that can be caught with ordinary effort.',
                    'The rule protects baserunners from being subjected to an unfair double play.',
                ],
                statement_indices=[6, 3],
                answer_length='short',
            ),
            AnswerOutput(
                answer_text='The infield fly rule protects baserunners by preventing an unfair double play. When invoked, runners may advance at their own risk whether the ball is caught or dropped.',
                supporting_statement_indices=[6, 3],
            ),
        ),
    ]

    def get_instruction(self, input_data: AnswerInput) -> str:
        instruction = self.instruction

        # Add answer length guidance
        length_guidance = {
            'short': 'Keep your answer brief and to the point, ideally 1-2 sentences.',
            'medium': 'Provide a balanced answer with sufficient detail, typically 2-4 sentences.',
            'long': 'Create a comprehensive answer with thorough explanations, typically 4-8 sentences.',
        }

        instruction += f'\n\nAnswer Length: {length_guidance.get(input_data.answer_length, length_guidance["medium"])}'

        # Append custom guidelines if provided
        if input_data.custom_guidelines:
            instruction += f'\n\nAdditional Guidelines:\n{input_data.custom_guidelines}'

        return instruction


class AnswerGenerator:
    """Generates answers to questions using statements and LLMs"""

    BATCH_SIZE = 5

    def __init__(self, llm: LLMRunnable, tracer: Optional[BaseTraceHandler] = None):
        self.llm = llm
        self.tracer = tracer

    async def generate_answer(
        self,
        question: Dict,
        statements: List[str],
        answer_length: str = 'medium',
        custom_guidelines: Optional[str] = None,
    ) -> Dict:
        """Generate an answer for a question using the provided statements"""

        model = Answer(llm=self.llm, tracer=self.tracer)
        input_data = AnswerInput(
            question_text=question['text'],
            statements=statements,
            statement_indices=question['statement_indices'],
            answer_length=answer_length,
            custom_guidelines=custom_guidelines,
        )
        result = await model.execute(input_data)

        return {
            'question': question['text'],
            'answer': result.answer_text,
            'question_type': question['type'],
            'difficulty': question['difficulty'],
            'statement_indices': result.supporting_statement_indices,
        }

    async def generate_answers_batch(
        self,
        questions: List[Dict],
        statements: List[str],
        answer_length: str = 'medium',
        custom_guidelines: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate answers for multiple questions in parallel.

        Args:
            questions: List of questions to answer
            statements: List of statements to use in generating answers
            answer_length: Desired length of the answers (short, medium, long)
            custom_guidelines: Optional custom guidelines for answer generation

        Returns:
            List of question-answer pairs
        """
        all_qa_pairs = []

        for i in range(0, len(questions), self.BATCH_SIZE):
            batch_questions = questions[i : i + self.BATCH_SIZE]
            tasks = []
            for question in batch_questions:
                tasks.append(
                    self.generate_answer(
                        question,
                        statements,
                        answer_length=answer_length,
                        custom_guidelines=custom_guidelines,
                    )
                )
            batch_qa_pairs = await asyncio.gather(*tasks)
            all_qa_pairs.extend(batch_qa_pairs)
            logger.info(
                f'Generated answers for {len(all_qa_pairs)}/{len(questions)} questions'
            )
        return all_qa_pairs
