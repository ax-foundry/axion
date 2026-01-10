from typing import Dict, List, Optional

from axion._core.logging import get_logger
from axion._core.schema import LLMRunnable, RichBaseModel
from axion._core.tracing.handlers import BaseTraceHandler
from axion._handlers.llm.handler import LLMHandler
from pydantic import Field

logger = get_logger(__name__)


class QuestionInput(RichBaseModel):
    statements: List[str] = Field(description='List of statements to base questions on')
    num_questions: int = Field(description='Number of questions to generate')
    question_types: List[str] = Field(description='Types of questions to generate')
    difficulty: str = Field(description='Difficulty level of questions')
    custom_guidelines: Optional[str] = Field(
        default=None, description='Custom guidelines for question generation'
    )


class QuestionOutput(RichBaseModel):
    questions: List[Dict] = Field(description='Generated questions with metadata')


class Question(LLMHandler[QuestionInput, QuestionOutput]):
    instruction = (
        'Generate diverse, engaging questions based on the provided statements. '
        'Each question should relate to one or more statements and be challenging but answerable. '
        'For each question, include the following fields in your response JSON:\n'
        "- 'text': The actual question text\n"
        "- 'type': The question type (factual, conceptual, application, analysis)\n"
        "- 'difficulty': The question difficulty level (easy, medium, hard)\n"
        "- 'statement_indices': A list of indices of statements that relate to this question"
    )
    input_model = QuestionInput
    output_model = QuestionOutput
    description = 'Question Generator'
    owner = 'QASystem'
    examples = [
        (
            QuestionInput(
                statements=[
                    'The Customer 360 Data Model standardizes data across Salesforce applications.',
                    'Data must be transformed and prepped before mapping it in the Data Cloud.',
                    'Once data is mapped to the Customer 360 Data Model, it can be manipulated with SQL.',
                    'The Customer 360 Data Model makes data more usable across the Salesforce ecosystem.',
                    'A data lake object can be mapped to a Customer 360 Data Model object.',
                    'The Unified Individual Data Model Object contains demographic data.',
                    'The Unified Individual Data Model Object enables personalized marketing campaigns.',
                ],
                num_questions=3,
                question_types=['factual', 'conceptual', 'application'],
                difficulty='medium',
            ),
            QuestionOutput(
                questions=[
                    {
                        'text': 'What is the purpose of transforming and prepping data before mapping it in the Data Cloud?',
                        'type': 'conceptual',
                        'difficulty': 'medium',
                        'statement_indices': [1],
                    },
                    {
                        'text': 'How does the Customer 360 Data Model enhance data usability across Salesforce?',
                        'type': 'application',
                        'difficulty': 'medium',
                        'statement_indices': [0, 3],
                    },
                    {
                        'text': 'What role does the Unified Individual Data Model Object play in marketing strategies?',
                        'type': 'factual',
                        'difficulty': 'medium',
                        'statement_indices': [5, 6],
                    },
                ]
            ),
        ),
        (
            QuestionInput(
                statements=[
                    'Python is a high-level, interpreted programming language.',
                    'Python was created by Guido van Rossum in 1991.',
                    'Python supports multiple programming paradigms including procedural, object-oriented, and functional.',
                    "Python's design philosophy emphasizes code readability.",
                    'Python has a comprehensive standard library.',
                ],
                num_questions=2,
                question_types=['factual', 'conceptual'],
                difficulty='easy',
            ),
            QuestionOutput(
                questions=[
                    {
                        'text': 'Who created Python and when was it first released?',
                        'type': 'factual',
                        'difficulty': 'easy',
                        'statement_indices': [1],
                    },
                    {
                        'text': "How does Python's design philosophy influence its usage among programmers?",
                        'type': 'conceptual',
                        'difficulty': 'easy',
                        'statement_indices': [3, 0],
                    },
                ]
            ),
        ),
    ]

    def get_instruction(self, input_data: QuestionInput) -> str:
        instruction = self.instruction
        if input_data.custom_guidelines:
            instruction += f'\n\nAdditional Guidelines:\n{input_data.custom_guidelines}'
        return instruction


class QuestionGenerator:
    """Generates questions from statements using LLMs"""

    def __init__(self, llm: LLMRunnable, tracer: Optional[BaseTraceHandler] = None):
        self.llm = llm
        self.tracer = tracer

    async def generate_questions(
        self,
        statements: List[str],
        num_questions: int,
        question_types: List[str],
        difficulty: str,
        custom_guidelines: Optional[str] = None,
    ) -> List[Dict]:
        """Generate questions from statements"""

        model = Question(llm=self.llm, tracer=self.tracer)
        input_data = QuestionInput(
            statements=statements,
            num_questions=num_questions,
            question_types=question_types,
            difficulty=difficulty,
            custom_guidelines=custom_guidelines,
        )
        result = await model.execute(input_data)

        # Standardize the output format
        standardized_questions = []
        for question in result.questions:
            standardized_question = {
                'text': question.get('text', question.get('question_text', '')),
                'type': question.get('type', question.get('question_type', '')),
                'difficulty': question.get(
                    'difficulty', question.get('difficulty_level', '')
                ),
                'statement_indices': question.get('statement_indices', []),
            }
            standardized_questions.append(standardized_question)

        return standardized_questions

    async def balance_question_types(
        self,
        statements: List[str],
        num_questions: int,
        question_types: List[str],
        difficulty: str,
        custom_guidelines: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate questions with a balanced distribution of types.

        Args:
            statements: List of statements to base questions on
            num_questions: Target number of questions to generate
            question_types: List of question types to generate
            difficulty: Difficulty level of questions
            custom_guidelines: Optional custom guidelines for question generation

        Returns:
            List of questions with balanced types
        """
        # Calculate how many questions of each type to generate
        type_counts = {}
        num_types = len(question_types)

        # Distribute questions evenly
        base_count = num_questions // num_types
        remainder = num_questions % num_types

        for i, q_type in enumerate(question_types):
            # Add extra question to some types if there's a remainder
            extra = 1 if i < remainder else 0
            type_counts[q_type] = base_count + extra

        # Generate questions for each type
        all_questions = []
        for q_type, count in type_counts.items():
            if count > 0:
                # self.info(f"Generating {count} {q_type} questions...")
                questions = await self.generate_questions(
                    statements=statements,
                    num_questions=count,
                    question_types=[q_type],
                    difficulty=difficulty,
                    custom_guidelines=custom_guidelines,
                )
                all_questions.extend(questions)

        return all_questions
