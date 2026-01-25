import re
from typing import Dict, List, Optional, Union

import numpy as np
from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.schema import SignalDescriptor

logger = get_logger(__name__)


class AspectDecomposerInput(RichBaseModel):
    query: str = Field(..., description='The original query to analyze')
    expected_aspects: List[str] = Field(
        ..., description='List of key aspects that should be addressed'
    )
    model_config = {'extra': 'forbid'}


class AspectDetail(RichBaseModel):
    """Key concepts for a specific aspect that should be mentioned."""

    aspect_name: str = Field(description='The name of the aspect')
    concepts: List[str] = Field(
        description='Key concepts for this aspect that should be mentioned'
    )
    model_config = {'extra': 'forbid'}


class AspectDecomposerOutput(RichBaseModel):
    aspects_details: List[AspectDetail] = Field(
        ...,
        description='Key concepts for each expected aspect that should be mentioned',
    )
    model_config = {'extra': 'forbid'}


class AspectDecomposer(BaseMetric[AspectDecomposerInput, AspectDecomposerOutput]):
    instruction = """For the given query, analyze the expected aspects that should be addressed in a complete response.
    For each expected aspect in the provided list, identify 1-2 key concepts or topics that should be mentioned
    to demonstrate the aspect was addressed. Focus on the main ideas rather than exhaustive implementation details.
    """

    input_model = AspectDecomposerInput
    output_model = AspectDecomposerOutput
    description = 'Query Aspect Analyzer'
    examples = [
        (
            AspectDecomposerInput(
                query='How can I configure Python virtual environments and pip for our development team to improve dependency management through VS Code?',
                expected_aspects=[
                    'Virtual Environments',
                    'pip',
                    'VS Code',
                    'Development Team',
                ],
            ),
            AspectDecomposerOutput(
                aspects_details=[
                    AspectDetail(
                        aspect_name='Virtual Environments',
                        concepts=['Setup or configuration'],
                    ),
                    AspectDetail(
                        aspect_name='pip',
                        concepts=['Package management setup'],
                    ),
                    AspectDetail(
                        aspect_name='VS Code',
                        concepts=['Python-VS Code integration'],
                    ),
                    AspectDetail(
                        aspect_name='Development Team',
                        concepts=['Workflow improvements for developers'],
                    ),
                ],
            ),
        )
    ]


class AspectCompletenessCheckerInput(RichBaseModel):
    query: str = Field(..., description='The original query')
    response: str = Field(..., description='The response to evaluate')
    expected_aspects: List[str] = Field(
        ..., description='List of key aspects that should be addressed'
    )
    aspect_details: List[AspectDetail] = Field(
        ..., description='Key concepts for each expected aspect'
    )
    expected_output: Optional[str] = Field(
        None,
        description='Optional gold standard comprehensive answer to compare against',
    )

    model_config = {'extra': 'forbid'}


class AspectCoverageResult(RichBaseModel):
    aspect: str = Field(..., description='The expected aspect that was evaluated')
    covered: bool = Field(
        ..., description='Whether the aspect was meaningfully addressed in the response'
    )
    concepts_covered: Optional[List[str]] = Field(
        None, description='The key concepts that were addressed'
    )
    concepts_missing: Optional[List[str]] = Field(
        None, description='The key concepts that were not addressed'
    )
    reason: str = Field(..., description='The reason for the coverage determination')

    model_config = {'extra': 'forbid'}


class AspectCompletenessCheckerOutput(RichBaseModel):
    aspect_results: List[AspectCoverageResult] = Field(
        ..., description='The evaluation results for each expected aspect'
    )
    model_config = {'extra': 'forbid'}


class AspectCompletenessChecker(
    BaseMetric[AspectCompletenessCheckerInput, AspectCompletenessCheckerOutput]
):
    instruction = """Evaluate whether each expected aspect of the query has been meaningfully addressed in the provided response.

    For each expected aspect:
    1. Determine if the aspect is covered (true if any relevant information is provided, false if completely missing)
    2. Identify which key concepts are mentioned and which are missing
    3. Provide a brief explanation for your verdict

    Be generous in your evaluation - if the response mentions the aspect and provides some relevant information,
    consider it covered even if not all details are present. The goal is to check if the main topics are addressed,
    not to require exhaustive coverage of every possible detail.

    If expected_output is provided, use it as a reference to help determine coverage standards.
    """

    input_model = AspectCompletenessCheckerInput
    output_model = AspectCompletenessCheckerOutput
    description = 'Response Aspect Coverage Checker'
    examples = [
        (
            AspectCompletenessCheckerInput(
                query='How can I configure Python virtual environments and pip for our development team to improve dependency management through VS Code?',
                response="To set up Python virtual environments, first ensure you have Python installed. Use 'python -m venv myenv' to create a virtual environment, then activate it using the appropriate command for your OS. For pip setup, run 'pip install --upgrade pip' within your environment. You'll need to create a requirements.txt file to manage dependencies.",
                expected_aspects=[
                    'Virtual Environments',
                    'pip',
                    'VS Code',
                    'Development Team',
                ],
                aspect_details=[
                    AspectDetail(
                        aspect_name='Virtual Environments',
                        concepts=['Setup or configuration'],
                    ),
                    AspectDetail(
                        aspect_name='pip',
                        concepts=['Package management setup'],
                    ),
                    AspectDetail(
                        aspect_name='VS Code',
                        concepts=['Python-VS Code integration'],
                    ),
                    AspectDetail(
                        aspect_name='Development Team',
                        concepts=['Workflow improvements for developers'],
                    ),
                ],
            ),
            AspectCompletenessCheckerOutput(
                aspect_results=[
                    AspectCoverageResult(
                        aspect='Virtual Environments',
                        covered=True,
                        concepts_covered=[
                            'Setup or configuration',
                        ],
                        concepts_missing=[],
                        reason='The response covers virtual environment setup including the venv command and activation.',
                    ),
                    AspectCoverageResult(
                        aspect='pip',
                        covered=True,
                        concepts_covered=[
                            'Package management setup',
                        ],
                        concepts_missing=[],
                        reason='The response covers pip setup and mentions requirements.txt for dependency management.',
                    ),
                    AspectCoverageResult(
                        aspect='VS Code',
                        covered=False,
                        concepts_covered=[],
                        concepts_missing=[
                            'Python-VS Code integration',
                        ],
                        reason='The response does not mention VS Code integration at all.',
                    ),
                    AspectCoverageResult(
                        aspect='Development Team',
                        covered=False,
                        concepts_covered=[],
                        concepts_missing=[
                            'Workflow improvements for developers',
                        ],
                        reason='The response does not address how these tools help with team workflow.',
                    ),
                ]
            ),
        )
    ]


class ExpectedAnswerAnalyzerInput(RichBaseModel):
    query: str = Field(..., description='The original user query')
    expected_output: str = Field(..., description='The gold standard answer to analyze')
    model_config = {'extra': 'forbid'}


class ExpectedAnswerAnalyzerOutput(RichBaseModel):
    key_points: List[str] = Field(
        ..., description='Key sub-questions or topics identified in the expected answer'
    )
    model_config = {'extra': 'forbid'}


class ExpectedAnswerAnalyzer(
    BaseMetric[ExpectedAnswerAnalyzerInput, ExpectedAnswerAnalyzerOutput]
):
    instruction = """Analyze the provided 'Expected Output' in the context of the 'Query'.
    Identify the distinct information components, steps, or topics that are present in the Expected Output.
    Convert these into a list of specific sub-questions or key points that a candidate response must address to be considered complete.
    The goal is to create a checklist of content that MUST be present.
    """

    input_model = ExpectedAnswerAnalyzerInput
    output_model = ExpectedAnswerAnalyzerOutput
    description = 'Extracts key coverage points from the expected answer.'
    examples = [
        (
            ExpectedAnswerAnalyzerInput(
                query='What is the infield fly rule?',
                expected_output='The infield fly rule is a baseball rule that protects baserunners by declaring the batter out on certain easy pop-ups. It applies with fewer than two outs and runners on first and second or bases loaded.',
            ),
            ExpectedAnswerAnalyzerOutput(
                key_points=[
                    'What is the primary purpose of the infield fly rule?',
                    'Under what conditions does the rule apply (outs, runners)?',
                    'Who does the rule protect and how?',
                ]
            ),
        )
    ]


class QueryDecomposerInput(RichBaseModel):
    query: str = Field(..., description='The original query to decompose')
    model_config = {'extra': 'forbid'}


class QueryDecomposerOutput(RichBaseModel):
    sub_questions: List[str] = Field(
        ..., description='The main sub-questions from the original query'
    )
    model_config = {'extra': 'forbid'}


class SubQuestionResult(RichBaseModel):
    sub_question: str = Field(..., description='The sub-question that was evaluated')
    addressed: bool = Field(
        ...,
        description='Whether the sub-question was meaningfully addressed in the response',
    )
    reason: str = Field(..., description='The reason for the verdict')
    model_config = {'extra': 'forbid'}


class QueryDecomposer(BaseMetric[QueryDecomposerInput, QueryDecomposerOutput]):
    instruction = """Break down the given query into 2-3 main sub-questions that capture the key information requests.
    Focus on the primary questions that need to be answered rather than every possible detail.
    Each sub-question should represent a major component of what the user is asking."""

    input_model = QueryDecomposerInput
    output_model = QueryDecomposerOutput
    description = 'Query Decomposer'
    examples = [
        (
            QueryDecomposerInput(
                query='What are the key differences between Python lists and tuples, and which would be better for our data processing pipeline that needs immutability?',
            ),
            QueryDecomposerOutput(
                sub_questions=[
                    'What are the key differences between Python lists and tuples?',
                    'Which data structure is better for immutability requirements?',
                ]
            ),
        ),
        (
            QueryDecomposerInput(
                query='How can I set up virtual environments and pip for our development team to improve dependency management through VS Code?'
            ),
            QueryDecomposerOutput(
                sub_questions=[
                    'How do you set up virtual environments and pip?',
                    'How do you integrate with VS Code for team development?',
                ]
            ),
        ),
    ]


class SubQuestionCheckerInput(RichBaseModel):
    query: str = Field(description='The original query')
    response: str = Field(description='The response to evaluate')
    sub_questions: List[str] = Field(
        description='The sub-questions to check in the response'
    )
    expected_output: Optional[str] = Field(
        default=None,
        description='Optional gold standard comprehensive answer to compare against',
    )
    model_config = {'extra': 'forbid'}


class SubQuestionCheckerOutput(RichBaseModel):
    results: List[SubQuestionResult] = Field(
        description='The evaluation results for each sub-question'
    )
    model_config = {'extra': 'forbid'}


class SubQuestionChecker(BaseMetric[SubQuestionCheckerInput, SubQuestionCheckerOutput]):
    instruction = """Evaluate whether each sub-question derived from the original query has been meaningfully addressed in the provided response.

    For each sub-question, determine if it was addressed (true if relevant information is provided, false if not mentioned).
    Be reasonable in your evaluation - if the response provides some relevant information about the sub-question,
    consider it addressed even if the answer is not comprehensive.

    Provide a brief explanation for each judgment."""

    input_model = SubQuestionCheckerInput
    output_model = SubQuestionCheckerOutput
    description = 'Response Sub-Question Checker'
    examples = [
        (
            SubQuestionCheckerInput(
                query="What's the difference between Python lists and tuples, and which one should our data pipeline use for configuration values?",
                response='Python lists are mutable, ordered collections that can be modified after creation. They support methods like append(), insert(), and remove(). Tuples are immutable, ordered collections that cannot be changed once created. They are typically faster and use less memory than lists. Tuples also support tuple unpacking and can be used as dictionary keys.',
                sub_questions=[
                    'What are the key differences between lists and tuples?',
                    'Which data structure is recommended for configuration values?',
                ],
            ),
            SubQuestionCheckerOutput(
                results=[
                    SubQuestionResult(
                        sub_question='What are the key differences between lists and tuples?',
                        addressed=True,
                        reason='The response highlights key differences such as mutability, available methods, performance, and use cases like dictionary keys.',
                    ),
                    SubQuestionResult(
                        sub_question='Which data structure is recommended for configuration values?',
                        addressed=False,
                        reason='The response does not provide a specific recommendation for configuration values.',
                    ),
                ]
            ),
        )
    ]


class CompletenessAspectBreakdown(RichBaseModel):
    """Detailed breakdown for a single evaluated aspect."""

    aspect: str
    covered: bool
    concepts_covered: List[str]
    concepts_missing: List[str]
    concepts_coverage_percentage: float
    reason: str


class AnswerCompletenessAspectResult(RichBaseModel):
    """Structured result for an aspect-based completeness evaluation."""

    evaluation_type: str = 'aspect'
    score: float
    covered_aspects_count: int
    total_aspects_count: int
    concept_coverage_score: float
    aspect_breakdown: List[CompletenessAspectBreakdown]


class CompletenessSubQuestionBreakdown(RichBaseModel):
    """Detailed breakdown for a single evaluated sub-question."""

    sub_question: str
    addressed: bool
    reason: str


class AnswerCompletenessSubQuestionResult(RichBaseModel):
    """Structured result for a sub-question-based completeness evaluation."""

    evaluation_type: str = 'sub_question'
    score: float
    addressed_count: int
    total_count: int
    sub_question_breakdown: List[CompletenessSubQuestionBreakdown]


AnswerCompletenessResult = Union[
    AnswerCompletenessAspectResult, AnswerCompletenessSubQuestionResult
]


@metric(
    name='Answer Completeness',
    key='answer_completeness',
    description='Evaluates the completeness of the answer using one of two approaches: Either. Aspect-based evaluation or sub-question based evaluation.',
    required_fields=['query', 'actual_output', 'expected_output'],
    optional_fields=['acceptance_criteria'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['knowledge', 'agent', 'single_turn'],
)
class AnswerCompleteness(BaseMetric):
    """
    Evaluates the completeness of product-related responses using one of two approaches:
    1. Aspect-based evaluation (when expected_aspects are provided)
    2. Sub-question based evaluation (when expected_aspects are not provided)
    """

    def __init__(self, use_expected_output: bool = True, **kwargs):
        """
        Initialize the answer completeness metric with required prompts for both approaches.

        Args:
            use_expected_output: It True, use expected answer if available, otherwise decompose query
        """
        super().__init__(**kwargs)
        self.use_expected_output = use_expected_output

        # Aspect-based completeness
        self.aspect_decomposer = AspectDecomposer(**kwargs)
        self.aspect_completeness_checker = AspectCompletenessChecker(**kwargs)

        # Sub-question based completeness
        self.expected_answer_analyzer = ExpectedAnswerAnalyzer(**kwargs)
        self.query_decomposer = QueryDecomposer(**kwargs)
        self.sub_question_checker = SubQuestionChecker(**kwargs)

    @trace(name='analyze_aspects', capture_args=True, capture_response=True)
    async def _analyze_aspects(
        self, query: str, expected_aspects: List[str]
    ) -> List[AspectDetail]:
        """
        Analyze the expected aspects to determine key concepts that should be mentioned.

        Args:
            query: The original query
            expected_aspects: List of key aspects that should be addressed

        Returns:
            List of AspectDetail with concepts for each aspect
        """
        result = await self.aspect_decomposer.execute(
            AspectDecomposerInput(query=query, expected_aspects=expected_aspects)
        )

        # Build a set of aspect names that have details
        aspects_with_details = {ad.aspect_name for ad in result.aspects_details}

        # Check if aspects_details contains entries for all expected aspects
        if result.aspects_details:
            missing_aspects = [
                aspect
                for aspect in expected_aspects
                if aspect not in aspects_with_details
            ]
            if missing_aspects:
                logger.warning(f'Some aspects were not analyzed: {missing_aspects}')

                # Add default concepts for missing aspects
                for aspect in missing_aspects:
                    result.aspects_details.append(
                        AspectDetail(
                            aspect_name=aspect,
                            concepts=[f'{aspect} overview or explanation'],
                        )
                    )
        else:
            logger.warning(
                'No aspect details were generated. Creating default concepts.'
            )
            result.aspects_details = [
                AspectDetail(
                    aspect_name=aspect,
                    concepts=[f'{aspect} overview or explanation'],
                )
                for aspect in expected_aspects
            ]

        return result.aspects_details

    @trace(name='decompose_query', capture_args=True, capture_response=True)
    async def _decompose_query(self, query: str) -> List[str]:
        """
        Break down the original query into main sub-questions.

        Args:
            query: The original query

        Returns:
            List of sub-questions
        """
        result = await self.query_decomposer.execute(QueryDecomposerInput(query=query))

        if not result.sub_questions:
            logger.warning('No sub-questions were generated from the query.')

        return result.sub_questions

    @trace(name='analyze_expected_answer', capture_args=True, capture_response=True)
    async def _analyze_expected_answer(
        self, query: str, expected_output: str
    ) -> List[str]:
        """
        Extract key topics or sub-questions directly from the expected answer.

        Args:
            query: The original query
            expected_output: The gold standard answer

        Returns:
            List of sub-questions/topics
        """
        result = await self.expected_answer_analyzer.execute(
            ExpectedAnswerAnalyzerInput(query=query, expected_output=expected_output)
        )

        if not result.key_points:
            logger.warning('No key points were generated from the expected answer.')

        return result.key_points

    @trace(name='check_aspect_completeness', capture_args=True, capture_response=True)
    async def _check_aspect_completeness(
        self,
        item: DatasetItem,
        expected_aspects: List[str],
        aspect_details: List[AspectDetail],
    ) -> AspectCompletenessCheckerOutput:
        """
        Check aspect coverage in the response.

        Args:
            item: DatasetItem
            aspect_details: Key concepts for each expected aspect

        Returns:
            Evaluation results for aspects
        """
        return await self.aspect_completeness_checker.execute(
            AspectCompletenessCheckerInput(
                query=self.get_field(item, 'query'),
                response=self.get_field(item, 'actual_output'),
                expected_aspects=expected_aspects,
                aspect_details=aspect_details,
                expected_output=self.get_field(item, 'expected_output'),
            )
        )

    @trace(
        name='check_sub_question_completeness', capture_args=True, capture_response=True
    )
    async def _check_sub_question_completeness(
        self, item: DatasetItem, sub_questions: List[str]
    ) -> SubQuestionCheckerOutput:
        """
        Check if each sub-question is addressed in the response.

        Args:
            item: Test case item containing query, actual output, and expected output
            sub_questions: List of sub-questions to check

        Returns:
            Evaluation result for each sub-question
        """
        return await self.sub_question_checker.execute(
            SubQuestionCheckerInput(
                query=self.get_field(item, 'query'),
                response=self.get_field(item, 'actual_output'),
                sub_questions=sub_questions,
                expected_output=self.get_field(item, 'expected_output'),
            )
        )

    @trace(name='compute_aspect_score', capture_args=True, capture_response=True)
    def _compute_aspect_score(
        self,
        evaluation_results: AspectCompletenessCheckerOutput,
    ) -> Dict:
        """
        Compute the aspect coverage score and detailed breakdown.

        Args:
            evaluation_results: Results of aspect evaluations.

        Returns:
            Dict containing aspect coverage score and detailed breakdown.
        """
        aspect_results = evaluation_results.aspect_results or []

        if not aspect_results:
            logger.warning('No aspects were evaluated.')
            return {
                'score': np.nan,
                'covered_aspects_count': 0,
                'total_aspects_count': 0,
                'percentage': np.nan,
                'concept_coverage_score': np.nan,
                'aspect_breakdown': [],
                'evaluation_type': 'aspect',
            }

        # Aspect-level coverage
        covered_aspects = [result for result in aspect_results if result.covered]
        covered_count = len(covered_aspects or [])
        total_count = len(aspect_results or [])

        aspect_coverage_score = covered_count / total_count if total_count else np.nan
        aspect_coverage_percentage = (
            aspect_coverage_score * 100
            if not np.isnan(aspect_coverage_score)
            else np.nan
        )

        # Concept-level coverage
        total_concepts_covered = sum(
            len(result.concepts_covered or []) for result in aspect_results
        )
        total_concepts = sum(
            len(result.concepts_covered or []) + len(result.concepts_missing or [])
            for result in aspect_results
        )

        concept_coverage_score = (
            total_concepts_covered / total_concepts if total_concepts else np.nan
        )

        # Aspect breakdown
        aspect_breakdown = []
        for result in aspect_results:
            concepts_total = len(result.concepts_covered or []) + len(
                result.concepts_missing or []
            )
            concepts_coverage_percentage = (
                (len(result.concepts_covered or []) / concepts_total * 100)
                if concepts_total
                else np.nan
            )

            aspect_breakdown.append(
                {
                    'aspect': result.aspect,
                    'covered': result.covered,
                    'concepts_covered': result.concepts_covered or [],
                    'concepts_missing': result.concepts_missing or [],
                    'concepts_coverage_percentage': concepts_coverage_percentage,
                    'reason': result.reason,
                }
            )

        return {
            'score': aspect_coverage_score,
            'covered_aspects_count': covered_count,
            'total_aspects_count': total_count,
            'percentage': aspect_coverage_percentage,
            'concept_coverage_score': concept_coverage_score,
            'aspect_breakdown': aspect_breakdown,
            'evaluation_type': 'aspect',
        }

    @trace(name='compute_sub_question_score', capture_args=True, capture_response=True)
    def _compute_sub_question_score(
        self, evaluation_results: SubQuestionCheckerOutput
    ) -> Dict:
        """
        Compute the sub-question coverage score and detailed breakdown.

        Args:
            evaluation_results: Results of sub-question evaluations

        Returns:
            Dict containing sub-question coverage score and detailed breakdown
        """
        with self.tracer.span('sub_question_score'):
            results = evaluation_results.results
            if not results:
                logger.warning('No sub-questions were evaluated.')
                return {
                    'score': np.nan,
                    'addressed_count': 0,
                    'total_count': 0,
                    'percentage': np.nan,
                    'sub_question_breakdown': [],
                    'evaluation_type': 'sub_question',
                }

            addressed_count = sum(1 for result in results if result.addressed)
            total_count = len(results)
            percentage = (
                (addressed_count / total_count) * 100 if total_count > 0 else np.nan
            )
            sub_question_coverage_score = (
                addressed_count / total_count if total_count > 0 else np.nan
            )

            sub_question_breakdown = [
                {
                    'sub_question': result.sub_question,
                    'addressed': result.addressed,
                    'reason': result.reason,
                }
                for result in results
            ]

            return {
                'score': sub_question_coverage_score,  # Primary score (0-1)
                'addressed_count': addressed_count,
                'total_count': total_count,
                'percentage': percentage,
                'sub_question_breakdown': sub_question_breakdown,
                'evaluation_type': 'sub_question',
            }

    @trace(name='AnswerCompleteness', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """
        Compute the completeness score, returning a structured result object
        in the `signals` field.
        """
        self._validate_required_metric_fields(item)
        expected_aspects = self.get_field(item, 'acceptance_criteria')

        if expected_aspects:
            logger.info('Using aspect-based evaluation approach')
            if not isinstance(expected_aspects, list):
                raise TypeError(
                    'acceptance_criteria must be a list for aspect-based evaluation.'
                )

            query = self.get_field(item, 'query')
            aspect_details = await self._analyze_aspects(query, expected_aspects)
            if not aspect_details:
                return MetricEvaluationResult(
                    score=np.nan, explanation='Could not analyze aspect details.'
                )

            eval_results = await self._check_aspect_completeness(
                item, expected_aspects, aspect_details
            )
            score_data = self._compute_aspect_score(eval_results)
            result_data = AnswerCompletenessAspectResult(
                score=score_data['score'],
                covered_aspects_count=score_data['covered_aspects_count'],
                total_aspects_count=score_data['total_aspects_count'],
                concept_coverage_score=score_data['concept_coverage_score'],
                aspect_breakdown=[
                    CompletenessAspectBreakdown(**b)
                    for b in score_data['aspect_breakdown']
                ],
            )
            return MetricEvaluationResult(score=result_data.score, signals=result_data)

        else:
            logger.info('Using sub-question based evaluation approach')

            # Use expected answer if available, otherwise decompose query
            query = self.get_field(item, 'query')
            expected_output = self.get_field(item, 'expected_output')
            if self.use_expected_output and expected_output:
                sub_questions = await self._analyze_expected_answer(
                    query, expected_output
                )
            else:
                sub_questions = await self._decompose_query(query)

            if not sub_questions:
                return MetricEvaluationResult(
                    score=np.nan,
                    explanation='Could not decompose query or expected answer into sub-questions.',
                )

            eval_results = await self._check_sub_question_completeness(
                item, sub_questions
            )
            score_data = self._compute_sub_question_score(eval_results)

            self.compute_cost_estimate(
                [
                    self.aspect_decomposer,
                    self.aspect_completeness_checker,
                    self.expected_answer_analyzer,
                    self.query_decomposer,
                    self.sub_question_checker,
                ]
            )

            result_data = AnswerCompletenessSubQuestionResult(
                score=score_data['score'],
                addressed_count=score_data['addressed_count'],
                total_count=score_data['total_count'],
                sub_question_breakdown=[
                    CompletenessSubQuestionBreakdown(**b)
                    for b in score_data['sub_question_breakdown']
                ],
            )
            return MetricEvaluationResult(score=result_data.score, signals=result_data)

    @staticmethod
    def get_signals(
        result: AnswerCompletenessResult,
    ) -> List[SignalDescriptor[AnswerCompletenessResult]]:
        """Generates a list of detailed signals from the evaluation result."""
        signals = []

        # Aspect-Based Results
        if isinstance(result, AnswerCompletenessAspectResult):
            signals.append(
                SignalDescriptor(
                    name='aspect_coverage_score',
                    group='overall',
                    description=f'Final score based on aspect coverage: {result.covered_aspects_count} of'
                    f' {result.total_aspects_count} aspects were covered.',
                    extractor=lambda r: r.score,
                    headline_display=True,
                )
            )
            signals.append(
                SignalDescriptor(
                    name='concept_coverage_score',
                    group='overall',
                    description='Secondary score based on concept coverage.',
                    extractor=lambda r: r.concept_coverage_score,
                )
            )

            for i, aspect in enumerate(result.aspect_breakdown):
                sanitized_name = re.sub(r'[^\w\s-]', '', aspect.aspect).strip()
                sanitized_name = re.sub(r'[-\s]+', '_', sanitized_name)
                group_name = f'aspect_{sanitized_name}'

                signals.extend(
                    [
                        SignalDescriptor(
                            name='is_covered',
                            group=group_name,
                            description='Whether the aspect was covered.',
                            extractor=lambda r, idx=i: r.aspect_breakdown[idx].covered,
                            headline_display=True,
                        ),
                        SignalDescriptor(
                            name='concept_coverage',
                            group=group_name,
                            description='Percentage of concepts covered for this aspect.',
                            extractor=lambda r, idx=i: r.aspect_breakdown[
                                idx
                            ].concepts_coverage_percentage
                            / 100.0,
                        ),
                        SignalDescriptor(
                            name='aspect',
                            group=group_name,
                            description='The name of the evaluated aspect.',
                            extractor=lambda r, idx=i: r.aspect_breakdown[idx].aspect,
                        ),
                        SignalDescriptor(
                            name='concepts_covered',
                            group=group_name,
                            description='List of concepts covered.',
                            extractor=lambda r, idx=i: r.aspect_breakdown[
                                idx
                            ].concepts_covered,
                        ),
                        SignalDescriptor(
                            name='concepts_missing',
                            group=group_name,
                            description='List of concepts missing.',
                            extractor=lambda r, idx=i: r.aspect_breakdown[
                                idx
                            ].concepts_missing,
                        ),
                        SignalDescriptor(
                            name='reason',
                            group=group_name,
                            description='LLM-provided reason for the evaluation.',
                            extractor=lambda r, idx=i: r.aspect_breakdown[idx].reason,
                        ),
                    ]
                )

        # Sub-Question-Based Results
        elif isinstance(result, AnswerCompletenessSubQuestionResult):
            signals.append(
                SignalDescriptor(
                    name='sub_question_coverage_score',
                    group='overall',
                    description=f'Final score based on sub-question coverage: {result.addressed_count} of {result.total_count} sub-questions were addressed.',
                    extractor=lambda r: r.score,
                    headline_display=True,
                )
            )

            for i, sub_q in enumerate(result.sub_question_breakdown):
                sanitized_name = re.sub(r'[^\w\s-]', '', sub_q.sub_question).strip()[
                    :30
                ]
                sanitized_name = re.sub(r'[-\s]+', '_', sanitized_name)
                group_name = f'subq_{sanitized_name}'

                signals.extend(
                    [
                        SignalDescriptor(
                            name='is_addressed',
                            group=group_name,
                            description='Whether the sub-question was addressed.',
                            extractor=lambda r, idx=i: r.sub_question_breakdown[
                                idx
                            ].addressed,
                            headline_display=True,
                        ),
                        SignalDescriptor(
                            name='sub_question',
                            group=group_name,
                            description='The evaluated sub-question.',
                            extractor=lambda r, idx=i: r.sub_question_breakdown[
                                idx
                            ].sub_question,
                        ),
                        SignalDescriptor(
                            name='reason',
                            group=group_name,
                            description='LLM-provided reason for the evaluation.',
                            extractor=lambda r, idx=i: r.sub_question_breakdown[
                                idx
                            ].reason,
                        ),
                    ]
                )

        return signals
