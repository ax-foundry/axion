import re
from typing import Dict, List, Optional, Union

import numpy as np
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
from pydantic import Field

logger = get_logger(__name__)


class AspectDecomposerInput(RichBaseModel):
    query: str = Field(..., description='The original query to analyze')
    expected_aspects: List[str] = Field(
        ..., description='List of key aspects that should be addressed'
    )
    model_config = {'extra': 'forbid'}


class AspectDecomposerOutput(RichBaseModel):
    aspects_details: Dict[str, List[str]] = Field(
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
    description = 'Salesforce Query Aspect Analyzer'
    examples = [
        (
            AspectDecomposerInput(
                query='How can I set up Agentforce and Data Cloud in Salesforce for our sales team to increase productivity through Slack?',
                expected_aspects=[
                    'Agentforce',
                    'Data Cloud',
                    'Slack',
                    'SDR',
                ],
            ),
            AspectDecomposerOutput(
                aspects_details={
                    'Agentforce': [
                        'Setup or configuration',
                    ],
                    'Data Cloud': [
                        'Setup or configuration',
                    ],
                    'Slack': [
                        'Salesforce-Slack integration',
                    ],
                    'SDR': [
                        'Productivity features for sales team',
                    ],
                }
            ),
        )
    ]


class AspectCompletenessCheckerInput(RichBaseModel):
    query: str = Field(..., description='The original query')
    response: str = Field(..., description='The response to evaluate')
    expected_aspects: List[str] = Field(
        ..., description='List of key aspects that should be addressed'
    )
    aspect_details: Dict[str, List[str]] = Field(
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
    description = 'Salesforce Response Aspect Coverage Checker'
    examples = [
        (
            AspectCompletenessCheckerInput(
                query='How can I set up Agentforce and Data Cloud in Salesforce for our sales team to increase productivity through Slack?',
                response="To set up Agentforce in Salesforce, first ensure you have the appropriate licenses. Navigate to Setup > Einstein > Agentforce and complete the initial configuration. You'll need to enable relevant features and assign user permissions. For Data Cloud setup, go to Setup > Integration > Data Cloud and follow the guided setup. You'll need to connect your data sources and configure data mapping.",
                expected_aspects=[
                    'Agentforce',
                    'Data Cloud',
                    'Slack',
                    'SDR',
                ],
                aspect_details={
                    'Agentforce': [
                        'Setup or configuration',
                    ],
                    'Data Cloud': [
                        'Setup or configuration',
                    ],
                    'Slack': [
                        'Salesforce-Slack integration',
                    ],
                    'SDR': [
                        'Productivity features for sales team',
                    ],
                },
            ),
            AspectCompletenessCheckerOutput(
                aspect_results=[
                    AspectCoverageResult(
                        aspect='Agentforce',
                        covered=True,
                        concepts_covered=[
                            'Setup or configuration',
                        ],
                        concepts_missing=[],
                        reason='The response covers Agentforce setup including navigation path and configuration steps.',
                    ),
                    AspectCoverageResult(
                        aspect='Data Cloud',
                        covered=True,
                        concepts_covered=[
                            'Setup or configuration',
                        ],
                        concepts_missing=[],
                        reason='The response covers Data Cloud setup process and mentions data source connections.',
                    ),
                    AspectCoverageResult(
                        aspect='Slack',
                        covered=False,
                        concepts_covered=[],
                        concepts_missing=[
                            'Salesforce-Slack integration',
                        ],
                        reason='The response does not mention Slack integration at all.',
                    ),
                    AspectCoverageResult(
                        aspect='SDR',
                        covered=False,
                        concepts_covered=[],
                        concepts_missing=[
                            'Productivity features for sales team',
                        ],
                        reason='The response does not address how these tools help with sales team productivity.',
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
                query='What is Sales Cloud?',
                expected_output='Sales Cloud is a CRM platform that helps manage leads, track opportunities, and automate sales processes. It is designed for B2B sales teams.',
            ),
            ExpectedAnswerAnalyzerOutput(
                key_points=[
                    'What is the primary function of Sales Cloud?',
                    'What specific features does it offer (leads, opportunities, automation)?',
                    'Who is the target audience for Sales Cloud?',
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
    description = 'Salesforce Query Decomposer'
    examples = [
        (
            QueryDecomposerInput(
                query='What are the key differences between Salesforce Sales Cloud and Service Cloud, and which would be better for our small business with 15 employees focused on customer retention?',
            ),
            QueryDecomposerOutput(
                sub_questions=[
                    'What are the key differences between Sales Cloud and Service Cloud?',
                    'Which platform is better for customer retention and small businesses?',
                ]
            ),
        ),
        (
            QueryDecomposerInput(
                query='How can I set up Agentforce and Data Cloud in Salesforce for our sales team to increase productivity through Slack?'
            ),
            QueryDecomposerOutput(
                sub_questions=[
                    'How do you set up Agentforce and Data Cloud in Salesforce?',
                    'How do you integrate with Slack for sales team productivity?',
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
    description = 'Salesforce Response Sub-Question Checker'
    examples = [
        (
            SubQuestionCheckerInput(
                query="What's the difference between Salesforce Professional and Enterprise editions, and which one should our 50-person company choose?",
                response='Salesforce Professional Edition offers essential CRM features including lead and opportunity management, customizable dashboards, and standard reports. It allows for up to 5 custom apps and limited API access. Enterprise Edition builds on these capabilities with more advanced features such as workflow automation, approval processes, and unlimited custom apps. It also provides full API access and supports more complex organizational structures through role hierarchies.',
                sub_questions=[
                    'What are the key differences between the two editions?',
                    'Which edition is recommended for a 50-person company?',
                ],
            ),
            SubQuestionCheckerOutput(
                results=[
                    SubQuestionResult(
                        sub_question='What are the key differences between the two editions?',
                        addressed=True,
                        reason='The response highlights key differences such as custom app limits, API access levels, workflow automation, and organizational structure support.',
                    ),
                    SubQuestionResult(
                        sub_question='Which edition is recommended for a 50-person company?',
                        addressed=False,
                        reason='The response does not provide a specific recommendation for a 50-person company.',
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
    ) -> Dict[str, List[str]]:
        """
        Analyze the expected aspects to determine key concepts that should be mentioned.

        Args:
            query: The original query
            expected_aspects: List of key aspects that should be addressed

        Returns:
            Dict mapping aspects to their key concepts
        """
        result = await self.aspect_decomposer.execute(
            AspectDecomposerInput(query=query, expected_aspects=expected_aspects)
        )

        # Check if aspects_details contains entries for all expected aspects
        if result.aspects_details:
            missing_aspects = [
                aspect
                for aspect in expected_aspects
                if aspect not in result.aspects_details
            ]
            if missing_aspects:
                logger.warning(f'Some aspects were not analyzed: {missing_aspects}')

                # Add default concepts for missing aspects
                for aspect in missing_aspects:
                    result.aspects_details[aspect] = [
                        f'{aspect} overview or explanation',
                    ]
        else:
            logger.warning(
                'No aspect details were generated. Creating default concepts.'
            )
            result.aspects_details = {}
            for aspect in expected_aspects:
                result.aspects_details[aspect] = [
                    f'{aspect} overview or explanation',
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
        aspect_details: Dict[str, List[str]],
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
                query=item.query,
                response=item.actual_output,
                expected_aspects=expected_aspects,
                aspect_details=aspect_details,
                expected_output=item.expected_output,
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
                query=item.query,
                response=item.actual_output,
                sub_questions=sub_questions,
                expected_output=item.expected_output,
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
        expected_aspects = item.acceptance_criteria

        if expected_aspects:
            logger.info('Using aspect-based evaluation approach')
            if not isinstance(expected_aspects, list):
                raise TypeError(
                    'acceptance_criteria must be a list for aspect-based evaluation.'
                )

            aspect_details = await self._analyze_aspects(item.query, expected_aspects)
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
            if self.use_expected_output and item.expected_output:
                sub_questions = await self._analyze_expected_answer(
                    item.query, item.expected_output
                )
            else:
                sub_questions = await self._decompose_query(item.query)

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
