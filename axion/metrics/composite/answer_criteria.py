import re
from typing import Dict, List, Literal, Optional, Set

import numpy as np
from pydantic import Field, field_validator

from axion._core.logging import get_logger
from axion._core.schema import AIMessage, HumanMessage, RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.schema import SignalDescriptor

logger = get_logger(__name__)


class CriteriaDecomposerInput(RichBaseModel):
    query: str = Field(description='The original query to analyze')
    criteria: str = Field(
        description='The criteria text describing what a complete answer should include'
    )


class CriteriaDecomposerOutput(RichBaseModel):
    key_aspects: List[str] = Field(
        description='The key aspects that should be addressed based on the criteria'
    )
    aspect_details: Dict[str, List[str]] = Field(
        description='Key concepts for each aspect that should be mentioned'
    )


class CriteriaDecomposer(BaseMetric[CriteriaDecomposerInput, CriteriaDecomposerOutput]):
    instruction = """Analyze the criteria text to create a structured evaluation rubric by extracting its core aspects and their specific, verifiable concepts.

    Your breakdown should follow these steps:

    1.  **Identify Core Aspects:** Group related requirements from the criteria into distinct, high-level aspects. Each aspect should represent a major topic that a downstream model should answer.

    2.  **Extract Verifiable Concepts:** For each aspect, identify 2-4 specific, verifiable facts, actions, or details mentioned in the criteria that prove the aspect has been covered.
        * **CRITICAL:** Concepts must be concrete details. **Avoid** generating concepts that are just vague restatements or synonyms of the aspect's name.

    3.  **Self-Critique and Refine:** Before finalizing your output, critically review the rubric:
        * **Merge Overlaps:** Do any of your aspects significantly overlap or describe a sub-task of another aspect? If so, **merge them** into a single, more comprehensive aspect. The goal is to have distinct, non-redundant categories.
        * **Final Check:** Are the aspects and concepts logical and clear? Do they accurately capture all requirements from the original criteria?

    Aim for 2-4 well-defined aspects, but adjust based on the criteria complexity. The goal is a balanced and precise rubric for evaluation.
    """

    input_model = CriteriaDecomposerInput
    output_model = CriteriaDecomposerOutput
    description = 'Balanced Criteria Decomposer'
    examples = [
        (
            CriteriaDecomposerInput(
                query='What is the infield fly rule in baseball?',
                criteria='The answer must accurately describe the infield fly rule, including when it applies (fewer than two outs, runners on first and second or bases loaded), what happens when it is called (batter is automatically out), and why it exists (to prevent the defense from intentionally dropping a fly ball to turn a double play). The answer should also mention that runners may advance at their own risk.',
            ),
            CriteriaDecomposerOutput(
                key_aspects=[
                    'Rule Conditions',
                    'Rule Effect',
                    'Purpose of the Rule',
                    'Runner Advancement',
                ],
                aspect_details={
                    'Rule Conditions': [
                        'Fewer than two outs required',
                        'Runners on first and second base, or bases loaded',
                        'Fair fly ball that can be caught with ordinary effort',
                    ],
                    'Rule Effect': [
                        'Batter is automatically out when rule is invoked',
                        'Umpire must declare infield fly while ball is in the air',
                    ],
                    'Purpose of the Rule': [
                        'Prevents defense from intentionally dropping the ball',
                        'Protects baserunners from unfair double plays',
                    ],
                    'Runner Advancement': [
                        'Runners may advance at their own risk',
                        'Applies whether ball is caught or dropped',
                    ],
                },
            ),
        ),
        (
            CriteriaDecomposerInput(
                query='What is the link for the official MLB rulebook?',
                criteria='The answer must accurately identify the link for the official MLB rulebook. The link provided should lead directly to the official rules page on MLB.com.',
            ),
            CriteriaDecomposerOutput(
                key_aspects=[
                    'Official Rulebook Link',
                    'Resource Description',
                ],
                aspect_details={
                    'Official Rulebook Link': [
                        'Precise and complete link to official MLB rulebook is provided',
                        'Link is accessible and functional',
                    ],
                    'Resource Description': [
                        'Indication that the link is the authoritative source for MLB rules',
                        'Comprehensive information on rule coverage mentioned',
                    ],
                },
            ),
        ),
    ]


class AspectCoverageResult(RichBaseModel):
    aspect: str = Field(description='The expected aspect that was evaluated')
    covered: bool = Field(
        description='Whether the aspect was meaningfully addressed in the response'
    )
    concepts_covered: Optional[List[str]] = Field(
        default=[], description='The key concepts that were addressed'
    )
    concepts_missing: Optional[List[str]] = Field(
        default=[], description='The key concepts that were not addressed'
    )
    reason: str = Field(description='The reason for the coverage determination')

    @field_validator('concepts_covered', 'concepts_missing', mode='before')
    @classmethod
    def ensure_list(cls, v):
        """Convert None to empty list to handle LLM inconsistencies."""
        return v if v is not None else []


class CriteriaCheckerInput(RichBaseModel):
    query: str = Field(description='The original query')
    response: str = Field(description='The response to evaluate')
    key_aspects: List[str] = Field(
        description='List of key aspects extracted from the criteria'
    )
    aspect_details: Dict[str, List[str]] = Field(
        description='Key concepts for each aspect that should be mentioned'
    )
    criteria: str = Field(description='The original criteria text for reference')
    expected_output: Optional[str] = Field(
        default=None,
        description='Optional gold standard comprehensive answer to compare against',
    )


class CriteriaCheckerOutput(RichBaseModel):
    aspect_results: List[AspectCoverageResult] = Field(
        description='The evaluation results for each expected aspect'
    )


class CriteriaChecker(BaseMetric[CriteriaCheckerInput, CriteriaCheckerOutput]):
    # This instruction will be dynamically replaced by the parent class.
    as_structured_llm = False
    instruction = 'Default instruction, should be overridden.'
    input_model = CriteriaCheckerInput
    output_model = CriteriaCheckerOutput
    description = 'Balanced Criteria-based Response Checker'

    class CriteriaChecker(BaseMetric[CriteriaCheckerInput, CriteriaCheckerOutput]):
        instruction = 'Default instruction, should be overridden.'
        input_model = CriteriaCheckerInput
        output_model = CriteriaCheckerOutput
        description = 'Balanced Criteria-based Response Checker'
        examples = [
            # Example 1: Simple factual requirement - fully covered
            (
                CriteriaCheckerInput(
                    query='How to delete a sandbox?',
                    response='When you delete a sandbox, it is permanently erased and cannot be recovered.',
                    key_aspects=['Permanence of Deletion'],
                    aspect_details={
                        'Permanence of Deletion': [
                            'Deletes are permanent',
                            'Cannot be undone',
                        ]
                    },
                    criteria='The answer must accurately explain that sandbox deletes are permanent and cannot be undone.',
                ),
                CriteriaCheckerOutput(
                    aspect_results=[
                        AspectCoverageResult(
                            aspect='Permanence of Deletion',
                            covered=True,
                            concepts_covered=[
                                'Deletes are permanent',
                                'Cannot be undone',
                            ],
                            concepts_missing=[],
                            reason="The response correctly states that the deletion is permanent and it cannot be recovered, which semantically matches 'cannot be undone'. Both key concepts are addressed.",
                        ),
                    ]
                ),
            ),
            # Example 2: Partial coverage - one aspect covered, one missing
            (
                CriteriaCheckerInput(
                    query='What is the infield fly rule?',
                    response='The infield fly rule is a baseball rule that protects baserunners by declaring the batter out on certain easy pop-ups.',
                    key_aspects=['Rule Definition', 'Umpire Declaration'],
                    aspect_details={
                        'Rule Definition': [
                            'Protects baserunners from unfair double plays'
                        ],
                        'Umpire Declaration': [
                            'Umpire must declare infield fly while ball is in the air'
                        ],
                    },
                    criteria='Correctly identifies that the rule protects baserunners. Mentions that the umpire must declare infield fly.',
                ),
                CriteriaCheckerOutput(
                    aspect_results=[
                        AspectCoverageResult(
                            aspect='Rule Definition',
                            covered=True,
                            concepts_covered=[
                                'Protects baserunners from unfair double plays'
                            ],
                            concepts_missing=[],
                            reason='The response explicitly states that the rule "protects baserunners", directly satisfying this factual requirement.',
                        ),
                        AspectCoverageResult(
                            aspect='Umpire Declaration',
                            covered=False,
                            concepts_covered=[],
                            concepts_missing=[
                                'Umpire must declare infield fly while ball is in the air'
                            ],
                            reason='The response makes no mention of the umpire declaration requirement, leaving this factual requirement completely unaddressed.',
                        ),
                    ]
                ),
            ),
            # Example 3: Factual inaccuracy - contradicts criteria
            (
                CriteriaCheckerInput(
                    query='When does the infield fly rule apply?',
                    response='The infield fly rule applies with two outs and a runner on first base only.',
                    key_aspects=['Rule Conditions'],
                    aspect_details={
                        'Rule Conditions': [
                            'Fewer than two outs required',
                            'Runners on first and second or bases loaded',
                        ]
                    },
                    criteria='The answer must clearly state that the rule applies with fewer than two outs, and requires runners on first and second base or bases loaded.',
                ),
                CriteriaCheckerOutput(
                    aspect_results=[
                        AspectCoverageResult(
                            aspect='Rule Conditions',
                            covered=False,
                            concepts_covered=[],
                            concepts_missing=[
                                'Fewer than two outs required',
                                'Runners on first and second or bases loaded',
                            ],
                            reason='The response incorrectly states that the rule applies with "two outs and a runner on first base only", which directly contradicts the criteria requirement that the rule applies with fewer than two outs and runners on first and second or bases loaded. This factual inaccuracy means neither concept is covered.',
                        ),
                    ]
                ),
            ),
            # Example 4: Precise factual requirement not met
            (
                CriteriaCheckerInput(
                    query="What's the link for the official MLB rulebook?",
                    response='Here are some links for baseball rules: [Baseball Wikipedia](https://en.wikipedia.org/wiki/Baseball_rules). Let me know if you need further assistance!',
                    key_aspects=[
                        'Official MLB Rulebook Link',
                        'Resource Description',
                    ],
                    aspect_details={
                        'Official MLB Rulebook Link': [
                            'Precise and complete link to official MLB rulebook is provided',
                            'Link is accessible and functional',
                        ],
                        'Resource Description': [
                            'Indication that the link is the authoritative source for MLB rules',
                            'Comprehensive information on rule coverage mentioned',
                        ],
                    },
                    criteria='The answer must accurately identify the link for the official MLB rulebook. The link provided should lead directly to the official rules page on MLB.com, not Wikipedia or other third-party sites.',
                ),
                CriteriaCheckerOutput(
                    aspect_results=[
                        AspectCoverageResult(
                            aspect='Official MLB Rulebook Link',
                            covered=False,
                            concepts_covered=[],
                            concepts_missing=[
                                'Precise and complete link to official MLB rulebook is provided',
                                'Link is accessible and functional',
                            ],
                            reason='The response provides a Wikipedia link but does NOT provide the precise link to the official MLB rulebook page as required. The criteria specifically asks for the official MLB.com rules page, not third-party sites. This fails to meet the factual precision requirement.',
                        ),
                        AspectCoverageResult(
                            aspect='Resource Description',
                            covered=False,
                            concepts_covered=[],
                            concepts_missing=[
                                'Indication that the link is the authoritative source for MLB rules',
                                'Comprehensive information on rule coverage mentioned',
                            ],
                            reason='Since the response does not provide the official MLB link, it also fails to indicate that link as the authoritative source. Additionally, no information about rule coverage is mentioned in the response.',
                        ),
                    ]
                ),
            ),
            # Example 5: Multiple concepts, partial coverage acceptable
            (
                CriteriaCheckerInput(
                    query='What are the key conditions for the infield fly rule?',
                    response='The infield fly rule applies when there are fewer than two outs, with runners on first and second base or bases loaded, and a fair fly ball is hit that can be caught with ordinary effort.',
                    key_aspects=['Rule Conditions'],
                    aspect_details={
                        'Rule Conditions': [
                            'Fewer than two outs',
                            'Runners on first and second or bases loaded',
                            'Fair fly ball',
                            'Ordinary effort required',
                            'Infielder positioning',
                        ]
                    },
                    criteria='The answer should describe the key conditions for the infield fly rule, including out count, runner positions, ball type, and effort level.',
                ),
                CriteriaCheckerOutput(
                    aspect_results=[
                        AspectCoverageResult(
                            aspect='Rule Conditions',
                            covered=True,
                            concepts_covered=[
                                'Fewer than two outs',
                                'Runners on first and second or bases loaded',
                                'Fair fly ball',
                                'Ordinary effort required',
                            ],
                            concepts_missing=[
                                'Infielder positioning',
                            ],
                            reason='The response addresses 4 out of 5 key concepts. It explicitly mentions fewer than two outs, runners on first and second or bases loaded, fair fly ball, and ordinary effort. While it does not specifically mention "infielder positioning", the overall coverage is substantial and addresses the core requirement of describing the key conditions.',
                        ),
                    ]
                ),
            ),
        ]


class AnswerCriteriaAspectResult(RichBaseModel):
    """Evaluation for a single aspect of the response."""

    aspect: str = Field(description='The aspect of the criteria being evaluated.')
    covered: bool = Field(
        description='Indicates whether the aspect was sufficiently covered in the response.'
    )
    concepts_covered: Optional[List[str]] = Field(
        default=[],
        description='List of key concepts within this aspect that were successfully addressed in the response.',
    )
    concepts_missing: Optional[List[str]] = Field(
        default=[],
        description='List of key concepts within this aspect that were not addressed in the response.',
    )
    reason: str = Field(
        description='Rationale for why the aspect was considered covered or not, citing evidence from the response.'
    )
    concepts_coverage_percentage: float = Field(
        description='The percentage of concepts for this aspect that were covered.'
    )
    turn_index: Optional[int] = Field(
        default=None,
        description='The turn number in a multi-turn evaluation (starts from 0).',
    )


class AnswerCriteriaResult(RichBaseModel):
    """Aggregated evaluation across all aspects."""

    scoring_strategy: str = Field(
        description='The scoring approach used to evaluate the criteria.'
    )
    covered_aspects_count: int = Field(
        description='The number of aspects that were sufficiently covered.'
    )
    total_aspects_count: int = Field(
        description='The total number of aspects expected for this evaluation.'
    )
    total_concepts_covered: int = Field(
        description='The total number of concepts that were successfully addressed across all aspects.'
    )
    total_concepts: int = Field(
        description='The total number of expected concepts across all aspects.'
    )
    concept_coverage_score: float = Field(
        description='Overall coverage score based on the proportion of covered concepts.'
    )
    aspect_breakdown: List[AnswerCriteriaAspectResult] = Field(
        description='Detailed breakdown of coverage for each aspect, including covered/missing concepts.'
    )
    evaluated_turns_count: Optional[int] = Field(
        default=None,
        description='The number of turns evaluated in a multi-turn conversation.',
    )


@metric(
    name='Answer Criteria',
    key='answer_criteria',
    description='Evaluates responses based on user-defined criteria, with support for single-turn and multi-turn conversations.',
    required_fields=['query', 'actual_output'],
    optional_fields=['acceptance_criteria', 'additional_input', 'conversation'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['knowledge', 'agent', 'single_turn', 'multi_turn'],
)
class AnswerCriteria(BaseMetric):
    """
    Evaluates responses based on specified criteria.
    This metric extracts key aspects from the criteria and checks whether each aspect
    is adequately and accurately addressed in the response.

    It supports two modes of operation:
    1.  **Single-Turn / Last-Turn (default):** Evaluates `item.query` vs. `item.actual_output`.
        If `item.conversation` is present, `item.query` and `item.actual_output` are
        auto-populated from the last turn (based on `conversation_extraction_strategy`).
    2.  **Multi-Turn:** If `multi_turn_strategy='all_turns'`, this metric will iterate
        through the entire `item.conversation` and evaluate every `HumanMessage` -> `AIMessage`
        pair. The aggregation method is controlled by `multi_turn_aggregation`.
    """

    as_structured_llm = False

    _base_checker_instruction = """Evaluate whether each aspect extracted from the criteria has been meaningfully addressed in the provided response.

    For each expected aspect:
    1. **Determine Coverage**: An aspect is 'covered' if the response addresses the core requirement with meaningful and accurate information.
       - For factual aspects (links, specific values, precise definitions): The aspect is only 'covered' if the information is factually correct and matches the criteria requirements.
       - For descriptive/explanatory aspects: The aspect is 'covered' if the main requirement is addressed with relevant detail, even if not exhaustive.

    2. **Evaluate Concepts**: Identify which key concepts are mentioned and which are missing.
       - Recognize that concepts can be covered using different wording; focus on whether the semantic meaning is present.
       - CRITICAL: If a concept requires factual accuracy (e.g., "precise link", "correct value", "specific definition"), it is only 'covered' if the factual information is correct. Incorrect, incomplete, or tangentially related information means the concept is 'missing'.

    3. **Provide Reasoning**: Give a detailed explanation for your verdict, specifically noting:
       - Whether factual requirements were met accurately
       - Which concepts were addressed and how
       - Any discrepancies between what was provided vs. what was required

    **Evaluation Guidelines**:
    - High Specificity Required: For aspects involving links, documentation references, specific values, or precise definitions, the response must provide exactly what is requested. Related but different information does NOT count as coverage.
    - Moderate Flexibility: For aspects with multiple related concepts (like a list of features), the aspect is 'covered' if a reasonable portion of the concepts are mentioned.
    - Semantic Matching: For general explanatory aspects, allow for paraphrasing and different wording as long as the core meaning is preserved.

    Use the complete_criteria text as the authoritative reference for what should be included and the required level of specificity."""

    _contradiction_checker_instruction = """Evaluate whether each aspect extracted from the criteria has been meaningfully AND ACCURATELY addressed in the provided response.

    For each expected aspect:
    1. Determine if the aspect is covered (true if the main requirement is addressed with relevant and correct detail, false if completely missing OR contradicted).
    2. Identify which key concepts are mentioned and which are missing. Recognize that concepts can be covered using different wording; focus on whether the semantic meaning is present.
    3. CRITICAL FOR ACCURACY: If a concept is mentioned but is factually incorrect, negated, or misrepresented when compared to the criteria, it MUST be marked as 'missing'. If a key concept is contradicted, the entire aspect MUST be marked as 'covered: false'.
    4. Provide a detailed explanation for your verdict, specifically mentioning any contradictions found.

    Be balanced in your evaluation, but prioritize accuracy. An aspect is only 'covered' if its core requirements are met without being contradicted."""

    def __init__(
        self,
        criteria_key: str = 'Complete',
        scoring_strategy: Literal['concept', 'aspect', 'weighted'] = 'concept',
        check_for_contradictions: bool = False,
        weighted_concept_score_weight: float = 0.7,
        multi_turn_strategy: Literal['last_turn', 'all_turns'] = 'last_turn',
        multi_turn_aggregation: Literal['cumulative', 'average'] = 'cumulative',
        **kwargs,
    ):
        """
        Initialize the criteria-based answer criteria metric.

        Args:
            criteria_key: The key in `additional_input` or `conversation.rubrics`
                          to find the criteria text (default: 'Complete').
            scoring_strategy: The scoring method: 'concept', 'aspect', or 'weighted' (default: 'concept').
            check_for_contradictions: If True, uses a stricter prompt to penalize contradictions (default: False).
            weighted_concept_score_weight: The weight for the concept score in 'weighted' strategy (default: 0.7).
            multi_turn_strategy: How to handle multi-turn conversations.
                                 'last_turn' (default): Evaluates only the last turn.
                                 'all_turns': Evaluates all Human->AI turns in the conversation.
            multi_turn_aggregation: Aggregation method for 'all_turns' strategy.
                                    'cumulative' (default): Scores unique aspects covered across all turns.
                                    'average': Scores average aspect coverage per turn.
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        if scoring_strategy not in ['concept', 'aspect', 'weighted']:
            raise ValueError(
                "scoring_strategy must be 'concept', 'aspect', or 'weighted'"
            )
        if not 0.0 <= weighted_concept_score_weight <= 1.0:
            raise ValueError(
                'weighted_concept_score_weight must be between 0.0 and 1.0'
            )

        self.criteria_key = criteria_key
        self.scoring_strategy = scoring_strategy
        self.check_for_contradictions = check_for_contradictions
        self.weighted_concept_score_weight = weighted_concept_score_weight
        self.multi_turn_strategy = multi_turn_strategy
        self.multi_turn_aggregation = multi_turn_aggregation

        self.criteria_decomposer = CriteriaDecomposer(**kwargs)

        # Dynamically create and configure the checker with the appropriate prompt
        self.criteria_checker = CriteriaChecker(**kwargs)
        if self.check_for_contradictions:
            self.criteria_checker.set_instruction(
                self._contradiction_checker_instruction
            )
        else:
            self.criteria_checker.set_instruction(self._base_checker_instruction)

    @trace(
        name='extract_single_turn_criteria', capture_args=True, capture_response=True
    )
    def _extract_single_turn_criteria(self, item: DatasetItem) -> Optional[str]:
        """
        Extract the criteria string for a single-turn eval, prioritizing 'acceptance_criteria'
        then 'additional_input'. Returns None if not found.
        """
        # Prioritize the top-level 'acceptance_criteria' field
        if item.acceptance_criteria:
            if isinstance(item.acceptance_criteria, list):
                return '\n'.join(item.acceptance_criteria)
            if isinstance(item.acceptance_criteria, str):
                return item.acceptance_criteria

        # Fallback to checking the 'additional_input' dictionary using the consolidated key
        if item.additional_input and self.criteria_key in item.additional_input:
            criteria = item.additional_input.get(self.criteria_key)
            if criteria and isinstance(criteria, str):
                return criteria

        return None

    @trace(name='extract_multi_turn_criteria', capture_args=True, capture_response=True)
    def _extract_multi_turn_criteria(self, item: DatasetItem) -> Optional[str]:
        """
        Extract criteria for multi-turn eval.
        1. Checks `item.conversation.rubrics[self.criteria_key]`
        2. Falls back to single-turn criteria logic.
        """
        if (
            item.conversation
            and item.conversation.rubrics
            and self.criteria_key in item.conversation.rubrics
        ):
            criteria = item.conversation.rubrics[self.criteria_key]
            if criteria and isinstance(criteria, str):
                return criteria

        # Fallback to the single-turn logic if no multi-turn rubric is found
        return self._extract_single_turn_criteria(item)

    @trace(name='decompose_criteria', capture_args=True, capture_response=True)
    async def _decompose_criteria(
        self, query: str, criteria: str
    ) -> CriteriaDecomposerOutput:
        """Decompose the criteria into key aspects and their details."""
        result = await self.criteria_decomposer.execute(
            CriteriaDecomposerInput(query=query, criteria=criteria)
        )
        if not result.key_aspects:
            logger.warning('No key aspects were extracted from the criteria.')
            return CriteriaDecomposerOutput(key_aspects=[], aspect_details={})
        for aspect in result.key_aspects:
            if aspect not in result.aspect_details:
                logger.warning(
                    f'No details found for aspect: {aspect}, adding placeholder.'
                )
                result.aspect_details[aspect] = [f'{aspect} overview or explanation']
        return result

    @trace(name='check_criteria', capture_args=True, capture_response=True)
    async def _check_criteria(
        self,
        query: str,
        actual_output: str,
        expected_output: Optional[str],
        key_aspects: List[str],
        aspect_details: Dict[str, List[str]],
        criteria: str,
    ) -> CriteriaCheckerOutput:
        """Check criteria-based coverage in the response."""
        return await self.criteria_checker.execute(
            CriteriaCheckerInput(
                query=query,
                response=actual_output,
                key_aspects=key_aspects,
                aspect_details=aspect_details,
                criteria=criteria,
                expected_output=expected_output,
            )
        )

    @trace(name='compute_criteria_score', capture_args=True, capture_response=True)
    def _compute_criteria_score(
        self,
        evaluation_results: CriteriaCheckerOutput,
        turn_index: Optional[int] = None,
    ) -> Dict:
        """
        Compute the criteria-based score and detailed breakdown for a single turn.
        """
        aspect_results = evaluation_results.aspect_results or []
        if not aspect_results:
            logger.warning(f'No aspects were evaluated for turn {turn_index}.')
            return {
                'score': np.nan,
                'covered_aspects_count': 0,
                'total_aspects_count': 0,
                'total_concepts_covered': 0,
                'total_concepts': 0,
                'aspect_coverage_percentage': np.nan,
                'concept_coverage_score': np.nan,
                'aspect_coverage_score': np.nan,
                'aspect_breakdown': [],
            }

        total_concepts_covered = sum(
            len(r.concepts_covered or []) for r in aspect_results
        )
        total_concepts = sum(
            len(r.concepts_covered or []) + len(r.concepts_missing or [])
            for r in aspect_results
        )
        concept_coverage_score = (
            total_concepts_covered / total_concepts if total_concepts else 0.0
        )

        covered_aspects_count = sum(1 for r in aspect_results if r.covered)
        total_aspects_count = len(aspect_results or [])
        aspect_coverage_score = (
            covered_aspects_count / total_aspects_count if total_aspects_count else 0.0
        )
        aspect_coverage_percentage = aspect_coverage_score * 100

        aspect_breakdown = []
        for result in aspect_results:
            concepts_total = len(result.concepts_covered or []) + len(
                result.concepts_missing or []
            )
            concepts_coverage_percentage = (
                (len(result.concepts_covered or []) / concepts_total * 100)
                if concepts_total
                else 0.0
            )
            aspect_breakdown.append(
                {
                    'aspect': result.aspect,
                    'covered': result.covered,
                    'concepts_covered': result.concepts_covered or [],
                    'concepts_missing': result.concepts_missing or [],
                    'concepts_coverage_percentage': concepts_coverage_percentage,
                    'reason': result.reason,
                    'turn_index': turn_index,
                }
            )

        if self.scoring_strategy == 'concept':
            primary_score = concept_coverage_score
        elif self.scoring_strategy == 'aspect':
            primary_score = aspect_coverage_score
        else:  # weighted
            aspect_weight = 1.0 - self.weighted_concept_score_weight
            primary_score = max(
                (self.weighted_concept_score_weight * concept_coverage_score)
                + (aspect_weight * aspect_coverage_score),
                1.0,
            )

        return {
            'score': primary_score,
            'covered_aspects_count': covered_aspects_count,
            'total_aspects_count': total_aspects_count,
            'total_concepts_covered': total_concepts_covered,
            'total_concepts': total_concepts,
            'aspect_coverage_percentage': aspect_coverage_percentage,
            'concept_coverage_score': concept_coverage_score,
            'aspect_coverage_score': aspect_coverage_score,
            'aspect_breakdown': aspect_breakdown,
        }

    async def _evaluate_single_turn(self, item: DatasetItem) -> MetricEvaluationResult:
        """
        Run criteria evaluation for a single turn (or the last turn of a conversation).
        """
        criteria = self._extract_single_turn_criteria(item)
        if not criteria:
            return MetricEvaluationResult(
                score=np.nan,
                explanation=f'No criteria found in `acceptance_criteria` '
                f'or `additional_input["{self.criteria_key}"]`.',
            )

        if not item.query:
            return MetricEvaluationResult(
                score=np.nan, explanation='DatasetItem has no `query` to evaluate.'
            )
        if not item.actual_output:
            return MetricEvaluationResult(
                score=np.nan,
                explanation='DatasetItem has no `actual_output` to evaluate.',
            )

        # Decompose criteria into key aspects
        decomposition_result = await self._decompose_criteria(item.query, criteria)

        if not decomposition_result.key_aspects:
            return MetricEvaluationResult(
                score=np.nan,
                explanation='Could not extract key aspects from the provided criteria.',
            )

        # Check the response against the decomposed criteria
        evaluation_results = await self._check_criteria(
            item.query,
            item.actual_output,
            item.expected_output,
            decomposition_result.key_aspects,
            decomposition_result.aspect_details,
            criteria,
        )

        # Delegate all scoring and data calculation to the centralized helper method
        score_data = self._compute_criteria_score(evaluation_results, turn_index=0)

        # Build the final Pydantic models from the calculated results
        aspect_breakdown = [
            AnswerCriteriaAspectResult(**aspect_data)
            for aspect_data in score_data['aspect_breakdown']
        ]

        result_data = AnswerCriteriaResult(
            scoring_strategy=self.scoring_strategy,
            covered_aspects_count=score_data['covered_aspects_count'],
            total_aspects_count=score_data['total_aspects_count'],
            total_concepts_covered=score_data['total_concepts_covered'],
            total_concepts=score_data['total_concepts'],
            concept_coverage_score=score_data['concept_coverage_score'],
            aspect_breakdown=aspect_breakdown,
            evaluated_turns_count=1,
        )
        return MetricEvaluationResult(score=score_data['score'], signals=result_data)

    async def _evaluate_multi_turn(self, item: DatasetItem) -> MetricEvaluationResult:
        """
        Run criteria evaluation for all turns in a conversation and aggregate results.
        """
        turns_to_eval = []
        if not item.conversation or not item.conversation.messages:
            return MetricEvaluationResult(
                score=np.nan, explanation='No conversation messages found to evaluate.'
            )

        criteria_for_all_turns = self._extract_multi_turn_criteria(item)
        if not criteria_for_all_turns:
            return MetricEvaluationResult(
                score=np.nan,
                explanation=f'No criteria found for multi-turn evaluation (checked rubrics["{self.criteria_key}"] and single-turn fallbacks).',
            )

        current_query = None
        for i, message in enumerate(item.conversation.messages):
            if isinstance(message, HumanMessage):
                current_query = message.content
            elif isinstance(message, AIMessage) and message.content:
                if current_query is None:
                    logger.warning(
                        f'Found AIMessage at index {i} with no preceding HumanMessage. Skipping turn.'
                    )
                    continue

                turns_to_eval.append(
                    {
                        'query': current_query,
                        'actual_output': message.content,
                        'criteria': criteria_for_all_turns,
                        'turn_index': len(turns_to_eval),
                    }
                )
                # Reset query to None to ensure we only link the *next* AI message
                current_query = None

        if not turns_to_eval:
            return MetricEvaluationResult(
                score=np.nan,
                explanation='No evaluable HumanMessage -> AIMessage turns found in conversation.',
            )

        # Initialize aggregation variables before the conditional block
        total_covered_aspects: int = 0
        total_aspects: int = 0
        total_concepts_covered: int = 0
        total_concepts: int = 0
        final_aspect_breakdown: List[AnswerCriteriaAspectResult] = []

        if self.multi_turn_aggregation == 'average':
            # Decomposes criteria for each turn and averages the results.
            all_turn_score_data = []
            for turn in turns_to_eval:
                decomp_result = await self._decompose_criteria(
                    turn['query'], turn['criteria']
                )
                if not decomp_result.key_aspects:
                    logger.warning(
                        f'Could not decompose criteria for turn {turn["turn_index"]}. Skipping turn.'
                    )
                    all_turn_score_data.append(
                        self._compute_criteria_score(
                            CriteriaCheckerOutput(aspect_results=[]), turn['turn_index']
                        )
                    )
                    continue

                eval_result = await self._check_criteria(
                    query=turn['query'],
                    actual_output=turn['actual_output'],
                    expected_output=None,  # No expected_output for intermediate turns
                    key_aspects=decomp_result.key_aspects,
                    aspect_details=decomp_result.aspect_details,
                    criteria=turn['criteria'],
                )
                score_data = self._compute_criteria_score(
                    eval_result, turn['turn_index']
                )
                all_turn_score_data.append(score_data)

            # Aggregate 'average' results
            final_aspect_breakdown = [
                AnswerCriteriaAspectResult(**aspect_data)
                for turn_data in all_turn_score_data
                for aspect_data in turn_data['aspect_breakdown']
            ]
            total_covered_aspects = sum(
                r['covered_aspects_count'] for r in all_turn_score_data
            )
            total_aspects = sum(r['total_aspects_count'] for r in all_turn_score_data)
            total_concepts_covered = sum(
                r['total_concepts_covered'] for r in all_turn_score_data
            )
            total_concepts = sum(r['total_concepts'] for r in all_turn_score_data)

        else:
            first_query = turns_to_eval[0]['query']
            decomp_result = await self._decompose_criteria(
                first_query, criteria_for_all_turns
            )

            if not decomp_result.key_aspects:
                return MetricEvaluationResult(
                    score=np.nan,
                    explanation=f'Could not decompose criteria for multi-turn cumulative evaluation. Query: "{first_query}"',
                )

            key_aspects = decomp_result.key_aspects
            aspect_details = decomp_result.aspect_details

            cumulative_covered_aspects: Set[str] = set()
            cumulative_covered_concepts: Set[str] = set()

            # Track the most recent state of each aspect across turns
            aspect_states: Dict[str, AnswerCriteriaAspectResult] = {}

            for turn in turns_to_eval:
                eval_result = await self._check_criteria(
                    query=turn['query'],
                    actual_output=turn['actual_output'],
                    expected_output=None,
                    key_aspects=key_aspects,
                    aspect_details=aspect_details,
                    criteria=turn['criteria'],
                )

                # Process results for this turn
                for aspect_res in eval_result.aspect_results:
                    if aspect_res.covered:
                        cumulative_covered_aspects.add(aspect_res.aspect)

                    for concept in aspect_res.concepts_covered:
                        cumulative_covered_concepts.add(concept)

                    # Store the per-turn breakdown
                    concepts_total = len(aspect_res.concepts_covered or []) + len(
                        aspect_res.concepts_missing or []
                    )
                    concepts_coverage_percentage = (
                        (len(aspect_res.concepts_covered or []) / concepts_total * 100)
                        if concepts_total
                        else 0.0
                    )

                    aspect_result = AnswerCriteriaAspectResult(
                        aspect=aspect_res.aspect,
                        covered=aspect_res.covered,
                        concepts_covered=aspect_res.concepts_covered or [],
                        concepts_missing=aspect_res.concepts_missing or [],
                        reason=aspect_res.reason,
                        concepts_coverage_percentage=concepts_coverage_percentage,
                        turn_index=turn['turn_index'],
                    )

                    # Add to breakdown (keeps all turn-by-turn details)
                    final_aspect_breakdown.append(aspect_result)

                    # Update the latest state for this aspect
                    # This ensures cumulative concept tracking across turns
                    if aspect_res.aspect not in aspect_states:
                        aspect_states[aspect_res.aspect] = aspect_result
                    else:
                        # Merge concepts from this turn with previous turns
                        prev_state = aspect_states[aspect_res.aspect]
                        merged_covered = set(prev_state.concepts_covered) | set(
                            aspect_res.concepts_covered
                        )
                        merged_missing = (
                            set(prev_state.concepts_missing) - merged_covered
                        )

                        aspect_states[aspect_res.aspect] = AnswerCriteriaAspectResult(
                            aspect=aspect_res.aspect,
                            covered=aspect_res.covered
                            or prev_state.covered,  # Covered if ANY turn covered it
                            concepts_covered=list(merged_covered),
                            concepts_missing=list(merged_missing),
                            reason=aspect_res.reason,  # Use latest reason
                            concepts_coverage_percentage=(
                                len(merged_covered)
                                / (len(merged_covered) + len(merged_missing))
                                * 100
                            )
                            if (len(merged_covered) + len(merged_missing)) > 0
                            else 0.0,
                            turn_index=turn['turn_index'],
                        )

            # Aggregate 'cumulative' results using UNIQUE concepts only
            total_covered_aspects = len(cumulative_covered_aspects)
            total_aspects = len(key_aspects)
            total_concepts = sum(len(v) for v in aspect_details.values())
            total_concepts_covered = len(
                cumulative_covered_concepts
            )  # This is already unique

        concept_coverage_score = (
            total_concepts_covered / total_concepts if total_concepts else 0.0
        )
        aspect_coverage_score = (
            total_covered_aspects / total_aspects if total_aspects else 0.0
        )

        if self.scoring_strategy == 'concept':
            final_score = concept_coverage_score
        elif self.scoring_strategy == 'aspect':
            final_score = aspect_coverage_score
        else:  # weighted
            aspect_weight = 1.0 - self.weighted_concept_score_weight
            final_score = (
                self.weighted_concept_score_weight * concept_coverage_score
            ) + (aspect_weight * aspect_coverage_score)

        final_score = final_score if not np.isnan(final_score) else np.nan

        result_data = AnswerCriteriaResult(
            scoring_strategy=self.scoring_strategy,
            covered_aspects_count=total_covered_aspects,
            total_aspects_count=total_aspects,
            total_concepts_covered=total_concepts_covered,
            total_concepts=total_concepts,
            concept_coverage_score=concept_coverage_score,
            aspect_breakdown=final_aspect_breakdown,
            evaluated_turns_count=len(turns_to_eval),
        )

        return MetricEvaluationResult(score=final_score, signals=result_data)

    @trace(name='AnswerCriteria', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """
        Compute the score based on criteria.
        Automatically handles single-turn or multi-turn evaluation based on
        `self.multi_turn_strategy` and `item.conversation`.
        """
        self._validate_required_metric_fields(item)

        if item.conversation and self.multi_turn_strategy == 'all_turns':
            logger.debug('Using multi-turn "all_turns" evaluation approach')
            result = await self._evaluate_multi_turn(item)
        else:
            logger.debug('Using single-turn / "last_turn" evaluation approach')
            result = await self._evaluate_single_turn(item)

        # Compute cost estimates from all sub-models used
        self.compute_cost_estimate(
            [
                self.criteria_decomposer,
                self.criteria_checker,
            ]
        )
        return result

    def get_signals(
        self,
        result: AnswerCriteriaResult,
    ) -> List[SignalDescriptor[AnswerCriteriaResult]]:
        """Generates a list of detailed signals from the evaluation result that explain the scoring."""
        signals = []

        def get_concept_score(r: AnswerCriteriaResult) -> float:
            return r.concept_coverage_score

        def get_aspect_score(r: AnswerCriteriaResult) -> float:
            return (
                r.covered_aspects_count / r.total_aspects_count
                if r.total_aspects_count
                else 0.0
            )

        def get_weighted_score(r: AnswerCriteriaResult) -> float:
            aspect_score = get_aspect_score(r)
            aspect_weight = 1.0 - self.weighted_concept_score_weight
            return (r.concept_coverage_score * self.weighted_concept_score_weight) + (
                aspect_score * aspect_weight
            )

        # --- Overall Score Signal ---
        if result.scoring_strategy == 'concept':
            headline_desc = (
                f'Final score based on concept coverage: {result.total_concepts_covered} of '
                f'{result.total_concepts} concepts were covered.'
            )
            headline_extractor = get_concept_score
            headline_name = 'concept_coverage_score'

        elif result.scoring_strategy == 'aspect':
            headline_desc = (
                f'Final score based on aspect coverage: {result.covered_aspects_count} '
                f'of {result.total_aspects_count} aspects were covered.'
            )
            headline_extractor = get_aspect_score
            headline_name = 'aspect_coverage_score'

        else:  # weighted
            headline_desc = (
                f'Final weighted score. ('
                f'Aspects: {result.covered_aspects_count}/{result.total_aspects_count}, '
                f'Concepts: {result.total_concepts_covered}/{result.total_concepts})'
            )
            headline_extractor = get_weighted_score
            headline_name = 'weighted_score'

        signals.append(
            SignalDescriptor(
                name=headline_name,
                group='overall',
                description=headline_desc,
                extractor=headline_extractor,
                headline_display=True,
            )
        )

        if result.evaluated_turns_count:
            signals.append(
                SignalDescriptor(
                    name='evaluated_turns',
                    group='overall',
                    description='Total number of Human->AI turns evaluated.',
                    extractor=lambda r: r.evaluated_turns_count,
                )
            )

        # --- Per-Aspect Breakdown Signals ---
        for i, aspect in enumerate(result.aspect_breakdown):
            sanitized_aspect_name = re.sub(r'[^\w\s-]', '', aspect.aspect).strip()
            sanitized_aspect_name = re.sub(r'[-\s]+', '_', sanitized_aspect_name)

            # Add turn_index to group name if present
            if aspect.turn_index is not None:
                group_name = f'turn_{aspect.turn_index}_aspect_{sanitized_aspect_name}'
            else:
                group_name = f'aspect_{sanitized_aspect_name}'

            # Determine which per-aspect signal should be the headline
            is_concept_headline = result.scoring_strategy == 'concept'
            is_aspect_headline = result.scoring_strategy == 'aspect'

            total_concepts_in_aspect = len(aspect.concepts_covered or []) + len(
                aspect.concepts_missing or []
            )
            concept_coverage_description = f'{len(aspect.concepts_covered or [])} of {total_concepts_in_aspect} concepts covered for this aspect.'

            signals.extend(
                [
                    SignalDescriptor(
                        name='is_covered',
                        group=group_name,
                        description='Whether the aspect was considered covered.',
                        extractor=lambda r, idx=i: r.aspect_breakdown[idx].covered,
                        headline_display=is_aspect_headline,
                    ),
                    SignalDescriptor(
                        name='concept_coverage',
                        group=group_name,
                        description=concept_coverage_description,
                        extractor=lambda r, idx=i: r.aspect_breakdown[
                            idx
                        ].concepts_coverage_percentage
                        / 100.0,
                        headline_display=is_concept_headline,
                    ),
                    SignalDescriptor(
                        name='aspect',
                        group=group_name,
                        description='The name of the evaluated aspect.',
                        extractor=lambda r, idx=i: r.aspect_breakdown[idx].aspect,
                    ),
                    SignalDescriptor(
                        name='turn_index',
                        group=group_name,
                        description='The conversation turn index, if applicable.',
                        extractor=lambda r, idx=i: r.aspect_breakdown[idx].turn_index,
                    ),
                    SignalDescriptor(
                        name='concepts_covered',
                        group=group_name,
                        description='List of concepts covered for this aspect.',
                        extractor=lambda r, idx=i: r.aspect_breakdown[
                            idx
                        ].concepts_covered,
                    ),
                    SignalDescriptor(
                        name='concepts_missing',
                        group=group_name,
                        description='List of concepts missing for this aspect.',
                        extractor=lambda r, idx=i: r.aspect_breakdown[
                            idx
                        ].concepts_missing,
                    ),
                    SignalDescriptor(
                        name='reason',
                        group=group_name,
                        description='The reasoning provided by the LLM for the evaluation of this aspect.',
                        extractor=lambda r, idx=i: r.aspect_breakdown[idx].reason,
                    ),
                ]
            )

        return signals
