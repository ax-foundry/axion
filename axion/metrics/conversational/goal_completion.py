from typing import List, Literal, Optional

import numpy as np
from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import HumanMessage, RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.cache import AnalysisCache
from axion.metrics.internals.conversation_utils import (
    ConversationMoment,
    MomentSegmenter,
    TurnAnalysis,
    TurnAnalyzer,
    UserFrictionAnalyzer,
    get_or_compute_moments,
    get_or_compute_turn_analysis,
)
from axion.metrics.schema import SignalDescriptor

logger = get_logger(__name__)


class SubGoalProgress(RichBaseModel):
    """Sub-goal with temporal tracking and effort metrics."""

    description: str = Field(
        description='A specific sub-goal or step required to achieve the main objective'
    )
    status: Literal['achieved', 'partially_achieved', 'not_achieved', 'abandoned'] = (
        Field(description='The completion status of this sub-goal')
    )

    # Temporal tracking
    first_mentioned_turn: Optional[int] = Field(
        default=None,
        description='When this sub-goal first appeared in the conversation',
    )
    achieved_at_turn: Optional[int] = Field(
        default=None, description='When this sub-goal was completed'
    )
    associated_moment_id: Optional[int] = Field(
        default=None, description='Which conversation moment addressed this sub-goal'
    )

    # Effort metrics
    turns_to_complete: Optional[int] = Field(
        default=None, description='How many turns it took to complete this sub-goal'
    )
    required_clarifications: int = Field(
        default=0, description='How many times user had to clarify this sub-goal'
    )
    agent_retries: int = Field(
        default=0, description='How many times agent had to retry this sub-goal'
    )


class GoalMomentMapping(RichBaseModel):
    """Maps sub-goals to conversation moments."""

    moment_id: int = Field(description='ID of the conversation moment')
    moment_topic: str = Field(description='Topic/theme of the moment')
    addressed_subgoals: List[str] = Field(
        description='Which sub-goal descriptions were addressed in this moment'
    )
    moment_efficiency: float = Field(
        description='How efficiently this moment addressed its sub-goals (0-1)'
    )


class GoalCompletionResult(RichBaseModel):
    """Rich result combining goal tracking with conversation analysis."""

    # Original goal tracking
    primary_goal: str = Field(description="The user's main objective")
    goal_evolution: str = Field(description='How the goal changed during conversation')
    final_outcome: Literal['achieved', 'partially_achieved', 'failed', 'abandoned']
    sub_goals: List[SubGoalProgress]

    # Multi-layered analysis (reused from ConversationFlow)
    goal_moment_mappings: List[GoalMomentMapping] = Field(
        description='How conversation moments map to sub-goals'
    )
    conversation_moments: List[ConversationMoment] = Field(
        description='Topical segments from shared analysis'
    )
    turn_analysis: List[TurnAnalysis] = Field(
        description='Turn-by-turn analysis from shared analysis'
    )

    # Composite scores
    goal_completion_score: float = Field(
        description='Pure goal achievement score (0-1)'
    )
    efficiency_score: float = Field(
        description='How efficiently goals were achieved (0-1)'
    )
    final_composite_score: float = Field(
        description='Weighted combination of completion and efficiency'
    )

    # Diagnostic info
    bottleneck_subgoals: List[str] = Field(
        default_factory=list, description='Sub-goals that took excessive effort'
    )
    abandoned_early: bool = Field(
        default=False, description='Whether user gave up before completing'
    )
    goal_drift_detected: bool = Field(
        default=False, description='Whether conversation drifted from original goal'
    )


class UnifiedGoalAnalysisInput(RichBaseModel):
    """Input for complete goal analysis in one pass."""

    conversation_transcript: str = Field(
        description='Full conversation transcript between user and agent'
    )
    user_persona_and_goal: str = Field(
        description="User's persona and stated goal/objective"
    )
    conversation_moments: List[ConversationMoment] = Field(
        description='Pre-computed topical segments of the conversation'
    )
    turn_analysis: List[TurnAnalysis] = Field(
        description='Pre-computed turn-by-turn analysis with friction points'
    )


class UnifiedGoalAnalysisOutput(RichBaseModel):
    """Complete goal analysis output from single LLM call."""

    primary_goal: str = Field(description="The user's main, high-level objective")
    goal_evolution: str = Field(
        description='How the goal changed or was refined during conversation'
    )
    sub_goals: List[SubGoalProgress] = Field(
        description='Sub-goals with complete temporal tracking'
    )
    goal_moment_mappings: List[GoalMomentMapping] = Field(
        description='Mapping of conversation moments to sub-goals they addressed'
    )


class UnifiedGoalAnalyzer(
    BaseMetric[UnifiedGoalAnalysisInput, UnifiedGoalAnalysisOutput]
):
    """Single-pass goal analysis with temporal tracking and moment mapping."""

    as_structured_llm = False

    instruction = """You are an expert evaluator analyzing a conversation for goal achievement and efficiency.

**PART 1: GOAL IDENTIFICATION**
Based on the user's persona and goal, identify:
1. **primary_goal**: The user's main, high-level objective (1-2 sentences)
2. **goal_evolution**: How the user's goal changed or was refined during the conversation (2-3 sentences)
3. **sub_goals**: Break down the primary goal into 3-6 necessary steps or sub-objectives

**PART 2: TEMPORAL TRACKING (for each sub-goal)**
Using the turn_analysis provided, determine for EACH sub-goal:
- **description**: What needs to be accomplished for this sub-goal
- **status**: "achieved", "partially_achieved", "not_achieved", or "abandoned"
- **first_mentioned_turn**: When the sub-goal was first discussed (turn index starting from 0, or null if never mentioned)
- **achieved_at_turn**: When it was completed (turn index, or null if not achieved)
- **turns_to_complete**: Duration from mention to completion (or null if not completed)
- **required_clarifications**: How many times the user had to re-explain or clarify this sub-goal (count based on friction points in turn_analysis)
- **agent_retries**: How many times the agent attempted this sub-goal before succeeding (count failed attempts)

**PART 3: MOMENT MAPPING**
Using the conversation_moments provided, map each moment to the sub-goals it addressed:
For each moment, create a GoalMomentMapping with:
- **moment_id**: The ID of the moment (use the moment's id field)
- **moment_topic**: The topic from the moment (use the moment's topic field)
- **addressed_subgoals**: List of sub-goal descriptions (exact text from Part 1) that were addressed in this moment
- **moment_efficiency**: How efficiently the moment addressed those sub-goals (0.0-1.0 scale)

Efficiency scoring criteria:
- 1.0: Moment directly and completely solved the sub-goal(s) with no issues
- 0.7-0.9: Moment addressed sub-goal(s) with minor detours or clarifications
- 0.4-0.6: Moment partially addressed sub-goal(s) but required follow-up
- 0.1-0.3: Moment touched on sub-goal(s) but made little progress
- 0.0: Moment did not address any sub-goals

**IMPORTANT GUIDELINES:**
- Be precise with turn indices - use exact turn numbers from the conversation (0-indexed)
- A sub-goal is "achieved" only if it was fully completed
- A sub-goal is "partially_achieved" if significant progress was made but not finished
- A sub-goal is "abandoned" if the user explicitly gave up on it or stopped pursuing it
- A sub-goal is "not_achieved" if it was never completed or attempted
- Count clarifications based on user having to repeat, rephrase, or re-explain their needs
- Count retries based on agent making errors and having to try again

**OUTPUT FORMAT:**
Return a single object with:
- primary_goal (string)
- goal_evolution (string)
- sub_goals (list of SubGoalProgress objects with all temporal fields filled)
- goal_moment_mappings (list of GoalMomentMapping objects)"""

    input_model = UnifiedGoalAnalysisInput
    output_model = UnifiedGoalAnalysisOutput


async def get_or_compute_unified_goal_analysis(
    item: DatasetItem,
    cache: Optional[AnalysisCache],
    unified_analyzer: UnifiedGoalAnalyzer,
    persona_goal: str,
    transcript: str,
    moments: List[ConversationMoment],
    turn_analysis: List[TurnAnalysis],
) -> UnifiedGoalAnalysisOutput:
    """Get unified goal analysis from cache or compute it."""

    if cache:
        # Create a stable cache key based on transcript and goal
        cache_key = f'unified_goal_analysis_{hash(transcript)}_{hash(persona_goal)}'
        cached = cache.get(item.id, cache_key)
        if cached:
            logger.info('Using cached unified goal analysis')
            return cached

    logger.info('Computing unified goal analysis')
    unified_input = UnifiedGoalAnalysisInput(
        conversation_transcript=transcript,
        user_persona_and_goal=persona_goal,
        conversation_moments=moments,
        turn_analysis=turn_analysis,
    )

    result = await unified_analyzer.execute(unified_input)

    if cache:
        cache.set(item.id, cache_key, result)

    return result


@metric(
    name='Goal Completion',
    key='goal_completion',
    description="Analyzes if the user's goal was achieved, tracking sub-goals and goal evolution.",
    required_fields=['conversation'],
    optional_fields=[],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['agent', 'multi_turn'],
)
class GoalCompletion(BaseMetric):
    """
    Composite metric that analyzes:
    1. Goal achievement (did we complete the task?)
    2. Conversation efficiency (how well did we complete it?)
    3. Temporal tracking (when/how did we achieve sub-goals?)

    Reuses conversation analysis from shared components to minimize LLM calls.
    Uses unified analysis approach to reduce LLM calls from ~3+N to ~3 total.
    """

    shares_internal_cache = True

    def __init__(
        self,
        goal_key: str = 'goal',
        completion_weight: float = 0.6,
        efficiency_weight: float = 0.4,
        bottleneck_threshold: int = 5,
        max_clarification_penalty: float = 0.3,
        clarification_penalty_rate: float = 0.1,
        goal_drift_threshold: float = 0.3,
        outcome_threshold_achieved: float = 0.8,
        outcome_threshold_partial: float = 0.4,
        **kwargs,
    ):
        """
        Initialize the Goal Completion metric.

        Args:
            goal_key: Key in additional_input containing the user's goal

            # Composite score weights
            completion_weight: Weight for pure goal achievement (default: 0.6)
                Rationale: Goal achievement is slightly more important than efficiency
            efficiency_weight: Weight for conversation efficiency (default: 0.4)
                Rationale: Efficiency matters, but achieving the goal is primary

            # Bottleneck detection
            bottleneck_threshold: Number of turns before a sub-goal is flagged as bottleneck (default: 5)
                Rationale: Most sub-goals should resolve within 3-5 turns in efficient conversations

            # Efficiency penalties
            max_clarification_penalty: Maximum penalty for clarifications (default: 0.3)
                Rationale: Excessive clarifications can reduce efficiency by up to 30%
            clarification_penalty_rate: Penalty per clarification (default: 0.1)
                Rationale: Each clarification represents a 10% efficiency loss

            # Goal drift detection
            goal_drift_threshold: Fraction of unmapped moments to trigger drift detection (default: 0.3)
                Rationale: If >30% of conversation is unrelated to goal, it indicates drift

            # Outcome thresholds
            outcome_threshold_achieved: Minimum score for "achieved" outcome (default: 0.8)
                Rationale: 80%+ completion indicates successful goal achievement
            outcome_threshold_partial: Minimum score for "partially_achieved" outcome (default: 0.4)
                Rationale: 40-80% completion indicates partial success
        """
        super().__init__(**kwargs)

        # Validate weights sum to 1.0
        if not abs((completion_weight + efficiency_weight) - 1.0) < 0.001:
            raise ValueError(
                f'completion_weight ({completion_weight}) and efficiency_weight ({efficiency_weight}) '
                f'must sum to 1.0, got {completion_weight + efficiency_weight}'
            )

        # Validate thresholds are in valid ranges
        if not 0 <= max_clarification_penalty <= 1:
            raise ValueError(
                f'max_clarification_penalty must be between 0 and 1, got {max_clarification_penalty}'
            )

        if not 0 <= clarification_penalty_rate <= 1:
            raise ValueError(
                f'clarification_penalty_rate must be between 0 and 1, got {clarification_penalty_rate}'
            )

        if not 0 <= goal_drift_threshold <= 1:
            raise ValueError(
                f'goal_drift_threshold must be between 0 and 1, got {goal_drift_threshold}'
            )

        if not outcome_threshold_partial < outcome_threshold_achieved:
            raise ValueError(
                f'outcome_threshold_achieved ({outcome_threshold_achieved}) must be greater than '
                f'outcome_threshold_partial ({outcome_threshold_partial})'
            )

        self.goal_key = goal_key
        self.completion_weight = completion_weight
        self.efficiency_weight = efficiency_weight
        self.bottleneck_threshold = bottleneck_threshold
        self.max_clarification_penalty = max_clarification_penalty
        self.clarification_penalty_rate = clarification_penalty_rate
        self.goal_drift_threshold = goal_drift_threshold
        self.outcome_threshold_achieved = outcome_threshold_achieved
        self.outcome_threshold_partial = outcome_threshold_partial

        self.moment_segmenter = MomentSegmenter(**kwargs)
        self.turn_analyzer = TurnAnalyzer(**kwargs)
        self.friction_analyzer = UserFrictionAnalyzer(**kwargs)

        # Single unified goal analyzer (replaces 3+ separate components)
        self.unified_goal_analyzer = UnifiedGoalAnalyzer(**kwargs)

    @trace(name='GoalCompletion', capture_args=True, capture_response=True)
    async def execute(
        self, item: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricEvaluationResult:
        """Execute goal completion analysis using unified approach."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=np.nan, explanation='No conversation provided.'
            )

        persona_goal = item.additional_input.get(self.goal_key, 'No goal specified.')
        transcript = (
            item.to_transcript()
            if hasattr(item, 'to_transcript')
            else self._build_transcript(item)
        )

        # Reuse Shared Conversation Analysis (Cached) ===
        moments = await get_or_compute_moments(item, cache, self.moment_segmenter)
        turn_analysis = await get_or_compute_turn_analysis(
            item, cache, self.turn_analyzer, self.friction_analyzer
        )

        # Single Unified Goal Analysis (Replaces 3+ LLM Calls) ===
        try:
            unified_result = await get_or_compute_unified_goal_analysis(
                item=item,
                cache=cache,
                unified_analyzer=self.unified_goal_analyzer,
                persona_goal=persona_goal,
                transcript=transcript,
                moments=moments,
                turn_analysis=turn_analysis,
            )

            primary_goal = unified_result.primary_goal
            goal_evolution = unified_result.goal_evolution
            subgoals = unified_result.sub_goals
            goal_moment_mappings = unified_result.goal_moment_mappings

        except Exception as e:
            logger.error(f'Unified goal analysis failed: {e}')
            return MetricEvaluationResult(
                score=np.nan, explanation=f'Goal analysis failed: {e}'
            )

        #  Compute Scores ===

        # Goal completion score (based on sub-goal achievement)
        status_scores = {
            'achieved': 1.0,
            'partially_achieved': 0.5,
            'not_achieved': 0.0,
            'abandoned': 0.0,
        }

        goal_completion_score = (
            sum(status_scores.get(sg.status, 0) for sg in subgoals) / len(subgoals)
            if subgoals
            else 0.0
        )

        # Efficiency score (based on moment efficiency and clarification penalties)
        avg_moment_efficiency = (
            sum(m.moment_efficiency for m in goal_moment_mappings)
            / len(goal_moment_mappings)
            if goal_moment_mappings
            else 0.5
        )

        # Penalty for excessive clarifications/retries
        avg_clarifications = (
            sum(sg.required_clarifications for sg in subgoals) / len(subgoals)
            if subgoals
            else 0
        )
        clarification_penalty = min(
            self.max_clarification_penalty,
            avg_clarifications * self.clarification_penalty_rate,
        )

        efficiency_score = max(0.0, avg_moment_efficiency - clarification_penalty)

        # Final composite score
        final_composite_score = (
            self.completion_weight * goal_completion_score
            + self.efficiency_weight * efficiency_score
        )

        # Detect Bottlenecks and Issues ===

        bottleneck_subgoals = [
            sg.description
            for sg in subgoals
            if sg.turns_to_complete and sg.turns_to_complete > self.bottleneck_threshold
        ]

        abandoned_early = any(
            sg.status == 'abandoned'
            and sg.first_mentioned_turn is not None
            and sg.first_mentioned_turn < len(turn_analysis) / 2
            for sg in subgoals
        )

        # Check if conversation drifted from goal
        mapped_moment_ids = {gm.moment_id for gm in goal_moment_mappings}
        unmapped_moments = [m for m in moments if m.id not in mapped_moment_ids]
        goal_drift_detected = (
            len(unmapped_moments) > len(moments) * self.goal_drift_threshold
            if moments
            else False
        )

        # Determine final outcome using configurable thresholds
        if goal_completion_score >= self.outcome_threshold_achieved:
            final_outcome = 'achieved'
        elif goal_completion_score >= self.outcome_threshold_partial:
            final_outcome = 'partially_achieved'
        elif abandoned_early:
            final_outcome = 'abandoned'
        else:
            final_outcome = 'failed'

        result_data = GoalCompletionResult(
            primary_goal=primary_goal,
            goal_evolution=goal_evolution,
            final_outcome=final_outcome,
            sub_goals=subgoals,
            goal_moment_mappings=goal_moment_mappings,
            conversation_moments=moments,
            turn_analysis=turn_analysis,
            goal_completion_score=goal_completion_score,
            efficiency_score=efficiency_score,
            final_composite_score=final_composite_score,
            bottleneck_subgoals=bottleneck_subgoals,
            abandoned_early=abandoned_early,
            goal_drift_detected=goal_drift_detected,
        )

        explanation = (
            f"Goal '{primary_goal[:50]}...' was {final_outcome}. "
            f'Completion: {goal_completion_score:.2f}, Efficiency: {efficiency_score:.2f}, '
            f'Final: {final_composite_score:.2f}. '
            f'{len(subgoals)} sub-goals tracked'
            + (
                f', {len(bottleneck_subgoals)} bottlenecks detected'
                if bottleneck_subgoals
                else ''
            )
        )

        # Compute cost estimate
        self.compute_cost_estimate(
            [
                self.moment_segmenter,
                self.turn_analyzer,
                self.friction_analyzer,
                self.unified_goal_analyzer,
            ]
        )

        return MetricEvaluationResult(
            score=final_composite_score, explanation=explanation, signals=result_data
        )

    def get_signals(
        self, result: GoalCompletionResult
    ) -> List[SignalDescriptor[GoalCompletionResult]]:
        """Generate comprehensive signals showing multi-layered analysis."""

        final_score = (
            self.last_result.score
            if hasattr(self, 'last_result') and self.last_result
            else 0.0
        )
        status_scores = {
            'achieved': 1.0,
            'partially_achieved': 0.5,
            'not_achieved': 0.0,
            'abandoned': 0.0,
        }

        signals = [
            # === HEADLINE ===
            SignalDescriptor(
                name='final_composite_score',
                extractor=lambda r: final_score,
                headline_display=True,
                description=f'Weighted score: {self.completion_weight}×completion + {self.efficiency_weight}×efficiency',
            ),
            # === SCORE COMPONENTS WITH CONFIGURATION VISIBILITY ===
            SignalDescriptor(
                name='goal_completion_score',
                extractor=lambda r: r.goal_completion_score,
                description='Pure goal achievement score (ignoring efficiency)',
            ),
            SignalDescriptor(
                name='efficiency_score',
                extractor=lambda r: r.efficiency_score,
                description='How efficiently goals were achieved',
            ),
            SignalDescriptor(
                name='completion_contribution',
                extractor=lambda r: r.goal_completion_score * self.completion_weight,
                description=f"Completion's contribution to final score (weight={self.completion_weight})",
            ),
            SignalDescriptor(
                name='efficiency_contribution',
                extractor=lambda r: r.efficiency_score * self.efficiency_weight,
                description=f"Efficiency's contribution to final score (weight={self.efficiency_weight})",
            ),
            # === CONFIGURATION PARAMETERS (for transparency) ===
            SignalDescriptor(
                name='config_completion_weight',
                extractor=lambda r: self.completion_weight,
                description='Configured weight for completion score',
            ),
            SignalDescriptor(
                name='config_efficiency_weight',
                extractor=lambda r: self.efficiency_weight,
                description='Configured weight for efficiency score',
            ),
            SignalDescriptor(
                name='config_bottleneck_threshold',
                extractor=lambda r: self.bottleneck_threshold,
                description='Configured turn threshold for bottleneck detection',
            ),
            SignalDescriptor(
                name='config_max_clarification_penalty',
                extractor=lambda r: self.max_clarification_penalty,
                description='Configured maximum clarification penalty',
            ),
            SignalDescriptor(
                name='config_goal_drift_threshold',
                extractor=lambda r: self.goal_drift_threshold,
                description='Configured threshold for goal drift detection',
            ),
            # === PRIMARY GOAL ===
            SignalDescriptor(
                name='primary_goal',
                extractor=lambda r: r.primary_goal,
                description="The user's main objective",
            ),
            SignalDescriptor(
                name='final_outcome',
                extractor=lambda r: r.final_outcome,
                score_mapping=status_scores,
                description=f'Final status (achieved≥{self.outcome_threshold_achieved}, partial≥{self.outcome_threshold_partial})',
            ),
            SignalDescriptor(
                name='goal_evolution',
                extractor=lambda r: r.goal_evolution,
                description='How the goal changed or was refined during conversation',
            ),
            # === AGGREGATE STATISTICS ===
            SignalDescriptor(
                name='total_subgoals',
                extractor=lambda r: len(r.sub_goals),
                description='Total number of sub-goals identified',
            ),
            SignalDescriptor(
                name='subgoals_achieved_count',
                extractor=lambda r: sum(
                    1 for sg in r.sub_goals if sg.status == 'achieved'
                ),
                description='Number of sub-goals fully achieved',
            ),
            SignalDescriptor(
                name='achievement_rate',
                extractor=lambda r: (
                    sum(1 for sg in r.sub_goals if sg.status == 'achieved')
                    / len(r.sub_goals)
                    if r.sub_goals
                    else 0.0
                ),
                description='Percentage of sub-goals fully achieved',
            ),
            # === TEMPORAL INSIGHTS ===
            SignalDescriptor(
                name='avg_turns_to_complete',
                extractor=lambda r: (
                    sum(
                        sg.turns_to_complete
                        for sg in r.sub_goals
                        if sg.turns_to_complete
                    )
                    / len([sg for sg in r.sub_goals if sg.turns_to_complete])
                    if any(sg.turns_to_complete for sg in r.sub_goals)
                    else 0
                ),
                description='Average turns needed to complete sub-goals',
            ),
            SignalDescriptor(
                name='total_clarifications_needed',
                extractor=lambda r: sum(
                    sg.required_clarifications for sg in r.sub_goals
                ),
                description='Total times user had to clarify across all sub-goals',
            ),
            SignalDescriptor(
                name='avg_clarifications_per_goal',
                extractor=lambda r: (
                    sum(sg.required_clarifications for sg in r.sub_goals)
                    / len(r.sub_goals)
                    if r.sub_goals
                    else 0
                ),
                description='Average clarifications per sub-goal',
            ),
            SignalDescriptor(
                name='total_agent_retries',
                extractor=lambda r: sum(sg.agent_retries for sg in r.sub_goals),
                description='Total times agent had to retry across all sub-goals',
            ),
            SignalDescriptor(
                name='clarification_penalty_applied',
                extractor=lambda r: min(
                    self.max_clarification_penalty,
                    (
                        sum(sg.required_clarifications for sg in r.sub_goals)
                        / len(r.sub_goals)
                        if r.sub_goals
                        else 0
                    )
                    * self.clarification_penalty_rate,
                ),
                description=f'Actual penalty applied (max={self.max_clarification_penalty}, rate={self.clarification_penalty_rate})',
            ),
            # === ISSUE FLAGS ===
            SignalDescriptor(
                name='bottleneck_count',
                extractor=lambda r: len(r.bottleneck_subgoals),
                description=f'Number of sub-goals that took >{self.bottleneck_threshold} turns',
            ),
            SignalDescriptor(
                name='abandoned_early',
                extractor=lambda r: r.abandoned_early,
                description='Whether user gave up before completing',
            ),
            SignalDescriptor(
                name='goal_drift_detected',
                extractor=lambda r: r.goal_drift_detected,
                description=f'Whether conversation drifted from original goal (>{self.goal_drift_threshold * 100}% moments unrelated)',
            ),
            SignalDescriptor(
                name='avg_moment_efficiency',
                extractor=lambda r: (
                    sum(m.moment_efficiency for m in r.goal_moment_mappings)
                    / len(r.goal_moment_mappings)
                    if r.goal_moment_mappings
                    else 0.5
                ),
                description='Average efficiency of moments addressing goals',
            ),
        ]

        # === PER-SUBGOAL SIGNALS===
        for i, sg in enumerate(result.sub_goals):
            desc_preview = sg.description[:60] + (
                '...' if len(sg.description) > 60 else ''
            )
            group = f'subgoal_{i + 1}: {desc_preview}'

            signals.extend(
                [
                    SignalDescriptor(
                        name='status',
                        group=group,
                        extractor=lambda r, idx=i: r.sub_goals[idx].status,
                        headline_display=True,
                        score_mapping=status_scores,
                        description='Completion status of this sub-goal',
                    ),
                    SignalDescriptor(
                        name='score_contribution',
                        group=group,
                        extractor=lambda r, idx=i: (
                            status_scores.get(r.sub_goals[idx].status, 0)
                            / len(r.sub_goals)
                        ),
                        description="This sub-goal's contribution to completion score",
                    ),
                    SignalDescriptor(
                        name='first_mentioned',
                        group=group,
                        extractor=lambda r, idx=i: (
                            f'Turn {r.sub_goals[idx].first_mentioned_turn}'
                            if r.sub_goals[idx].first_mentioned_turn is not None
                            else 'Never'
                        ),
                        description='When this sub-goal was first mentioned',
                    ),
                    SignalDescriptor(
                        name='achieved_at',
                        group=group,
                        extractor=lambda r, idx=i: (
                            f'Turn {r.sub_goals[idx].achieved_at_turn}'
                            if r.sub_goals[idx].achieved_at_turn is not None
                            else 'Never'
                        ),
                        description='When this sub-goal was achieved',
                    ),
                    SignalDescriptor(
                        name='turns_to_complete',
                        group=group,
                        extractor=lambda r, idx=i: r.sub_goals[idx].turns_to_complete
                        or 'N/A',
                        description='Number of turns to complete',
                    ),
                    SignalDescriptor(
                        name='clarifications',
                        group=group,
                        extractor=lambda r, idx=i: r.sub_goals[
                            idx
                        ].required_clarifications,
                        description='Times user had to clarify',
                    ),
                    SignalDescriptor(
                        name='agent_retries',
                        group=group,
                        extractor=lambda r, idx=i: r.sub_goals[idx].agent_retries,
                        description='Times agent had to retry',
                    ),
                    SignalDescriptor(
                        name='description',
                        group=group,
                        extractor=lambda r, idx=i: r.sub_goals[idx].description,
                        description='Full sub-goal description',
                    ),
                ]
            )

        # === GOAL-MOMENT MAPPINGS ===
        for i, mapping in enumerate(result.goal_moment_mappings):
            topic_preview = mapping.moment_topic[:50] + (
                '...' if len(mapping.moment_topic) > 50 else ''
            )
            group = f'moment_{mapping.moment_id}: {topic_preview}'

            signals.extend(
                [
                    SignalDescriptor(
                        name='efficiency',
                        group=group,
                        extractor=lambda r, idx=i: r.goal_moment_mappings[
                            idx
                        ].moment_efficiency,
                        headline_display=True,
                        description='How efficiently this moment addressed sub-goals',
                    ),
                    SignalDescriptor(
                        name='subgoals_addressed',
                        group=group,
                        extractor=lambda r, idx=i: (
                            ', '.join(
                                r.goal_moment_mappings[idx].addressed_subgoals[:2]
                            )
                            + (
                                '...'
                                if len(r.goal_moment_mappings[idx].addressed_subgoals)
                                > 2
                                else ''
                            )
                        ),
                        description='Which sub-goals were addressed',
                    ),
                    SignalDescriptor(
                        name='subgoal_count',
                        group=group,
                        extractor=lambda r, idx=i: len(
                            r.goal_moment_mappings[idx].addressed_subgoals
                        ),
                        description='Number of sub-goals addressed in this moment',
                    ),
                ]
            )

        return signals

    @staticmethod
    def _build_transcript(item: DatasetItem) -> str:
        """Build transcript from conversation messages."""
        if not item.conversation:
            return ''

        lines = []
        for msg in item.conversation.messages:
            role = 'User' if isinstance(msg, HumanMessage) else 'Agent'
            content = msg.content or ''
            lines.append(f'{role}: {content}')
        return '\n'.join(lines)
