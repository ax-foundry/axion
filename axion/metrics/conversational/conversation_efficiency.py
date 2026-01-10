from typing import List, Literal, Optional

import numpy as np
from axion._core.logging import get_logger
from axion._core.schema import (
    AIMessage,
    HumanMessage,
    RichBaseModel,
    ToolMessage,
)
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
    get_or_compute_moments,
    get_or_compute_turn_analysis,
)
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

logger = get_logger(__name__)


class InefficiencyType(RichBaseModel):
    """Represents a specific type of inefficiency in the conversation."""

    type: Literal[
        'redundant_question',
        'circular_logic',
        'unnecessary_clarification',
        'repetitive_information',
        'off_topic_detour',
        'premature_tool_use',
        'delayed_action',
        'redundant_tool_call',
        'incorrect_tool_call',
    ]
    turn_indices: List[int]
    description: str
    turns_wasted: int
    severity: Literal['minor', 'moderate', 'severe']
    tool_related: bool = Field(
        default=False, description='Whether this inefficiency involves tool usage'
    )


class OptimalPath(RichBaseModel):
    """Describes what an optimal conversation path would look like."""

    optimal_turn_count: int
    key_steps: List[str]
    removed_steps: List[str] = Field(default_factory=list)
    efficiency_gain: float


class GoalIdentificationInput(RichBaseModel):
    """Input for identifying the user's goal from conversation."""

    conversation_transcript: str
    conversation_moments: List[ConversationMoment]


class GoalIdentificationOutput(RichBaseModel):
    """Output from goal identification."""

    primary_goal: str
    goal_complexity: Literal['simple', 'moderate', 'complex']
    expected_turn_range: str


class GoalIdentifier(BaseMetric[GoalIdentificationInput, GoalIdentificationOutput]):
    """Identifies the user's goal and estimates complexity."""

    as_structured_llm = True

    instruction = """Analyze the conversation to identify the user's primary goal.

Determine:
1. Primary Goal: What is the user trying to accomplish?
2. Goal Complexity: How complex is this goal?
   - "simple": Single-step task (e.g., "check my balance")
   - "moderate": Multi-step task (e.g., "transfer money")
   - "complex": Multi-faceted task (e.g., "compare options, make decision")
3. Expected Turn Range: How many turns would be reasonable?
   - Simple: "2-4 turns"
   - Moderate: "5-8 turns"
   - Complex: "10-15 turns"
"""

    input_model = GoalIdentificationInput
    output_model = GoalIdentificationOutput

    examples = [
        (
            GoalIdentificationInput(
                conversation_transcript="User: I need to reset my password\nAgent: What's your email?\nUser: john@example.com\nAgent: Reset link sent!",
                conversation_moments=[
                    ConversationMoment(
                        id=1,
                        topic='Password Reset',
                        start_turn=0,
                        end_turn=3,
                        turn_count=4,
                    )
                ],
            ),
            GoalIdentificationOutput(
                primary_goal='Reset password',
                goal_complexity='simple',
                expected_turn_range='2-4 turns',
            ),
        ),
    ]


class InefficiencyDetectionInput(RichBaseModel):
    """Input for detecting inefficiencies."""

    conversation_transcript: str
    primary_goal: str
    turn_by_turn_analysis: List[TurnAnalysis]
    conversation_moments: List[ConversationMoment]


class InefficiencyDetectionOutput(RichBaseModel):
    """Output from inefficiency detection."""

    inefficiencies: List[InefficiencyType]
    total_wasted_turns: int
    primary_inefficiency_category: Optional[str] = None


class InefficiencyDetector(
    BaseMetric[InefficiencyDetectionInput, InefficiencyDetectionOutput]
):
    """Detects inefficiencies in the conversation path."""

    as_structured_llm = True

    instruction = """Analyze the conversation for inefficiencies that prevented the optimal path.

Goal: {primary_goal}

Inefficiency Types:
1. redundant_question: Agent asks for already-provided information
2. circular_logic: Conversation loops unnecessarily
3. unnecessary_clarification: Agent asks when context was clear
4. repetitive_information: Agent repeats same information
5. off_topic_detour: Conversation drifts from goal
6. premature_tool_use: Agent uses tools before gathering info
7. delayed_action: Agent takes too long to execute action

For each inefficiency:
- Identify type
- List turn indices
- Explain what happened
- Estimate turns wasted
- Rate severity
"""

    input_model = InefficiencyDetectionInput
    output_model = InefficiencyDetectionOutput

    examples = [
        (
            InefficiencyDetectionInput(
                conversation_transcript="User: Reset password for john@example.com\nAgent: What's your email?\nUser: john@example.com\nAgent: Reset link sent",
                primary_goal='Reset password',
                turn_by_turn_analysis=[],
                conversation_moments=[],
            ),
            InefficiencyDetectionOutput(
                inefficiencies=[
                    InefficiencyType(
                        type='redundant_question',
                        turn_indices=[1],
                        description='Asked for email already provided',
                        turns_wasted=2,
                        severity='moderate',
                    )
                ],
                total_wasted_turns=2,
                primary_inefficiency_category='redundant_question',
            ),
        ),
    ]

    def _build_instruction_with_goal(self, goal: str) -> str:
        return self.instruction.format(primary_goal=goal)

    @trace(name='execute', capture_args=True, capture_response=True)
    async def execute(
        self, input_data: InefficiencyDetectionInput
    ) -> InefficiencyDetectionOutput:
        original_instruction = self.instruction
        self.instruction = self._build_instruction_with_goal(input_data.primary_goal)
        try:
            result = await super().execute(input_data)
            return result
        finally:
            self.instruction = original_instruction


class OptimalPathAnalysisInput(RichBaseModel):
    """Input for optimal path analysis."""

    conversation_transcript: str
    primary_goal: str
    goal_complexity: str
    expected_turn_range: str
    actual_turn_count: int
    inefficiencies: List[InefficiencyType]


class OptimalPathAnalyzer(BaseMetric[OptimalPathAnalysisInput, OptimalPath]):
    """Determines the optimal conversation path."""

    as_structured_llm = True

    instruction = """Design the optimal conversation path.

Goal: {primary_goal}
Complexity: {goal_complexity}
Expected: {expected_turn_range}
Actual: {actual_turn_count} turns

Design the most efficient path:
1. Optimal Turn Count: Minimum turns needed
2. Key Steps: List essential steps only
3. Removed Steps: Actual steps that could be eliminated
4. Efficiency Gain: (actual - optimal) / actual

Be realistic - don't over-optimize.
"""

    input_model = OptimalPathAnalysisInput
    output_model = OptimalPath

    examples = [
        (
            OptimalPathAnalysisInput(
                conversation_transcript="User: Reset password for john@example.com\nAgent: What's your email?\nUser: john@example.com\nAgent: Reset link sent",
                primary_goal='Reset password',
                goal_complexity='simple',
                expected_turn_range='2-4 turns',
                actual_turn_count=4,
                inefficiencies=[],
            ),
            OptimalPath(
                optimal_turn_count=2,
                key_steps=[
                    'User provides reset request with email',
                    'Agent sends reset link',
                ],
                removed_steps=['Agent asks for email', 'User repeats email'],
                efficiency_gain=0.5,
            ),
        ),
    ]

    def _build_instruction_with_context(
        self, goal: str, complexity: str, expected_range: str, actual_count: int
    ) -> str:
        return self.instruction.format(
            primary_goal=goal,
            goal_complexity=complexity,
            expected_turn_range=expected_range,
            actual_turn_count=actual_count,
        )

    @trace(name='execute', capture_args=True, capture_response=True)
    async def execute(self, input_data: OptimalPathAnalysisInput) -> OptimalPath:
        original_instruction = self.instruction
        self.instruction = self._build_instruction_with_context(
            input_data.primary_goal,
            input_data.goal_complexity,
            input_data.expected_turn_range,
            input_data.actual_turn_count,
        )
        try:
            result = await super().execute(input_data)
            return result
        finally:
            self.instruction = original_instruction


class ConversationEfficiencyResult(RichBaseModel):
    """Complete result for conversation efficiency metric."""

    primary_goal: str
    goal_complexity: str
    expected_turn_range: str

    # Turn counts (excluding tool messages)
    actual_turn_count: int = Field(
        description='Total conversational turns (excludes tool messages)'
    )
    actual_agent_turn_count: int = Field(description='Agent conversational turns only')
    total_messages: int = Field(description='Total messages including tool messages')
    tool_message_count: int = Field(description='Number of tool messages')
    tool_call_count: int = Field(description='Number of tool calls made')

    inefficiencies: List[InefficiencyType]
    total_wasted_turns: int
    primary_inefficiency_category: Optional[str]
    optimal_path: OptimalPath
    redundancy_score: float
    efficiency_score: float
    optimal_path_score: float
    final_composite_score: float
    conversation_moments: List[ConversationMoment]
    turn_analysis: List[TurnAnalysis]


@metric(
    name='Conversation Efficiency',
    key='conversation_efficiency',
    description='Evaluates whether the conversation achieved its goal through the most efficient path',
    required_fields=['conversation'],
    optional_fields=[],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['agent', 'multi_turn'],
)
class ConversationEfficiency(BaseMetric):
    """
    Conversation Efficiency Metric

    Evaluates whether a conversation achieved its goal through the most efficient path possible.
    Identifies redundant turns, unnecessary questions, circular logic, and suggests optimizations.

    This metric reuses conversation analysis components from shared_conversation_analysis
    to minimize LLM calls and ensure consistency across metrics.
    ."""

    def __init__(
        self,
        efficiency_weight: float = 0.6,
        optimal_path_weight: float = 0.4,
        severe_inefficiency_penalty: float = 0.15,
        moderate_inefficiency_penalty: float = 0.08,
        minor_inefficiency_penalty: float = 0.03,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not abs((efficiency_weight + optimal_path_weight) - 1.0) < 0.001:
            raise ValueError('Weights must sum to 1.0')

        self.efficiency_weight = efficiency_weight
        self.optimal_path_weight = optimal_path_weight
        self.severe_penalty = severe_inefficiency_penalty
        self.moderate_penalty = moderate_inefficiency_penalty
        self.minor_penalty = minor_inefficiency_penalty

        self.moment_segmenter = MomentSegmenter(**kwargs)
        self.turn_analyzer = TurnAnalyzer(**kwargs)
        self.goal_identifier = GoalIdentifier(**kwargs)
        self.inefficiency_detector = InefficiencyDetector(**kwargs)
        self.optimal_path_analyzer = OptimalPathAnalyzer(**kwargs)

    @trace(name='execute', capture_args=True, capture_response=True)
    async def execute(
        self, item: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricEvaluationResult:
        if not item.conversation or not item.conversation.messages:
            return MetricEvaluationResult(
                score=np.nan, explanation='No conversation provided.'
            )

        # Count different message types
        total_messages = len(item.conversation.messages)
        tool_message_count = sum(
            1 for m in item.conversation.messages if isinstance(m, ToolMessage)
        )

        # Conversational turns exclude ToolMessage (system responses)
        conversational_messages = [
            m for m in item.conversation.messages if not isinstance(m, ToolMessage)
        ]
        actual_turn_count = len(conversational_messages)
        actual_agent_turn_count = sum(
            1 for m in conversational_messages if isinstance(m, AIMessage)
        )

        # Count tool calls
        tool_call_count = sum(
            len(m.tool_calls)
            for m in item.conversation.messages
            if isinstance(m, AIMessage) and hasattr(m, 'tool_calls') and m.tool_calls
        )

        transcript = self._build_transcript(item)

        # Shared analysis (only on conversational messages)
        moments = await get_or_compute_moments(item, cache, self.moment_segmenter)
        turn_analysis = await get_or_compute_turn_analysis(
            item, cache, self.turn_analyzer
        )

        # Identify goal
        try:
            goal_result = await self.goal_identifier.execute(
                GoalIdentificationInput(
                    conversation_transcript=transcript, conversation_moments=moments
                )
            )
        except Exception as e:
            logger.error(f'Goal identification failed: {e}')
            return MetricEvaluationResult(
                score=np.nan, explanation=f'Goal identification failed: {e}'
            )

        # Detect inefficiencies
        try:
            inefficiency_result = await self.inefficiency_detector.execute(
                InefficiencyDetectionInput(
                    conversation_transcript=transcript,
                    primary_goal=goal_result.primary_goal,
                    turn_by_turn_analysis=turn_analysis,
                    conversation_moments=moments,
                )
            )
        except Exception as e:
            logger.warning(f'Inefficiency detection failed: {e}')
            inefficiency_result = InefficiencyDetectionOutput(
                inefficiencies=[],
                total_wasted_turns=0,
                primary_inefficiency_category=None,
            )

        # Optimal path (based on conversational turns only)
        try:
            optimal_result = await self.optimal_path_analyzer.execute(
                OptimalPathAnalysisInput(
                    conversation_transcript=transcript,
                    primary_goal=goal_result.primary_goal,
                    goal_complexity=goal_result.goal_complexity,
                    expected_turn_range=goal_result.expected_turn_range,
                    actual_turn_count=actual_turn_count,
                    inefficiencies=inefficiency_result.inefficiencies,
                )
            )
        except Exception as e:
            logger.warning(f'Optimal path analysis failed: {e}')
            optimal_count = {'simple': 3, 'moderate': 7, 'complex': 12}.get(
                goal_result.goal_complexity, 5
            )
            optimal_result = OptimalPath(
                optimal_turn_count=optimal_count,
                key_steps=['Unable to determine'],
                removed_steps=[],
                efficiency_gain=max(
                    0.0, (actual_turn_count - optimal_count) / actual_turn_count
                ),
            )

        # Compute scores (based on conversational turns)
        redundancy_score = min(
            1.0,
            inefficiency_result.total_wasted_turns / actual_turn_count
            if actual_turn_count > 0
            else 0.0,
        )
        base_efficiency_score = 1.0 - redundancy_score

        penalties = sum(
            self.severe_penalty
            if i.severity == 'severe'
            else self.moderate_penalty
            if i.severity == 'moderate'
            else self.minor_penalty
            for i in inefficiency_result.inefficiencies
        )

        efficiency_score = max(0.0, base_efficiency_score - penalties)

        optimal_path_score = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    (actual_turn_count - optimal_result.optimal_turn_count)
                    / actual_turn_count
                    if actual_turn_count > 0
                    else 0.0
                ),
            ),
        )

        final_composite_score = (
            self.efficiency_weight * efficiency_score
            + self.optimal_path_weight * optimal_path_score
        )

        result_data = ConversationEfficiencyResult(
            primary_goal=goal_result.primary_goal,
            goal_complexity=goal_result.goal_complexity,
            expected_turn_range=goal_result.expected_turn_range,
            actual_turn_count=actual_turn_count,
            actual_agent_turn_count=actual_agent_turn_count,
            total_messages=total_messages,
            tool_message_count=tool_message_count,
            tool_call_count=tool_call_count,
            inefficiencies=inefficiency_result.inefficiencies,
            total_wasted_turns=inefficiency_result.total_wasted_turns,
            primary_inefficiency_category=inefficiency_result.primary_inefficiency_category,
            optimal_path=optimal_result,
            redundancy_score=redundancy_score,
            efficiency_score=efficiency_score,
            optimal_path_score=optimal_path_score,
            final_composite_score=final_composite_score,
            conversation_moments=moments,
            turn_analysis=turn_analysis,
        )

        explanation = (
            f"Goal '{goal_result.primary_goal}' ({goal_result.goal_complexity}): "
            f'{actual_turn_count} turns ({tool_call_count} tool calls, optimal: {optimal_result.optimal_turn_count}). '
            f'{inefficiency_result.total_wasted_turns} wasted, '
            f'{len(inefficiency_result.inefficiencies)} inefficiencies. '
            f'Score: {final_composite_score:.2f}'
        )

        self.compute_cost_estimate(
            [
                self.moment_segmenter,
                self.turn_analyzer,
                self.goal_identifier,
                self.inefficiency_detector,
                self.optimal_path_analyzer,
            ]
        )

        return MetricEvaluationResult(
            score=final_composite_score, explanation=explanation, signals=result_data
        )

    def get_signals(
        self, result: ConversationEfficiencyResult
    ) -> List[SignalDescriptor[ConversationEfficiencyResult]]:
        final_score = (
            self.last_result.score
            if hasattr(self, 'last_result') and self.last_result
            else 0.0
        )

        signals = [
            SignalDescriptor(
                name='final_composite_score',
                extractor=lambda r: final_score,
                headline_display=True,
            ),
            SignalDescriptor(
                name='efficiency_score', extractor=lambda r: r.efficiency_score
            ),
            SignalDescriptor(
                name='optimal_path_score', extractor=lambda r: r.optimal_path_score
            ),
            SignalDescriptor(
                name='redundancy_score', extractor=lambda r: r.redundancy_score
            ),
            SignalDescriptor(name='primary_goal', extractor=lambda r: r.primary_goal),
            SignalDescriptor(
                name='goal_complexity', extractor=lambda r: r.goal_complexity
            ),
            SignalDescriptor(
                name='actual_turn_count',
                extractor=lambda r: r.actual_turn_count,
                description='Conversational turns (excludes tool messages)',
            ),
            SignalDescriptor(
                name='total_messages',
                extractor=lambda r: r.total_messages,
                description='Total messages including tool responses',
            ),
            SignalDescriptor(
                name='tool_message_count',
                extractor=lambda r: r.tool_message_count,
                description='Number of tool response messages',
            ),
            SignalDescriptor(
                name='tool_call_count',
                extractor=lambda r: r.tool_call_count,
                description='Number of tool calls made',
            ),
            SignalDescriptor(
                name='optimal_turn_count',
                extractor=lambda r: r.optimal_path.optimal_turn_count,
            ),
            SignalDescriptor(
                name='total_wasted_turns', extractor=lambda r: r.total_wasted_turns
            ),
            SignalDescriptor(
                name='total_inefficiencies', extractor=lambda r: len(r.inefficiencies)
            ),
        ]

        for i, ineff in enumerate(result.inefficiencies[:5]):
            group = f'inefficiency_{i + 1}'
            signals.extend(
                [
                    SignalDescriptor(
                        name='type',
                        group=group,
                        extractor=lambda r, idx=i: r.inefficiencies[idx].type,
                        headline_display=True,
                    ),
                    SignalDescriptor(
                        name='severity',
                        group=group,
                        extractor=lambda r, idx=i: r.inefficiencies[idx].severity,
                    ),
                    SignalDescriptor(
                        name='turns_wasted',
                        group=group,
                        extractor=lambda r, idx=i: r.inefficiencies[idx].turns_wasted,
                    ),
                    SignalDescriptor(
                        name='description',
                        group=group,
                        extractor=lambda r, idx=i: r.inefficiencies[idx].description,
                    ),
                ]
            )

        return signals

    @staticmethod
    def _build_transcript(item: DatasetItem) -> str:
        """Build transcript from conversation messages, handling tool calls properly."""
        if not item.conversation:
            return ''

        lines = []
        for msg in item.conversation.messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content or ''}")
            elif isinstance(msg, AIMessage):
                # Check for tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_call_strs = []
                    for tc in msg.tool_calls:
                        args_str = (
                            ', '.join(f'{k}={v}' for k, v in tc.args.items())
                            if hasattr(tc, 'args') and tc.args
                            else ''
                        )
                        tool_call_strs.append(f'{tc.name}({args_str})')
                    lines.append(f"Agent: [Tool Calls: {', '.join(tool_call_strs)}]")
                # Regular content
                if msg.content:
                    lines.append(f'Agent: {msg.content}')
            elif isinstance(msg, ToolMessage):
                # Tool messages are system responses, mark them clearly
                lines.append(
                    f"[Tool Response: {msg.content[:100]}{'...' if len(msg.content or '') > 100 else ''}]"
                )

        return '\n'.join(lines)
