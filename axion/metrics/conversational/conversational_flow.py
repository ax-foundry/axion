from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
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
    MomentCoherenceEvaluator,
    MomentSegmenter,
    TurnAnalysis,
    TurnAnalyzer,
    UserFrictionAnalyzer,
    get_or_compute_moments,
    get_or_compute_turn_analysis,
)
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

logger = get_logger(__name__)


@dataclass
class FlowConfig:
    """Centralized configuration for conversation flow analysis."""

    # Penalty weights for different issue types
    penalty_weights: Dict[str, float] = field(
        default_factory=lambda: {
            'state_loop': 0.25,
            'effort_degradation': 0.30,
            'correction_cascade': 0.20,
            'coherence_drop': 0.15,
        }
    )

    # Maximum total penalty (prevents excessive score reduction)
    max_total_penalty: float = 0.50

    # User effort analysis
    effort_window_size: int = 3
    min_turns_for_degradation: int = 6
    degradation_threshold: float = 0.15  # Minimum increase to flag

    # State loop detection
    loop_detection_enabled: bool = True
    loop_window_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])

    # Correction cascade detection
    cascade_window_size: int = 4
    cascade_threshold: int = 2  # Min corrections in window

    # Score calculation
    use_geometric_mean: bool = False  # True = emphasize weakest dimension

    # User effort calculation weights
    effort_weights: Dict[str, float] = field(
        default_factory=lambda: {
            'friction': 0.50,
            'length': 0.20,
            'repetition': 0.30,
        }
    )


class IssueType(str, Enum):
    """Enumerated issue types for type-safe detection."""

    STATE_LOOP = 'state_loop'
    EFFORT_DEGRADATION = 'effort_degradation'
    CORRECTION_CASCADE = 'correction_cascade'
    COHERENCE_DROP = 'coherence_drop'


class FlowIssue(RichBaseModel):
    """Structured representation of a detected conversation issue."""

    type: IssueType = Field(description='Type of issue detected')
    severity: float = Field(description='Severity score 0-1', ge=0.0, le=1.0)
    turn_index: Optional[int] = Field(
        default=None, description='Turn where issue occurs'
    )
    turn_range: Optional[tuple] = Field(
        default=None, description='Range of affected turns'
    )
    description: str = Field(description='Human-readable explanation')
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description='Additional context'
    )

    def get_penalty(self, weights: Dict[str, float]) -> float:
        """Calculate penalty contribution based on severity and configured weight."""
        weight = weights.get(self.type.value, 0.2)
        return weight * self.severity


class FlowScoreComponents(RichBaseModel):
    """Decomposed score components for transparency."""

    # Core dimensions
    coherence: float = Field(description='Conversation coherence (0-1)', ge=0.0, le=1.0)
    efficiency: float = Field(
        description='Conversation efficiency (0-1)', ge=0.0, le=1.0
    )
    user_experience: float = Field(
        description='User experience quality (0-1)', ge=0.0, le=1.0
    )

    # Penalty
    raw_penalty: float = Field(
        description='Sum of all penalties before capping', ge=0.0
    )
    capped_penalty: float = Field(description='Penalty after applying cap', ge=0.0)

    # Final
    base_score: float = Field(description='Score before penalties', ge=0.0, le=1.0)
    final_score: float = Field(
        description='Final score after penalties', ge=0.0, le=1.0
    )

    def explain(self) -> str:
        """Generate human-readable explanation of score calculation."""
        return (
            f'Base Score: {self.base_score:.3f} '
            f'(coherence={self.coherence:.2f}, efficiency={self.efficiency:.2f}, '
            f'ux={self.user_experience:.2f}) '
            f'- Penalty: {self.capped_penalty:.3f} '
            f'= Final: {self.final_score:.3f}'
        )


class StateLoopDetector:
    """Detects conversation state loops and oscillations."""

    def __init__(self, config: FlowConfig):
        self.config = config

    def detect(self, state_progression: List[Dict[str, Any]]) -> List[FlowIssue]:
        """
        Detect state loops using multiple pattern matching strategies.

        Patterns detected:
        - Immediate reversions (A → B → A)
        - Longer cycles (A → B → C → A)
        - Oscillations (A ↔ B ↔ A ↔ B)
        """
        if not self.config.loop_detection_enabled:
            return []

        issues = []
        states = [s['state'] for s in state_progression]

        if len(states) < 3:
            return issues

        # Strategy 1: Detect immediate back-and-forth
        for i in range(len(states) - 2):
            if states[i] == states[i + 2] and states[i] != states[i + 1]:
                # Exclude legitimate clarification loops
                if states[i] in ['Intent Clarification', 'Confirmation']:
                    continue

                issues.append(
                    FlowIssue(
                        type=IssueType.STATE_LOOP,
                        severity=0.4,
                        turn_index=state_progression[i + 2]['start_turn'],
                        turn_range=(
                            state_progression[i]['start_turn'],
                            state_progression[i + 2]['end_turn'],
                        ),
                        description=f"Reverted to '{states[i]}' after '{states[i + 1]}'",
                        metadata={'pattern': [states[i], states[i + 1], states[i]]},
                    )
                )

        # Strategy 2: Detect longer repeated sequences
        for window_size in self.config.loop_window_sizes:
            if len(states) < window_size * 2:
                continue

            for i in range(len(states) - window_size * 2 + 1):
                window1 = states[i : i + window_size]
                window2 = states[i + window_size : i + window_size * 2]

                if window1 == window2:
                    issues.append(
                        FlowIssue(
                            type=IssueType.STATE_LOOP,
                            severity=min(1.0, 0.5 + window_size * 0.1),
                            turn_index=state_progression[i + window_size]['start_turn'],
                            turn_range=(
                                state_progression[i]['start_turn'],
                                state_progression[i + window_size * 2 - 1]['end_turn'],
                            ),
                            description=f"Repeated sequence: {' → '.join(window1)}",
                            metadata={'pattern': window1, 'repetitions': 2},
                        )
                    )
                    break  # Avoid overlapping detections

        return issues


class UserEffortAnalyzer:
    """Analyzes user effort trajectory for degradation patterns."""

    def __init__(self, config: FlowConfig):
        self.config = config

    def detect_degradation(self, effort_trajectory: List[float]) -> Optional[FlowIssue]:
        """
        Detect statistically significant increases in user effort.

        Uses moving averages and trend analysis to identify degradation.
        """
        if len(effort_trajectory) < self.config.min_turns_for_degradation:
            return None

        # Calculate moving average to smooth noise
        window = self.config.effort_window_size
        moving_avg = self._calculate_moving_average(effort_trajectory, window)

        # Compare first half vs second half
        mid = len(moving_avg) // 2
        first_half_avg = np.mean(moving_avg[:mid]) if mid > 0 else 0
        second_half_avg = np.mean(moving_avg[mid:]) if len(moving_avg) > mid else 0

        increase = second_half_avg - first_half_avg

        if increase > self.config.degradation_threshold:
            # Calculate severity based on magnitude of increase
            severity = min(1.0, increase / 0.5)  # Normalized to 0.5 max increase

            return FlowIssue(
                type=IssueType.EFFORT_DEGRADATION,
                severity=severity,
                description=f'User effort increased by {increase:.2f} points',
                metadata={
                    'first_half_avg': first_half_avg,
                    'second_half_avg': second_half_avg,
                    'increase': increase,
                    'trajectory': effort_trajectory,
                },
            )

        return None

    @staticmethod
    def _calculate_moving_average(values: List[float], window: int) -> List[float]:
        """Calculate moving average with given window size."""
        if not values:
            return []

        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_values = values[start : i + 1]
            result.append(sum(window_values) / len(window_values))

        return result


class CorrectionCascadeDetector:
    """Detects patterns of repeated corrections and clarifications."""

    def __init__(self, config: FlowConfig):
        self.config = config

    def detect(self, turn_analysis: List[TurnAnalysis]) -> List[FlowIssue]:
        """
        Detect correction cascades indicating agent confusion or misunderstanding.

        A cascade is multiple correction-related interactions in a short span.
        """
        issues = []

        correction_indicators = {
            ('user', 'disagreement'),
            ('agent', 'apology'),
        }

        window = self.config.cascade_window_size

        for i in range(len(turn_analysis) - window + 1):
            window_turns = turn_analysis[i : i + window]

            # Count correction-related interactions
            correction_count = sum(
                1
                for t in window_turns
                if (t.role, t.dialogue_act) in correction_indicators
            )

            if correction_count >= self.config.cascade_threshold:
                severity = min(1.0, correction_count / window)

                issues.append(
                    FlowIssue(
                        type=IssueType.CORRECTION_CASCADE,
                        severity=severity,
                        turn_index=i,
                        turn_range=(i, i + window - 1),
                        description=f'{correction_count} corrections in {window} turns',
                        metadata={
                            'correction_count': correction_count,
                            'window_size': window,
                            'turns': [
                                (t.turn_index, t.role, t.dialogue_act)
                                for t in window_turns
                            ],
                        },
                    )
                )

        return issues


class CoherenceDropDetector:
    """Detects significant drops in moment coherence."""

    def __init__(self, config: FlowConfig):
        self.config = config

    def detect(self, moments: List[ConversationMoment]) -> List[FlowIssue]:
        """Detect moments with significantly lower coherence than average."""
        if len(moments) < 2:
            return []

        issues = []
        coherence_scores = [m.coherence_score for m in moments]
        avg_coherence = np.mean(coherence_scores)
        std_coherence = np.std(coherence_scores) if len(coherence_scores) > 1 else 0

        for moment in moments:
            # Flag moments more than 1 std dev below average
            if (
                moment.coherence_score < avg_coherence - std_coherence
                and moment.coherence_score < 0.5
            ):
                severity = 1.0 - moment.coherence_score

                issues.append(
                    FlowIssue(
                        type=IssueType.COHERENCE_DROP,
                        severity=severity,
                        turn_index=moment.start_turn,
                        turn_range=(moment.start_turn, moment.end_turn),
                        description=f"Low coherence in '{moment.topic[:50]}' (score={moment.coherence_score:.2f})",
                        metadata={
                            'moment_id': moment.id,
                            'coherence': moment.coherence_score,
                            'avg_coherence': avg_coherence,
                        },
                    )
                )

        return issues


class IssueDetector:
    """Orchestrates all issue detection strategies."""

    def __init__(self, config: FlowConfig):
        self.config = config
        self.state_loop_detector = StateLoopDetector(config)
        self.effort_analyzer = UserEffortAnalyzer(config)
        self.cascade_detector = CorrectionCascadeDetector(config)
        self.coherence_detector = CoherenceDropDetector(config)

    def detect_all(
        self,
        moments: List[ConversationMoment],
        turn_analysis: List[TurnAnalysis],
        state_progression: List[Dict[str, Any]],
    ) -> List[FlowIssue]:
        """Run all detection strategies and aggregate results."""
        all_issues = []

        # Detect state loops
        all_issues.extend(self.state_loop_detector.detect(state_progression))

        # Detect user effort degradation
        effort_trajectory = [
            t.user_effort_score
            for t in turn_analysis
            if t.user_effort_score is not None
        ]
        if effort_issue := self.effort_analyzer.detect_degradation(effort_trajectory):
            all_issues.append(effort_issue)

        # Detect correction cascades
        all_issues.extend(self.cascade_detector.detect(turn_analysis))

        # Detect coherence drops
        all_issues.extend(self.coherence_detector.detect(moments))

        return all_issues


class FlowScoreCalculator:
    """Calculates final flow score from multiple components."""

    def __init__(self, config: FlowConfig):
        self.config = config

    def calculate(
        self,
        moments: List[ConversationMoment],
        turn_analysis: List[TurnAnalysis],
        issues: List[FlowIssue],
    ) -> FlowScoreComponents:
        """
        Calculate comprehensive flow score with transparent components.

        Components:
        1. Coherence: Weighted average of moment coherence
        2. Efficiency: Ratio of productive turns (heuristic)
        3. User Experience: Inverse of average user effort
        4. Penalties: From detected issues
        """
        # Calculate coherence
        coherence = self._calculate_coherence(moments)

        # Calculate efficiency
        efficiency = self._calculate_efficiency(turn_analysis, moments)

        # Calculate user experience
        user_experience = self._calculate_user_experience(turn_analysis)

        # Calculate base score
        if self.config.use_geometric_mean:
            # Geometric mean emphasizes weakest dimension
            base_score = (coherence * efficiency * user_experience) ** (1 / 3)
        else:
            # Weighted average (can be configured)
            base_score = 0.40 * coherence + 0.30 * efficiency + 0.30 * user_experience

        # Calculate penalties
        raw_penalty = sum(
            issue.get_penalty(self.config.penalty_weights) for issue in issues
        )
        capped_penalty = min(raw_penalty, self.config.max_total_penalty)

        # Final score
        final_score = max(0.0, base_score - capped_penalty)

        return FlowScoreComponents(
            coherence=coherence,
            efficiency=efficiency,
            user_experience=user_experience,
            raw_penalty=raw_penalty,
            capped_penalty=capped_penalty,
            base_score=base_score,
            final_score=final_score,
        )

    @staticmethod
    def _calculate_coherence(moments: List[ConversationMoment]) -> float:
        """Calculate weighted average coherence across moments."""
        if not moments:
            return 0.0

        total_turns = sum(m.turn_count for m in moments)
        if total_turns == 0:
            return 0.0

        weighted_sum = sum(m.coherence_score * m.turn_count for m in moments)
        return weighted_sum / total_turns

    @staticmethod
    def _calculate_efficiency(
        turn_analysis: List[TurnAnalysis], moments: List[ConversationMoment]
    ) -> float:
        """
        Calculate conversation efficiency.

        Heuristic: Ratio of productive states to total turns.
        """
        if not turn_analysis:
            return 0.0

        productive_states = {
            'Information Gathering',
            'Tool Execution',
            'Solution Proposal',
            'Confirmation',
        }

        productive_turns = sum(1 for t in turn_analysis if t.state in productive_states)

        return productive_turns / len(turn_analysis)

    @staticmethod
    def _calculate_user_experience(turn_analysis: List[TurnAnalysis]) -> float:
        """
        Calculate user experience quality from effort trajectory.

        Lower average effort = better experience.
        """
        effort_scores = [
            t.user_effort_score
            for t in turn_analysis
            if t.user_effort_score is not None
        ]

        if not effort_scores:
            return 0.5  # Neutral default

        avg_effort = np.mean(effort_scores)
        # Invert: low effort = high UX
        return 1.0 - min(1.0, avg_effort)


class ConversationFlowResult(RichBaseModel):
    """Complete results of conversation flow analysis."""

    score_components: FlowScoreComponents
    moments: List[ConversationMoment]
    turn_analysis: List[TurnAnalysis]
    state_progression: List[Dict[str, Any]]
    issues: List[FlowIssue]

    @property
    def final_score(self) -> float:
        return self.score_components.final_score

    def summary(self) -> str:
        """Generate concise summary of analysis."""
        return (
            f"Flow Score: {self.final_score:.3f} | "
            f"{self.score_components.explain()} | "
            f"Issues: {len(self.issues)} detected | "
            f"States: {' → '.join(s['state'] for s in self.state_progression[:5])}"
            f"{'...' if len(self.state_progression) > 5 else ''}"
        )


@metric(
    name='Conversation Flow',
    key='conversation_flow',
    description='Modular conversation flow analysis with coherence, efficiency, and issue detection.',
    required_fields=['conversation'],
    default_threshold=0.7,
    score_range=(0, 1),
    tags=['agent', 'multi_turn'],
)
class ConversationFlow(BaseMetric):
    """
    Refactored conversation flow metric with modular, testable components.

    Improvements over v1:
    - Configurable penalties and thresholds
    - Enum-based issue types (no string matching)
    - Separate, testable detector classes
    - Transparent score decomposition
    - Better statistical methods
    - Comprehensive signal generation
    """

    shares_internal_cache = True

    def __init__(self, config: Optional[FlowConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or FlowConfig()

        # Initialize sub-analyzers (reuse from conversation_utils)
        self.moment_segmenter = MomentSegmenter(**kwargs)
        self.turn_analyzer = TurnAnalyzer(**kwargs)
        self.friction_analyzer = UserFrictionAnalyzer(**kwargs)
        self.coherence_evaluator = MomentCoherenceEvaluator(**kwargs)

        # Initialize new components
        self.issue_detector = IssueDetector(self.config)
        self.score_calculator = FlowScoreCalculator(self.config)

    @trace(name='execute_flow_analysis', capture_args=True, capture_response=True)
    async def execute(
        self, item: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricEvaluationResult:
        """Execute comprehensive conversation flow analysis."""

        if not item.conversation or not item.conversation.messages:
            return MetricEvaluationResult(
                score=np.nan, explanation='No conversation provided.'
            )

        # Segment conversation into moments
        moments = await get_or_compute_moments(item, cache, self.moment_segmenter)

        # Evaluate coherence for each moment
        for moment in moments:
            if moment.coherence_score == 0.0:  # Not yet evaluated
                moment_transcript = self._extract_moment_transcript(item, moment)
                try:
                    from axion.metrics.internals.conversation_utils import (
                        MomentCoherenceInput,
                    )

                    coherence_result = await self.coherence_evaluator.execute(
                        MomentCoherenceInput(moment_transcript=moment_transcript)
                    )
                    moment.coherence_score = coherence_result.coherence_score
                except Exception as e:
                    logger.warning(
                        f'Coherence evaluation failed for moment {moment.id}: {e}'
                    )
                    moment.coherence_score = 0.5

        # Analyze turns
        turn_analysis = await get_or_compute_turn_analysis(
            item, cache, self.turn_analyzer, self.friction_analyzer
        )

        # Build state progression
        state_progression = self._build_state_progression(turn_analysis)

        # Detect issues
        issues = self.issue_detector.detect_all(
            moments, turn_analysis, state_progression
        )

        # Calculate score
        score_components = self.score_calculator.calculate(
            moments, turn_analysis, issues
        )

        # Build result
        result = ConversationFlowResult(
            score_components=score_components,
            moments=moments,
            turn_analysis=turn_analysis,
            state_progression=state_progression,
            issues=issues,
        )

        # Compute cost estimate
        self.compute_cost_estimate(
            [
                self.moment_segmenter,
                self.turn_analyzer,
                self.friction_analyzer,
                self.coherence_evaluator,
            ]
        )

        return MetricEvaluationResult(
            score=result.final_score, explanation=result.summary(), signals=result
        )

    @staticmethod
    def _extract_moment_transcript(
        item: DatasetItem, moment: ConversationMoment
    ) -> str:
        """Extract transcript for a specific moment."""
        lines = []
        for i in range(moment.start_turn, moment.end_turn + 1):
            if i >= len(item.conversation.messages):
                break
            msg = item.conversation.messages[i]
            role = 'User' if isinstance(msg, HumanMessage) else 'Agent'
            lines.append(f"{role}: {msg.content or ''}")
        return '\n'.join(lines)

    @staticmethod
    def _build_state_progression(
        turn_analysis: List[TurnAnalysis],
    ) -> List[Dict[str, Any]]:
        """Build state progression summary from turn analysis."""
        if not turn_analysis:
            return []

        progression = []
        current_state = turn_analysis[0].state
        start_turn = 0

        for i, turn in enumerate(turn_analysis):
            if turn.state != current_state:
                progression.append(
                    {
                        'state': current_state,
                        'start_turn': start_turn,
                        'end_turn': i - 1,
                        'duration': i - start_turn,
                    }
                )
                current_state = turn.state
                start_turn = i

        # Add final state
        progression.append(
            {
                'state': current_state,
                'start_turn': start_turn,
                'end_turn': len(turn_analysis) - 1,
                'duration': len(turn_analysis) - start_turn,
            }
        )

        return progression

    def get_signals(self, result: ConversationFlowResult) -> List[SignalDescriptor]:
        """Generate comprehensive signals showing score calculation."""

        signals = [
            # === HEADLINE: Final Score ===
            SignalDescriptor(
                name='final_score',
                extractor=lambda r: r.score_components.final_score,
                headline_display=True,
                description='Overall conversation flow quality score',
            ),
            # === SCORE COMPONENTS ===
            SignalDescriptor(
                name='base_score',
                group='Score Breakdown',
                extractor=lambda r: r.score_components.base_score,
                description='Score before penalties applied',
            ),
            SignalDescriptor(
                name='coherence',
                group='Score Breakdown',
                extractor=lambda r: r.score_components.coherence,
                headline_display=True,
                description='Conversation coherence across moments',
            ),
            SignalDescriptor(
                name='efficiency',
                group='Score Breakdown',
                extractor=lambda r: r.score_components.efficiency,
                description='Ratio of productive conversation turns',
            ),
            SignalDescriptor(
                name='user_experience',
                group='Score Breakdown',
                extractor=lambda r: r.score_components.user_experience,
                description='User experience quality (inverse of effort)',
            ),
            # === PENALTIES ===
            SignalDescriptor(
                name='total_penalty',
                group='Score Breakdown',
                extractor=lambda r: r.score_components.capped_penalty,
                description='Total penalty applied (after cap)',
            ),
            SignalDescriptor(
                name='raw_penalty',
                group='Score Breakdown',
                extractor=lambda r: r.score_components.raw_penalty,
                description='Sum of all issue penalties (before cap)',
            ),
            # === AGGREGATE METRICS ===
            SignalDescriptor(
                name='total_moments',
                group='Conversation Structure',
                extractor=lambda r: len(r.moments),
                description='Number of topical moments',
            ),
            SignalDescriptor(
                name='total_turns',
                group='Conversation Structure',
                extractor=lambda r: len(r.turn_analysis),
                description='Total conversation turns',
            ),
            SignalDescriptor(
                name='state_changes',
                group='Conversation Structure',
                extractor=lambda r: len(r.state_progression),
                description='Number of state transitions',
            ),
            # === ISSUES SUMMARY ===
            SignalDescriptor(
                name='total_issues',
                group='Issues Detected',
                extractor=lambda r: len(r.issues),
                headline_display=True,
                description='Total number of flow issues detected',
            ),
            SignalDescriptor(
                name='state_loops',
                group='Issues Detected',
                extractor=lambda r: sum(
                    1 for i in r.issues if i.type == IssueType.STATE_LOOP
                ),
                description='Number of state loop issues',
            ),
            SignalDescriptor(
                name='effort_degradations',
                group='Issues Detected',
                extractor=lambda r: sum(
                    1 for i in r.issues if i.type == IssueType.EFFORT_DEGRADATION
                ),
                description='Number of user effort degradation issues',
            ),
            SignalDescriptor(
                name='correction_cascades',
                group='Issues Detected',
                extractor=lambda r: sum(
                    1 for i in r.issues if i.type == IssueType.CORRECTION_CASCADE
                ),
                description='Number of correction cascade issues',
            ),
            SignalDescriptor(
                name='coherence_drops',
                group='Issues Detected',
                extractor=lambda r: sum(
                    1 for i in r.issues if i.type == IssueType.COHERENCE_DROP
                ),
                description='Number of coherence drop issues',
            ),
        ]

        # === PER-ISSUE SIGNALS ===
        for i, issue in enumerate(result.issues):
            group_name = f'Issue {i + 1}: {issue.type.value}'

            signals.extend(
                [
                    SignalDescriptor(
                        name='type',
                        group=group_name,
                        extractor=lambda r, idx=i: r.issues[idx].type.value,
                        headline_display=True,
                        description='Issue type',
                    ),
                    SignalDescriptor(
                        name='severity',
                        group=group_name,
                        extractor=lambda r, idx=i: r.issues[idx].severity,
                        description='Severity score (0-1)',
                    ),
                    SignalDescriptor(
                        name='penalty_contribution',
                        group=group_name,
                        extractor=lambda r, idx=i: r.issues[idx].get_penalty(
                            self.config.penalty_weights
                        ),
                        description='Penalty contribution to final score',
                    ),
                    SignalDescriptor(
                        name='description',
                        group=group_name,
                        extractor=lambda r, idx=i: r.issues[idx].description,
                        description='Issue description',
                    ),
                    SignalDescriptor(
                        name='turn_location',
                        group=group_name,
                        extractor=lambda r, idx=i: (
                            f'Turn {r.issues[idx].turn_index}'
                            if r.issues[idx].turn_index is not None
                            else f'Turns {r.issues[idx].turn_range[0]}-{r.issues[idx].turn_range[1]}'
                            if r.issues[idx].turn_range
                            else 'N/A'
                        ),
                        description='Where the issue occurs',
                    ),
                ]
            )

        # === PER-MOMENT SIGNALS ===
        for i, moment in enumerate(result.moments):
            moment_id = moment.id if moment.id is not None else i
            group_name = f"Moment {moment_id}: {moment.topic[:40]}{'...' if len(moment.topic) > 40 else ''}"

            signals.extend(
                [
                    SignalDescriptor(
                        name='coherence_score',
                        group=group_name,
                        extractor=lambda r, idx=i: r.moments[idx].coherence_score,
                        headline_display=True,
                        description='Coherence within this moment',
                    ),
                    SignalDescriptor(
                        name='turn_count',
                        group=group_name,
                        extractor=lambda r, idx=i: r.moments[idx].turn_count,
                        description='Number of turns',
                    ),
                    SignalDescriptor(
                        name='weight',
                        group=group_name,
                        extractor=lambda r, idx=i: (
                            r.moments[idx].turn_count
                            / sum(m.turn_count for m in r.moments)
                        ),
                        description='Weight in coherence calculation',
                    ),
                    SignalDescriptor(
                        name='contribution',
                        group=group_name,
                        extractor=lambda r, idx=i: (
                            r.moments[idx].coherence_score
                            * r.moments[idx].turn_count
                            / sum(m.turn_count for m in r.moments)
                        ),
                        description='Contribution to overall coherence',
                    ),
                    SignalDescriptor(
                        name='turn_range',
                        group=group_name,
                        extractor=lambda r,
                        idx=i: f'{r.moments[idx].start_turn}-{r.moments[idx].end_turn}',
                        description='Turn indices for this moment',
                    ),
                ]
            )

        # === STATE PROGRESSION ===
        for i, state_seg in enumerate(result.state_progression):
            group_name = f"State {i + 1}: {state_seg['state']}"

            signals.extend(
                [
                    SignalDescriptor(
                        name='state',
                        group=group_name,
                        extractor=lambda r, idx=i: r.state_progression[idx]['state'],
                        headline_display=True,
                        description='Conversation state',
                    ),
                    SignalDescriptor(
                        name='duration',
                        group=group_name,
                        extractor=lambda r, idx=i: r.state_progression[idx]['duration'],
                        description='Number of turns in this state',
                    ),
                    SignalDescriptor(
                        name='turn_range',
                        group=group_name,
                        extractor=lambda r, idx=i: (
                            f"{r.state_progression[idx]['start_turn']}-{r.state_progression[idx]['end_turn']}"
                        ),
                        description='Turn range for this state',
                    ),
                ]
            )

        # === SAMPLE TURN ANALYSIS (first 5, last 3) ===
        turns_to_show = list(range(min(5, len(result.turn_analysis)))) + list(
            range(max(5, len(result.turn_analysis) - 3), len(result.turn_analysis))
        )
        turns_to_show = sorted(set(turns_to_show))

        for turn_idx in turns_to_show:
            if turn_idx >= len(result.turn_analysis):
                continue

            turn = result.turn_analysis[turn_idx]
            group_name = f'Turn {turn.turn_index} ({turn.role})'

            signals.extend(
                [
                    SignalDescriptor(
                        name='state',
                        group=group_name,
                        extractor=lambda r, idx=turn_idx: r.turn_analysis[idx].state,
                        headline_display=True,
                        description='Conversational state',
                    ),
                    SignalDescriptor(
                        name='dialogue_act',
                        group=group_name,
                        extractor=lambda r, idx=turn_idx: r.turn_analysis[
                            idx
                        ].dialogue_act,
                        description='Dialogue act performed',
                    ),
                    SignalDescriptor(
                        name='user_effort',
                        group=group_name,
                        extractor=lambda r, idx=turn_idx: (
                            f'{r.turn_analysis[idx].user_effort_score:.3f}'
                            if r.turn_analysis[idx].user_effort_score is not None
                            else 'N/A (agent turn)'
                        ),
                        description='User effort score',
                    ),
                ]
            )

        return signals


class EnhancedUserEffortCalculator:
    """
    Optional enhanced user effort calculator with multiple signals.

    This can replace or supplement the simple length+friction calculation
    from the original implementation.
    """

    def __init__(self, config: FlowConfig):
        self.config = config

    def calculate(
        self,
        message_content: str,
        friction_score: float,
        previous_messages: Optional[List[str]] = None,
    ) -> float:
        """
        Calculate comprehensive user effort score.

        Args:
            message_content: The user's message text
            friction_score: Friction score from LLM analysis
            previous_messages: Recent user messages for repetition detection

        Returns:
            Effort score between 0 and 1
        """
        length_score = self._calculate_length_score(message_content, previous_messages)

        repetition_score = self._calculate_repetition_score(
            message_content, previous_messages
        )

        # Weighted combination
        weights = self.config.effort_weights
        effort = (
            weights['friction'] * friction_score
            + weights['length'] * length_score
            + weights['repetition'] * repetition_score
        )

        return min(1.0, effort)

    @staticmethod
    def _calculate_length_score(
        message: str, previous_messages: Optional[List[str]] = None
    ) -> float:
        """
        Calculate length-based effort score, normalized by conversation context.

        If we have previous messages, normalize relative to their median length.
        Otherwise, use absolute normalization.
        """
        msg_length = len(message)

        if previous_messages and len(previous_messages) > 0:
            # Normalize relative to conversation median
            prev_lengths = [len(m) for m in previous_messages]
            median_length = np.median(prev_lengths)

            if median_length > 0:
                # Score increases if message is longer than typical
                relative_length = msg_length / median_length
                return min(1.0, relative_length / 2.0)  # 2x median = score of 1.0

        # Fallback: absolute normalization
        return min(1.0, msg_length / 400.0)

    @staticmethod
    def _calculate_repetition_score(
        message: str, previous_messages: Optional[List[str]] = None
    ) -> float:
        """
        Calculate repetition score based on overlap with previous messages.

        High repetition suggests user is having to repeat themselves.
        """
        if not previous_messages or len(previous_messages) == 0:
            return 0.0

        # Simple word-level overlap check
        current_words = set(message.lower().split())

        if len(current_words) == 0:
            return 0.0

        # Check overlap with recent messages (last 2-3)
        recent_messages = previous_messages[-3:]

        max_overlap = 0.0
        for prev_msg in recent_messages:
            prev_words = set(prev_msg.lower().split())
            if len(prev_words) == 0:
                continue

            overlap = len(current_words & prev_words)
            overlap_ratio = overlap / len(current_words)
            max_overlap = max(max_overlap, overlap_ratio)

        # High overlap = high repetition = high effort
        return max_overlap


class FlowConfigPresets:
    """Predefined configurations for different use cases."""

    @staticmethod
    def strict() -> FlowConfig:
        """Strict configuration with high penalties for issues."""
        return FlowConfig(
            penalty_weights={
                'state_loop': 0.35,
                'effort_degradation': 0.40,
                'correction_cascade': 0.30,
                'coherence_drop': 0.20,
            },
            max_total_penalty=0.60,
            degradation_threshold=0.10,
            cascade_threshold=2,
        )

    @staticmethod
    def lenient() -> FlowConfig:
        """Lenient configuration with lower penalties."""
        return FlowConfig(
            penalty_weights={
                'state_loop': 0.15,
                'effort_degradation': 0.20,
                'correction_cascade': 0.15,
                'coherence_drop': 0.10,
            },
            max_total_penalty=0.35,
            degradation_threshold=0.25,
            cascade_threshold=3,
        )

    @staticmethod
    def balanced() -> FlowConfig:
        """Balanced configuration (default)."""
        return FlowConfig()  # Uses defaults

    @staticmethod
    def research_mode() -> FlowConfig:
        """Configuration optimized for research/analysis with detailed detection."""
        return FlowConfig(
            loop_window_sizes=[2, 3, 4, 5],
            cascade_window_size=5,
            effort_window_size=4,
            min_turns_for_degradation=4,
        )
