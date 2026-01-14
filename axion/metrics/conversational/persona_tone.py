import statistics
from typing import Any, Dict, List, Literal, Optional, Union

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
    MomentSegmenter,
    get_or_compute_moments,
)
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

logger = get_logger(__name__)


class PersonaDefinition(RichBaseModel):
    """
    Defines the target persona/tone the agent should maintain.

    This can be provided in additional_input or configured in the metric.
    """

    description: str = Field(
        description="Overall description of the persona (e.g., 'friendly and professional', 'witty and creative')"
    )

    key_characteristics: List[str] = Field(
        default_factory=list,
        description="Specific traits to exhibit (e.g., ['empathetic', 'concise', 'uses analogies'])",
    )

    tone_indicators: List[str] = Field(
        default_factory=list,
        description="Specific tone markers (e.g., ['warm', 'formal', 'enthusiastic'])",
    )

    example_phrases: List[str] = Field(
        default_factory=list, description='Example phrases that exemplify the persona'
    )

    anti_patterns: List[str] = Field(
        default_factory=list,
        description="Behaviors to avoid (e.g., ['overly casual', 'technical jargon', 'impatience'])",
    )


class TurnPersonaAnalysisInput(RichBaseModel):
    """Input for analyzing a single agent turn's persona adherence."""

    persona_definition: PersonaDefinition
    agent_turn_content: str
    conversation_context: str = Field(description='Previous turns for context')
    turn_index: int


class TurnPersonaAnalysisOutput(RichBaseModel):
    """Output from analyzing a single turn's persona adherence."""

    adherence_score: float = Field(
        description='Score from 0.0 (complete break) to 1.0 (perfect adherence)'
    )

    tone_classification: str = Field(
        description="The actual tone detected in this turn (e.g., 'professional', 'casual', 'frustrated')"
    )

    persona_match: bool = Field(
        description='Whether the turn matches the target persona'
    )

    positive_indicators: List[str] = Field(
        default_factory=list, description='Specific elements that matched the persona'
    )

    negative_indicators: List[str] = Field(
        default_factory=list, description='Specific elements that violated the persona'
    )

    deviation_type: Optional[
        Literal['tone_shift', 'characteristic_break', 'anti_pattern', 'none']
    ] = Field(default=None, description='Type of persona deviation detected, if any')


class TurnPersonaAnalyzer(
    BaseMetric[TurnPersonaAnalysisInput, TurnPersonaAnalysisOutput]
):
    """Analyzes a single agent turn for persona/tone adherence."""

    as_structured_llm = True

    instruction = """You are an expert at analyzing conversational tone and persona consistency.

Analyze the agent's turn against the defined persona/tone criteria.

**Target Persona:**
- Description: {persona_description}
- Key Characteristics: {key_characteristics}
- Tone Indicators: {tone_indicators}
- Anti-Patterns to Avoid: {anti_patterns}

**Your Task:**
1. **Assess Adherence**: Score how well this turn matches the target persona (0.0 = complete break, 1.0 = perfect match)
2. **Classify Tone**: Identify the actual tone/style used in this turn
3. **Identify Match**: Does this turn align with the persona? (true/false)
4. **Find Positive Indicators**: List specific phrases/choices that matched the persona
5. **Find Negative Indicators**: List specific phrases/choices that violated the persona
6. **Detect Deviation Type**: If there's a break, classify it as:
   - "tone_shift": Change in formality, warmth, or style
   - "characteristic_break": Missing expected trait or behavior
   - "anti_pattern": Exhibited a behavior that should be avoided
   - "none": No deviation detected

Consider the conversation context to understand if the tone/persona is appropriate for the situation."""

    input_model = TurnPersonaAnalysisInput
    output_model = TurnPersonaAnalysisOutput

    examples = [
        (
            TurnPersonaAnalysisInput(
                persona_definition=PersonaDefinition(
                    description='friendly and professional',
                    key_characteristics=['empathetic', 'clear', 'solution-focused'],
                    tone_indicators=['warm', 'respectful'],
                    anti_patterns=['overly casual', 'dismissive'],
                ),
                agent_turn_content='I understand this is frustrating. Let me help you resolve this issue right away.',
                conversation_context="User: I've been trying to fix this for hours!\nAgent: ",
                turn_index=1,
            ),
            TurnPersonaAnalysisOutput(
                adherence_score=1.0,
                tone_classification='empathetic and professional',
                persona_match=True,
                positive_indicators=[
                    'Acknowledges user frustration (empathetic)',
                    'Offers immediate help (solution-focused)',
                    'Respectful tone throughout',
                ],
                negative_indicators=[],
                deviation_type='none',
            ),
        ),
        (
            TurnPersonaAnalysisInput(
                persona_definition=PersonaDefinition(
                    description='friendly and professional',
                    key_characteristics=['empathetic', 'clear', 'solution-focused'],
                    tone_indicators=['warm', 'respectful'],
                    anti_patterns=['overly casual', 'dismissive'],
                ),
                agent_turn_content="Yeah, just click the button. It's not that hard.",
                conversation_context="User: I can't figure out how to submit this form.\nAgent: ",
                turn_index=1,
            ),
            TurnPersonaAnalysisOutput(
                adherence_score=0.2,
                tone_classification='dismissive and casual',
                persona_match=False,
                positive_indicators=[],
                negative_indicators=[
                    "Overly casual language ('Yeah')",
                    "Dismissive tone ('It's not that hard')",
                    "Lacks empathy for user's struggle",
                ],
                deviation_type='anti_pattern',
            ),
        ),
    ]

    def _build_instruction_with_persona(self, persona: PersonaDefinition) -> str:
        """Build instruction with persona details injected."""
        return self.instruction.format(
            persona_description=persona.description,
            key_characteristics=', '.join(persona.key_characteristics)
            if persona.key_characteristics
            else 'None specified',
            tone_indicators=', '.join(persona.tone_indicators)
            if persona.tone_indicators
            else 'None specified',
            anti_patterns=', '.join(persona.anti_patterns)
            if persona.anti_patterns
            else 'None specified',
        )

    async def execute(
        self, input_data: TurnPersonaAnalysisInput
    ) -> TurnPersonaAnalysisOutput:
        """Execute with persona-specific instruction."""
        # Temporarily update instruction with persona details
        original_instruction = self.instruction
        self.instruction = self._build_instruction_with_persona(
            input_data.persona_definition
        )

        try:
            result = await super().execute(input_data)
            return result
        finally:
            self.instruction = original_instruction


class PersonaConsistencyAnalysisInput(RichBaseModel):
    """Input for analyzing persona consistency across the conversation."""

    persona_definition: PersonaDefinition
    turn_analyses: List[TurnPersonaAnalysisOutput]
    conversation_moments: List[ConversationMoment]


class PersonaConsistencyAnalysisOutput(RichBaseModel):
    """Output from consistency analysis."""

    consistency_score: float = Field(
        description='Variance-based score: 1.0 = perfectly consistent, 0.0 = highly inconsistent'
    )

    tone_drift_detected: bool = Field(
        description='Whether significant tone drift was detected over time'
    )

    drift_direction: Optional[str] = Field(
        default=None,
        description="Direction of drift if detected (e.g., 'increasingly casual', 'becoming more formal')",
    )

    most_consistent_moments: List[int] = Field(
        default_factory=list, description='IDs of moments with best persona adherence'
    )

    least_consistent_moments: List[int] = Field(
        default_factory=list, description='IDs of moments with worst persona adherence'
    )


class PersonaConsistencyAnalyzer(
    BaseMetric[PersonaConsistencyAnalysisInput, PersonaConsistencyAnalysisOutput]
):
    """Analyzes persona consistency patterns across the conversation."""

    as_structured_llm = True

    instruction = """Analyze the pattern of persona adherence across the conversation.

You are given:
1. A list of turn-by-turn adherence scores and analyses
2. Conversation moments (topical segments)

**Your Task:**
1. **Calculate Consistency Score**: Based on the variance in adherence scores, rate overall consistency
   - 1.0 = perfectly consistent (low variance)
   - 0.0 = highly inconsistent (high variance)

2. **Detect Tone Drift**: Determine if there's a directional trend (true/false)
   - Look at scores over time - do they progressively increase, decrease, or fluctuate?

3. **Identify Drift Direction**: If drift exists, describe it
   - Examples: "increasingly casual", "becoming more formal", "degrading over time"
   - Set to null if no drift detected

4. **Find Best/Worst Moments**: Identify moment IDs with strongest/weakest adherence
   - Use the moment IDs from the conversation_moments list

Look for patterns like:
- Progressive degradation (starts strong, weakens over time)
- Situational breaks (persona breaks under stress/complexity)
- Recovery patterns (breaks then recovers)

Return all required fields: consistency_score, tone_drift_detected, drift_direction, most_consistent_moments, least_consistent_moments"""

    input_model = PersonaConsistencyAnalysisInput
    output_model = PersonaConsistencyAnalysisOutput

    examples = [
        (
            PersonaConsistencyAnalysisInput(
                persona_definition=PersonaDefinition(
                    description='friendly and professional',
                    key_characteristics=['empathetic', 'clear'],
                    tone_indicators=['warm', 'respectful'],
                    anti_patterns=['dismissive'],
                ),
                turn_analyses=[
                    TurnPersonaAnalysisOutput(
                        adherence_score=0.9,
                        tone_classification='professional',
                        persona_match=True,
                        positive_indicators=['warm greeting'],
                        negative_indicators=[],
                        deviation_type='none',
                    ),
                    TurnPersonaAnalysisOutput(
                        adherence_score=0.85,
                        tone_classification='professional',
                        persona_match=True,
                        positive_indicators=['clear explanation'],
                        negative_indicators=[],
                        deviation_type='none',
                    ),
                    TurnPersonaAnalysisOutput(
                        adherence_score=0.88,
                        tone_classification='professional',
                        persona_match=True,
                        positive_indicators=['empathetic response'],
                        negative_indicators=[],
                        deviation_type='none',
                    ),
                ],
                conversation_moments=[
                    ConversationMoment(
                        id=1, topic='Greeting', start_turn=0, end_turn=2, turn_count=3
                    )
                ],
            ),
            PersonaConsistencyAnalysisOutput(
                consistency_score=0.95,
                tone_drift_detected=False,
                drift_direction=None,
                most_consistent_moments=[1],
                least_consistent_moments=[],
            ),
        ),
        (
            PersonaConsistencyAnalysisInput(
                persona_definition=PersonaDefinition(
                    description='formal and respectful',
                    key_characteristics=['professional'],
                    tone_indicators=['formal'],
                    anti_patterns=['casual'],
                ),
                turn_analyses=[
                    TurnPersonaAnalysisOutput(
                        adherence_score=0.9,
                        tone_classification='formal',
                        persona_match=True,
                        positive_indicators=['professional tone'],
                        negative_indicators=[],
                        deviation_type='none',
                    ),
                    TurnPersonaAnalysisOutput(
                        adherence_score=0.7,
                        tone_classification='somewhat casual',
                        persona_match=False,
                        positive_indicators=[],
                        negative_indicators=['casual language'],
                        deviation_type='tone_shift',
                    ),
                    TurnPersonaAnalysisOutput(
                        adherence_score=0.5,
                        tone_classification='very casual',
                        persona_match=False,
                        positive_indicators=[],
                        negative_indicators=['slang used'],
                        deviation_type='tone_shift',
                    ),
                ],
                conversation_moments=[
                    ConversationMoment(
                        id=1,
                        topic='Initial Query',
                        start_turn=0,
                        end_turn=1,
                        turn_count=2,
                    ),
                    ConversationMoment(
                        id=2, topic='Follow-up', start_turn=2, end_turn=2, turn_count=1
                    ),
                ],
            ),
            PersonaConsistencyAnalysisOutput(
                consistency_score=0.3,
                tone_drift_detected=True,
                drift_direction='increasingly casual',
                most_consistent_moments=[1],
                least_consistent_moments=[2],
            ),
        ),
    ]


class PersonaToneAdherenceResult(RichBaseModel):
    """Complete result for persona & tone adherence metric."""

    # Target persona
    target_persona: PersonaDefinition

    # Overall scores
    overall_adherence_score: float = Field(
        description='Average adherence across all agent turns'
    )
    consistency_score: float = Field(
        description='How consistent the persona was maintained'
    )
    final_composite_score: float = Field(
        description='Weighted combination of adherence and consistency'
    )

    # Turn-by-turn analysis
    turn_analyses: List[TurnPersonaAnalysisOutput]

    # Consistency analysis
    tone_drift_detected: bool
    drift_direction: Optional[str]

    # Diagnostics
    persona_breaks: List[Dict[str, Any]] = Field(
        default_factory=list, description='Turns where persona was broken'
    )
    adherence_variance: float = Field(
        description='Variance in adherence scores (lower is better)'
    )
    worst_turn_index: Optional[int] = Field(
        default=None, description='Turn with lowest adherence score'
    )

    # Context
    conversation_moments: List[ConversationMoment]


@metric(
    name='Persona & Tone Adherence',
    key='persona_tone_adherence',
    description='Evaluates whether the agent maintains its intended persona and tone consistently',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['agent', 'multi_turn'],
)
class PersonaToneAdherence(BaseMetric):
    """
    Persona & Tone Adherence Metric

    Evaluates whether the agent maintains its intended persona and tone consistently
    throughout the conversation. Detects persona breaks, tone shifts, and provides
    turn-by-turn adherence scores.

    This metric reuses conversation analysis components from shared_conversation_analysis
    to minimize LLM calls and ensure consistency across metrics.
    """

    shares_internal_cache = True

    def __init__(
        self,
        persona_key: str = 'persona',
        persona: Optional[Union[PersonaDefinition, Dict[str, Any]]] = None,
        adherence_weight: float = 0.7,
        consistency_weight: float = 0.3,
        persona_break_threshold: float = 0.6,
        drift_detection_threshold: float = 0.15,
        analyze_only_agent_turns: bool = True,
        **kwargs,
    ):
        """
        Initialize the Persona & Tone Adherence metric.

        Args:
            persona_key: Key in additional_input containing persona definition (default: 'persona')
            persona: Global persona if not provided in input

            # Score weights
            adherence_weight: Weight for average adherence score (default: 0.7)
                Rationale: Actual adherence to persona is more important than consistency
            consistency_weight: Weight for consistency score (default: 0.3)
                Rationale: Consistency matters but secondary to actual adherence

            # Detection thresholds
            persona_break_threshold: Score below which a turn is flagged as persona break (default: 0.6)
                Rationale: Scores below 60% indicate significant deviation
            drift_detection_threshold: Variance threshold for detecting tone drift (default: 0.15)
                Rationale: Variance >0.15 indicates inconsistent persona

            # Analysis scope
            analyze_only_agent_turns: If True, only analyze agent turns (default: True)
                Rationale: Persona adherence is only relevant for agent responses
        """
        super().__init__(**kwargs)

        # Validate weights
        if not abs((adherence_weight + consistency_weight) - 1.0) < 0.001:
            raise ValueError(
                f'adherence_weight ({adherence_weight}) and consistency_weight ({consistency_weight}) '
                f'must sum to 1.0, got {adherence_weight + consistency_weight}'
            )

        # Validate thresholds
        if not 0 <= persona_break_threshold <= 1:
            raise ValueError(
                f'persona_break_threshold must be between 0 and 1, got {persona_break_threshold}'
            )

        if not 0 <= drift_detection_threshold <= 1:
            raise ValueError(
                f'drift_detection_threshold must be between 0 and 1, got {drift_detection_threshold}'
            )

        self.persona_key = persona_key
        self.persona = persona or PersonaDefinition(
            description='helpful, professional, and friendly',
            key_characteristics=['clear', 'empathetic', 'solution-focused'],
            tone_indicators=['respectful', 'warm'],
            anti_patterns=['dismissive', 'overly technical', 'impatient'],
        )
        if isinstance(self.persona, dict):
            self.persona = PersonaDefinition(**self.persona)

        self.adherence_weight = adherence_weight
        self.consistency_weight = consistency_weight
        self.persona_break_threshold = persona_break_threshold
        self.drift_detection_threshold = drift_detection_threshold
        self.analyze_only_agent_turns = analyze_only_agent_turns

        # Shared components
        self.moment_segmenter = MomentSegmenter(**kwargs)

        # Persona-specific components
        self.turn_persona_analyzer = TurnPersonaAnalyzer(**kwargs)
        self.consistency_analyzer = PersonaConsistencyAnalyzer(**kwargs)

    def _extract_persona(self, item: DatasetItem) -> PersonaDefinition:
        """Extract persona definition from item or use default."""
        if item.additional_input and self.persona_key in item.additional_input:
            persona_data = item.additional_input[self.persona_key]

            # If it's already a PersonaDefinition, use it
            if isinstance(persona_data, PersonaDefinition):
                return persona_data

            # If it's a dict, convert it
            if isinstance(persona_data, dict):
                return PersonaDefinition(**persona_data)

            # If it's a string, create a simple persona
            if isinstance(persona_data, str):
                return PersonaDefinition(description=persona_data)

        # Use default
        logger.debug(
            f"No persona found in additional_input['{self.persona_key}'], using passed persona"
        )
        return self.persona

    @trace(name='PersonaTone.execute', capture_args=True, capture_response=True)
    async def execute(
        self, item: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricEvaluationResult:
        """Execute persona & tone adherence analysis."""

        if not item.conversation or not item.conversation.messages:
            return MetricEvaluationResult(
                score=np.nan, explanation='No conversation provided.'
            )

        # Extract target persona
        target_persona = self._extract_persona(item)

        moments = await get_or_compute_moments(item, cache, self.moment_segmenter)

        turn_analyses = []
        agent_turn_indices = []

        for i, msg in enumerate(item.conversation.messages):
            if self.analyze_only_agent_turns and isinstance(msg, HumanMessage):
                continue
            if isinstance(msg, HumanMessage):
                continue

            # Build context (all previous turns)
            context = '\n'.join(
                [
                    f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content or ''}"
                    for m in item.conversation.messages[:i]
                ]
            )

            try:
                analysis_input = TurnPersonaAnalysisInput(
                    persona_definition=target_persona,
                    agent_turn_content=msg.content or '',
                    conversation_context=context,
                    turn_index=i,
                )

                turn_result = await self.turn_persona_analyzer.execute(analysis_input)
                turn_analyses.append(turn_result)
                agent_turn_indices.append(i)

            except Exception as e:
                logger.warning(f'Turn persona analysis failed at turn {i}: {e}')
                # Add fallback analysis -- TODO
                turn_analyses.append(
                    TurnPersonaAnalysisOutput(
                        adherence_score=0.5,
                        tone_classification='unknown',
                        persona_match=True,
                        positive_indicators=[],
                        negative_indicators=[f'Analysis failed: {str(e)}'],
                        deviation_type='none',
                    )
                )
                agent_turn_indices.append(i)

        if not turn_analyses:
            return MetricEvaluationResult(
                score=0.0, explanation='No agent turns to analyze.'
            )

        try:
            consistency_input = PersonaConsistencyAnalysisInput(
                persona_definition=target_persona,
                turn_analyses=turn_analyses,
                conversation_moments=moments,
            )
            consistency_result = await self.consistency_analyzer.execute(
                consistency_input
            )
        except Exception as e:
            logger.warning(f'Consistency analysis failed: {e}')
            # Fallback: compute variance manually
            scores = [t.adherence_score for t in turn_analyses]
            variance = statistics.variance(scores) if len(scores) > 1 else 0.0
            consistency_result = PersonaConsistencyAnalysisOutput(
                consistency_score=max(0.0, 1.0 - variance),
                tone_drift_detected=variance > self.drift_detection_threshold,
                drift_direction=None,
                most_consistent_moments=[],
                least_consistent_moments=[],
            )

        overall_adherence_score = statistics.mean(
            t.adherence_score for t in turn_analyses
        )
        consistency_score = consistency_result.consistency_score
        final_composite_score = (
            self.adherence_weight * overall_adherence_score
            + self.consistency_weight * consistency_score
        )

        # Detect persona breaks
        persona_breaks = [
            {
                'turn_index': agent_turn_indices[i],
                'adherence_score': t.adherence_score,
                'tone_classification': t.tone_classification,
                'negative_indicators': t.negative_indicators,
                'deviation_type': t.deviation_type,
            }
            for i, t in enumerate(turn_analyses)
            if t.adherence_score < self.persona_break_threshold
        ]

        # Calculate variance
        scores = [t.adherence_score for t in turn_analyses]
        adherence_variance = statistics.variance(scores) if len(scores) > 1 else 0.0

        # Find worst turn
        worst_turn_index = None
        if turn_analyses:
            min_score_idx = min(
                range(len(turn_analyses)),
                key=lambda i: turn_analyses[i].adherence_score,
            )
            worst_turn_index = agent_turn_indices[min_score_idx]

        # === Build result ===
        result_data = PersonaToneAdherenceResult(
            target_persona=target_persona,
            overall_adherence_score=overall_adherence_score,
            consistency_score=consistency_score,
            final_composite_score=final_composite_score,
            turn_analyses=turn_analyses,
            tone_drift_detected=consistency_result.tone_drift_detected,
            drift_direction=consistency_result.drift_direction,
            persona_breaks=persona_breaks,
            adherence_variance=adherence_variance,
            worst_turn_index=worst_turn_index,
            conversation_moments=moments,
        )

        # Explanation
        explanation = (
            f"Persona '{target_persona.description}' adherence: {overall_adherence_score:.2f}, "
            f'consistency: {consistency_score:.2f}, final: {final_composite_score:.2f}. '
            f'{len(persona_breaks)} persona break(s) detected'
            + (
                f', tone drift: {consistency_result.drift_direction}'
                if consistency_result.tone_drift_detected
                else ''
            )
        )

        self.compute_cost_estimate(
            [
                self.moment_segmenter,
                self.turn_persona_analyzer,
                self.consistency_analyzer,
            ]
        )

        return MetricEvaluationResult(
            score=final_composite_score, explanation=explanation, signals=result_data
        )

    def get_signals(
        self, result: PersonaToneAdherenceResult
    ) -> List[SignalDescriptor[PersonaToneAdherenceResult]]:
        """Generate comprehensive signals for persona adherence analysis."""

        final_score = (
            self.last_result.score
            if hasattr(self, 'last_result') and self.last_result
            else 0.0
        )

        signals = [
            # === HEADLINE ===
            SignalDescriptor(
                name='final_composite_score',
                extractor=lambda r: final_score,
                headline_display=True,
                description=f'Weighted score: {self.adherence_weight}×adherence + {self.consistency_weight}×consistency',
            ),
            # === SCORE COMPONENTS ===
            SignalDescriptor(
                name='overall_adherence_score',
                extractor=lambda r: r.overall_adherence_score,
                description='Average persona adherence across all agent turns',
            ),
            SignalDescriptor(
                name='consistency_score',
                extractor=lambda r: r.consistency_score,
                description='How consistently persona was maintained (1.0 = no variance)',
            ),
            SignalDescriptor(
                name='adherence_contribution',
                extractor=lambda r: r.overall_adherence_score * self.adherence_weight,
                description=f'Adherence contribution to final score (weight={self.adherence_weight})',
            ),
            SignalDescriptor(
                name='consistency_contribution',
                extractor=lambda r: r.consistency_score * self.consistency_weight,
                description=f'Consistency contribution to final score (weight={self.consistency_weight})',
            ),
            # === CONFIGURATION ===
            SignalDescriptor(
                name='config_adherence_weight',
                extractor=lambda r: self.adherence_weight,
                description='Configured weight for adherence score',
            ),
            SignalDescriptor(
                name='config_consistency_weight',
                extractor=lambda r: self.consistency_weight,
                description='Configured weight for consistency score',
            ),
            SignalDescriptor(
                name='config_break_threshold',
                extractor=lambda r: self.persona_break_threshold,
                description='Configured threshold for persona break detection',
            ),
            # === TARGET PERSONA ===
            SignalDescriptor(
                name='target_persona_description',
                extractor=lambda r: r.target_persona.description,
                description='The target persona/tone being evaluated',
            ),
            SignalDescriptor(
                name='target_characteristics',
                extractor=lambda r: ', '.join(r.target_persona.key_characteristics)
                if r.target_persona.key_characteristics
                else 'None',
                description='Expected characteristics',
            ),
            # === DIAGNOSTICS ===
            SignalDescriptor(
                name='persona_breaks_count',
                extractor=lambda r: len(r.persona_breaks),
                description=f'Number of turns below {self.persona_break_threshold} adherence threshold',
            ),
            SignalDescriptor(
                name='tone_drift_detected',
                extractor=lambda r: r.tone_drift_detected,
                description='Whether tone drift was detected over time',
            ),
            SignalDescriptor(
                name='drift_direction',
                extractor=lambda r: r.drift_direction or 'none',
                description='Direction of tone drift if detected',
            ),
            SignalDescriptor(
                name='adherence_variance',
                extractor=lambda r: r.adherence_variance,
                description='Variance in adherence scores (lower is better)',
            ),
            SignalDescriptor(
                name='worst_turn_index',
                extractor=lambda r: r.worst_turn_index
                if r.worst_turn_index is not None
                else 'N/A',
                description='Turn with lowest adherence score',
            ),
            # === AGGREGATE STATISTICS ===
            SignalDescriptor(
                name='total_agent_turns_analyzed',
                extractor=lambda r: len(r.turn_analyses),
                description='Number of agent turns analyzed',
            ),
            SignalDescriptor(
                name='perfect_adherence_count',
                extractor=lambda r: sum(
                    1 for t in r.turn_analyses if t.adherence_score >= 0.9
                ),
                description='Turns with adherence ≥ 0.9',
            ),
            SignalDescriptor(
                name='perfect_adherence_rate',
                extractor=lambda r: (
                    sum(1 for t in r.turn_analyses if t.adherence_score >= 0.9)
                    / len(r.turn_analyses)
                    if r.turn_analyses
                    else 0.0
                ),
                description='Percentage of turns with excellent adherence',
            ),
        ]

        # === PER-TURN SIGNALS (sample first 5 and last 3) ===
        turns_to_show = list(range(min(5, len(result.turn_analyses)))) + list(
            range(max(5, len(result.turn_analyses) - 3), len(result.turn_analyses))
        )
        turns_to_show = sorted(set(turns_to_show))

        for idx in turns_to_show:
            if idx >= len(result.turn_analyses):
                continue

            group = f'agent_turn_{idx}'

            signals.extend(
                [
                    SignalDescriptor(
                        name='adherence_score',
                        group=group,
                        extractor=lambda r, i=idx: r.turn_analyses[i].adherence_score,
                        headline_display=True,
                        description='Persona adherence for this turn',
                    ),
                    SignalDescriptor(
                        name='tone_classification',
                        group=group,
                        extractor=lambda r, i=idx: r.turn_analyses[
                            i
                        ].tone_classification,
                        description='Detected tone in this turn',
                    ),
                    SignalDescriptor(
                        name='persona_match',
                        group=group,
                        extractor=lambda r, i=idx: r.turn_analyses[i].persona_match,
                        description='Whether turn matched target persona',
                    ),
                    SignalDescriptor(
                        name='deviation_type',
                        group=group,
                        extractor=lambda r, i=idx: r.turn_analyses[i].deviation_type
                        or 'none',
                        description='Type of deviation if any',
                    ),
                    SignalDescriptor(
                        name='positive_indicators',
                        group=group,
                        extractor=lambda r, i=idx: ', '.join(
                            r.turn_analyses[i].positive_indicators[:2]
                        )
                        + (
                            '...'
                            if len(r.turn_analyses[i].positive_indicators) > 2
                            else ''
                        )
                        if r.turn_analyses[i].positive_indicators
                        else 'none',
                        description='What matched the persona',
                    ),
                    SignalDescriptor(
                        name='negative_indicators',
                        group=group,
                        extractor=lambda r, i=idx: ', '.join(
                            r.turn_analyses[i].negative_indicators[:2]
                        )
                        + (
                            '...'
                            if len(r.turn_analyses[i].negative_indicators) > 2
                            else ''
                        )
                        if r.turn_analyses[i].negative_indicators
                        else 'none',
                        description='What violated the persona',
                    ),
                ]
            )

        # === PERSONA BREAK DETAILS ===
        for i, pb in enumerate(result.persona_breaks[:5]):  # Show first 5 breaks
            group = f"persona_break_{i + 1} (turn_{pb['turn_index']})"

            signals.extend(
                [
                    SignalDescriptor(
                        name='adherence_score',
                        group=group,
                        extractor=lambda r, idx=i: r.persona_breaks[idx][
                            'adherence_score'
                        ],
                        headline_display=True,
                        description='Adherence score at break point',
                    ),
                    SignalDescriptor(
                        name='tone_detected',
                        group=group,
                        extractor=lambda r, idx=i: r.persona_breaks[idx][
                            'tone_classification'
                        ],
                        description='Actual tone detected',
                    ),
                    SignalDescriptor(
                        name='deviation_type',
                        group=group,
                        extractor=lambda r, idx=i: r.persona_breaks[idx][
                            'deviation_type'
                        ],
                        description='Type of persona violation',
                    ),
                    SignalDescriptor(
                        name='violations',
                        group=group,
                        extractor=lambda r, idx=i: ', '.join(
                            r.persona_breaks[idx]['negative_indicators'][:2]
                        )
                        + (
                            '...'
                            if len(r.persona_breaks[idx]['negative_indicators']) > 2
                            else ''
                        )
                        if r.persona_breaks[idx]['negative_indicators']
                        else 'none',
                        description='What violated the persona',
                    ),
                ]
            )

        return signals
