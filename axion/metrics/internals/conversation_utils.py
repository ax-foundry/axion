from typing import List, Literal, Optional, Union

from axion._core.logging import get_logger
from axion._core.schema import AIMessage, HumanMessage, RichBaseModel
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric
from axion.metrics.cache import AnalysisCache
from pydantic import ConfigDict, Field

logger = get_logger(__name__)


class ConversationMoment(RichBaseModel):
    """
    Represents a single topical segment or resolved aspect within a conversation.

    A "moment" is a self-contained segment focused on a single topic or sub-goal.
    Moments are identified by shifts in conversation topic or direction.

    Used by:
    - ConversationFlow: To analyze coherence within topical segments
    - EnhancedGoalCompletion: To map which moments addressed which sub-goals
    """

    id: Optional[int] = Field(
        default=None, description='Sequential identifier for the moment'
    )
    topic: str = Field(description='Primary topic or theme of this moment')
    start_turn: int = Field(
        description='Turn index where this moment begins (inclusive)'
    )
    end_turn: int = Field(description='Turn index where this moment ends (inclusive)')
    turn_count: int = Field(description='Number of turns in this moment')
    coherence_score: float = Field(
        description='Coherence within this moment (0-1)', default=0.0
    )


class TurnAnalysis(RichBaseModel):
    """
    Detailed analysis for a single turn in the conversation.

    Classifies the conversational purpose and linguistic function of each turn.

    Used by:
    - ConversationFlow: To track state progression and detect issues
    - EnhancedGoalCompletion: To understand temporal goal progression
    """

    turn_index: int = Field(description='Index of this turn in the conversation')
    role: Literal['user', 'agent'] = Field(description='Speaker role for this turn')
    state: Literal[
        'Greeting/Small Talk',
        'Intent Clarification',
        'Information Gathering',
        'Tool Execution',
        'Solution Proposal',
        'Refinement/Correction',
        'Confirmation',
        'Error/Recovery',
    ] = Field(description='The conversational state of this turn')
    dialogue_act: Literal[
        'statement',
        'question',
        'command',
        'apology',
        'thanking',
        'agreement',
        'disagreement',
        'instruction',
    ] = Field(description='The dialogue act performed in this turn')
    user_effort_score: Optional[float] = Field(
        default=None,
        description='A score from 0-1 representing the effort required from the user in this turn (user turns only)',
    )


class MomentSegmentationInput(RichBaseModel):
    """Input for moment segmentation."""

    conversation_transcript: str = Field(
        description='The full conversation as a formatted transcript'
    )


class MomentSegmentationOutput(RichBaseModel):
    """Output from moment segmentation."""

    moments: List[ConversationMoment] = Field(
        description='The list of identified conversation moments'
    )


class MomentSegmenter(BaseMetric[MomentSegmentationInput, MomentSegmentationOutput]):
    """
    Segments a conversation into topical moments using an LLM.

    A moment is a self-contained segment focused on a single topic or sub-goal.
    A new moment begins when a new topic is introduced or a significant shift
    in direction occurs.

    This is a foundational analysis used by multiple metrics.
    """

    as_structured_llm = False

    instruction = """You are an expert conversation analyst. Your task is to segment the conversation into distinct "moments".

A moment is a self-contained segment focused on a single topic or sub-goal. A new moment begins when a new topic is introduced or a significant shift in direction occurs.

### Instructions (Chain of Thought):
1. **Parse the Input**: Read through the conversation transcript carefully.
2. **Identify Topic Shifts**: Pinpoint where the conversation changes focus or direction.
3. **Define Boundaries**: For each identified topic, determine its start and end turn index.
4. **Assign Topics**: Give each moment a concise, descriptive topic name (e.g., "Password Reset Request", "Billing Inquiry").
5. **Format Output**: Structure your final analysis into the required JSON format with a single key "moments".

Your output MUST be a JSON object with a single key "moments" containing a list of moment objects."""

    input_model = MomentSegmentationInput
    output_model = MomentSegmentationOutput

    examples = [
        (
            MomentSegmentationInput(
                conversation_transcript="""User: Hi, I'm having trouble logging in.
Assistant: I can help. What seems to be the problem?
User: My password isn't working.
Assistant: Okay, let's reset it. What's your email?
User: test@example.com
Assistant: Thanks. Reset link sent. Anything else?
User: Yes, I also wanted to check my last bill.
Assistant: I can pull that up now."""
            ),
            MomentSegmentationOutput(
                moments=[
                    ConversationMoment(
                        id=1,
                        topic='Password Reset Request',
                        start_turn=0,
                        end_turn=5,
                        turn_count=6,
                    ),
                    ConversationMoment(
                        id=2,
                        topic='Billing Inquiry',
                        start_turn=6,
                        end_turn=7,
                        turn_count=2,
                    ),
                ]
            ),
        )
    ]

    async def execute_from_item(self, item: DatasetItem) -> MomentSegmentationOutput:
        """
        Convenience method to segment moments directly from a DatasetItem.

        Args:
            item: DatasetItem containing the conversation

        Returns:
            MomentSegmentationOutput with identified moments
        """
        transcript = (
            item.to_transcript()
            if hasattr(item, 'to_transcript')
            else self._build_transcript(item)
        )
        input_data = MomentSegmentationInput(conversation_transcript=transcript)
        return await self.execute(input_data)

    @staticmethod
    def _build_transcript(item: DatasetItem) -> str:
        """Build transcript from conversation messages."""
        if not item.conversation:
            return ''

        lines = []
        for msg in item.conversation.messages:
            role = (
                msg.role.capitalize()
                if hasattr(msg.role, 'capitalize')
                else str(msg.role)
            )
            content = msg.content or ''
            lines.append(f'{role}: {content}')
        return '\n'.join(lines)


class TurnAnalyzerInput(RichBaseModel):
    """Input for analyzing a single turn."""

    transcript_context: str = Field(
        description='The conversation transcript up to (but not including) the current turn'
    )
    current_turn_role: str = Field(
        description="The role of the speaker for the current turn (e.g., 'User' or 'Agent')"
    )
    current_turn_content: str = Field(
        description='The text content of the current turn to be analyzed'
    )


class TurnAnalyzerOutput(RichBaseModel):
    """Output from turn analysis."""

    state: Literal[
        'Greeting/Small Talk',
        'Intent Clarification',
        'Information Gathering',
        'Tool Execution',
        'Solution Proposal',
        'Refinement/Correction',
        'Confirmation',
        'Error/Recovery',
    ] = Field(description='The conversational state of this turn')
    dialogue_act: Literal[
        'statement',
        'question',
        'command',
        'apology',
        'thanking',
        'agreement',
        'disagreement',
        'instruction',
    ] = Field(description='The dialogue act performed in this turn')


class TurnAnalyzer(BaseMetric[TurnAnalyzerInput, TurnAnalyzerOutput]):
    """
    Analyzes a single conversation turn to classify its state and dialogue act.

    State: The conversational purpose (what is trying to be accomplished)
    Dialogue Act: The linguistic function (how it's being said)

    This is used to build turn-by-turn analysis for conversation flow tracking.
    """

    as_structured_llm = False

    instruction = """You are an expert conversation analyst. Analyze the current turn within the context of the conversation.

### Evaluation Criteria:
1. **State Classification**: Classify the primary purpose of the current turn in the conversation's flow.
   - **State Options**: "Greeting/Small Talk", "Intent Clarification", "Information Gathering", "Tool Execution", "Solution Proposal", "Refinement/Correction", "Confirmation", "Error/Recovery"

2. **Dialogue Act Classification**: Classify the rhetorical function of the current turn.
   - **Dialogue Act Options**: "statement", "question", "command", "apology", "thanking", "agreement", "disagreement", "instruction"

### Instructions (Chain of Thought):
1. **Analyze Context**: Read the transcript_context to understand what happened before this turn.
2. **Analyze Current Turn**: Read the current_turn_content.
3. **Determine State**: Based on the context and current turn, decide its main purpose. Is it asking for information? Providing it? Correcting something?
4. **Determine Dialogue Act**: Decide the grammatical or social function of the turn. Is it a question? A statement of fact? An agreement?
5. **Format Output**: Your output MUST be a JSON object with the exact keys "state" and "dialogue_act"."""

    input_model = TurnAnalyzerInput
    output_model = TurnAnalyzerOutput

    examples = [
        (
            TurnAnalyzerInput(
                transcript_context="User: Hi, I'd like to book a flight.\nAgent: I can help with that. Where would you like to go?",
                current_turn_role='User',
                current_turn_content='I need to go to San Francisco.',
            ),
            TurnAnalyzerOutput(state='Information Gathering', dialogue_act='statement'),
        )
    ]


class UserFrictionAnalyzerInput(RichBaseModel):
    """Input for analyzing user friction."""

    transcript_context: str = Field(
        description="The conversation transcript leading up to the user's turn"
    )
    user_turn_content: str = Field(
        description="The user's message to be analyzed for friction"
    )


class UserFrictionAnalyzerOutput(RichBaseModel):
    """Output from friction analysis."""

    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    friction_score: float = Field(
        description='A score from 0.0 (no friction) to 1.0 (high friction), indicating user frustration, confusion, or effort'
    )
    indicators: List[str] = Field(
        default_factory=list,
        description="Specific indicators of friction found in the user's message",
    )


class UserFrictionAnalyzer(
    BaseMetric[UserFrictionAnalyzerInput, UserFrictionAnalyzerOutput]
):
    """
    Analyzes user messages for signs of friction, frustration, or confusion.

    Used to calculate user effort scores and detect conversation problems.
    """

    as_structured_llm = True

    instruction = """You are an expert in user experience analysis. Analyze the user's message for signs of conversational friction (confusion, frustration, repetition, correction).

**Friction Scoring Guide:**
- **0.0**: User is perfectly happy, no signs of friction
- **0.3**: Minor friction (slight confusion, minor rephrasing)
- **0.5**: Moderate friction (clear rephrasing, some frustration)
- **0.7**: Significant friction (repeating information, visible frustration)
- **1.0**: High friction (explicit complaints, anger, giving up)

**Friction Indicators to Look For:**
- Repetition of information already provided
- Corrections of agent misunderstandings
- Expressions of frustration ("I already told you...", "No, that's not what I meant...")
- Sarcasm or passive aggression
- Increasingly brief or curt responses
- Excessive detail/explanation (user over-compensating)

Analyze the user's turn and provide:
1. A friction_score between 0.0 and 1.0
2. A list of specific indicators (can be empty list if no friction detected)"""

    input_model = UserFrictionAnalyzerInput
    output_model = UserFrictionAnalyzerOutput

    examples = [
        (
            UserFrictionAnalyzerInput(
                transcript_context="Agent: What's your email?\nUser: john@example.com\nAgent: Can you provide your email?",
                user_turn_content="I just told you, it's john@example.com",
            ),
            UserFrictionAnalyzerOutput(
                friction_score=0.6,
                indicators=[
                    'User is repeating information already provided',
                    "Slight frustration in tone ('I just told you')",
                ],
            ),
        ),
        (
            UserFrictionAnalyzerInput(
                transcript_context='Agent: How can I help you today?',
                user_turn_content='I need to reset my password',
            ),
            UserFrictionAnalyzerOutput(friction_score=0.0, indicators=[]),
        ),
    ]


class MomentCoherenceInput(RichBaseModel):
    """Input for evaluating moment coherence."""

    moment_transcript: str = Field(
        description='The transcript of a single conversational moment'
    )


class MomentCoherenceOutput(RichBaseModel):
    """Output from coherence evaluation."""

    coherence_score: float = Field(
        description='A score from 0.0 (incoherent) to 1.0 (perfectly coherent) for the moment'
    )
    analysis: str = Field(description='A brief justification for the coherence score')


class MomentCoherenceEvaluator(BaseMetric[MomentCoherenceInput, MomentCoherenceOutput]):
    """
    Evaluates the internal coherence of a conversation moment.

    Coherence means the conversation flows logically and stays on topic within the moment.
    """

    as_structured_llm = False

    instruction = """Evaluate the internal coherence of the provided moment transcript.

**Coherence Criteria:**
- Does the conversation flow logically?
- Do responses address what was said before?
- Does the conversation stay on the stated topic?
- Are there awkward transitions or non-sequiturs?

**Scoring Guide:**
- **1.0**: Perfectly coherent - smooth logical flow, all on topic
- **0.7**: Mostly coherent - minor awkward transitions
- **0.5**: Moderately coherent - some confusion or off-topic drift
- **0.3**: Low coherence - frequent confusion or topic jumping
- **0.0**: Completely incoherent - no logical flow

Provide a `coherence_score` and a brief `analysis` explaining your reasoning."""

    input_model = MomentCoherenceInput
    output_model = MomentCoherenceOutput

    examples = [
        # Example 1: High Coherence
        (
            MomentCoherenceInput(
                moment_transcript="""User: I need to book a flight to New York.
    Agent: I can help with that. What are your travel dates?
    User: I'd like to leave on October 26th and return on the 31st."""
            ),
            MomentCoherenceOutput(
                coherence_score=0.9,
                analysis="The conversation is highly coherent. The agent's question directly follows the user's request, and the user's response provides the necessary information.",
            ),
        ),
        # Example 2: Moderate Coherence
        (
            MomentCoherenceInput(
                moment_transcript="""User: What's the status of my order?
    Agent: It's currently out for delivery and should arrive today.
    User: Great. Can you also check my account balance?"""
            ),
            MomentCoherenceOutput(
                coherence_score=0.6,
                analysis="The moment starts coherently but the user introduces a new, unrelated topic ('account balance') without a transition, slightly disrupting the flow.",
            ),
        ),
        # Example 3: Low Coherence
        (
            MomentCoherenceInput(
                moment_transcript="""User: My internet is down.
    Agent: Okay, let's start by restarting your router.
    User: I like pizza."""
            ),
            MomentCoherenceOutput(
                coherence_score=0.1,
                analysis="The user's final turn is a complete non-sequitur and has no logical connection to the agent's instruction, making the moment highly incoherent.",
            ),
        ),
    ]


async def _compute_moments(
    item: DatasetItem, segmenter: MomentSegmenter
) -> List[ConversationMoment]:
    """
    Compute conversation moments with validation and fallback handling.
    """
    try:
        # Build transcript (supports both custom and generic builders)
        transcript = (
            item.to_transcript()
            if hasattr(item, 'to_transcript')
            else _build_transcript(item)
        )
        result = await segmenter.execute(
            MomentSegmentationInput(conversation_transcript=transcript)
        )
        moments = result.moments or []

        # Ensure moments align with message bounds
        if item.conversation.messages:
            last_valid_index = len(item.conversation.messages) - 1
            for moment in moments:
                if moment.end_turn > last_valid_index:
                    logger.warning(
                        f'Correcting out-of-bounds end_turn {moment.end_turn} '
                        f'→ {last_valid_index} for item {item.id}'
                    )
                    moment.end_turn = last_valid_index

                if moment.start_turn > last_valid_index:
                    logger.warning(
                        f'Correcting out-of-bounds start_turn {moment.start_turn} '
                        f'→ {last_valid_index} for item {item.id}'
                    )
                    moment.start_turn = last_valid_index

                if moment.start_turn > moment.end_turn:
                    moment.start_turn = moment.end_turn

                moment.turn_count = moment.end_turn - moment.start_turn + 1

        return moments

    except Exception as e:
        logger.warning(
            f'Moment segmentation failed for item {item.id}: {e}. Using fallback.'
        )
        return [
            ConversationMoment(
                id=1,
                topic='Full Conversation',
                start_turn=0,
                end_turn=len(item.conversation.messages) - 1
                if item.conversation.messages
                else 0,
                turn_count=len(item.conversation.messages),
            )
        ]


async def get_or_compute_moments(
    item: DatasetItem, cache: Optional[AnalysisCache], segmenter: MomentSegmenter
) -> List[ConversationMoment]:
    """
    Retrieve or safely compute conversation moments with:
    - Cache validation and correction
    - Concurrency-safe lock handling
    - Graceful fallback on segmentation failure
    """
    cache_key = 'moments'

    # Use cache if available and valid
    if cache and cache.has(item.id, cache_key):
        cached_moments = cache.get(item.id, cache_key)
        if cached_moments:
            max_turn = max((m.end_turn for m in cached_moments), default=-1)
            if max_turn < len(item.conversation.messages):
                logger.debug(f'Using validated cached moments for item {item.id}')
                return cached_moments
            logger.warning(
                f'Invalid cached moments detected for item {item.id}. Recomputing.'
            )

    # ⚙Handle missing/invalid cache safely with concurrency lock
    if not cache:
        logger.debug(
            f'No cache provided — computing moments directly for item {item.id}'
        )
        return await _compute_moments(item, segmenter)

    lock = cache.get_lock(item.id, cache_key)
    async with lock:
        # Double-check cache after acquiring the lock
        if cache.has(item.id, cache_key):
            cached_moments = cache.get(item.id, cache_key)
            if cached_moments:
                max_turn = max((m.end_turn for m in cached_moments), default=-1)
                if max_turn < len(item.conversation.messages):
                    return cached_moments

        logger.debug(f'Computing moments for item {item.id} inside lock')
        moments = await _compute_moments(item, segmenter)
        cache.set(item.id, cache_key, moments)
        return moments


async def _compute_turn_analysis(
    item: DatasetItem,
    turn_analyzer: TurnAnalyzer,
    friction_analyzer: Optional[UserFrictionAnalyzer] = None,
) -> List[TurnAnalysis]:
    """
    Compute detailed turn-by-turn analysis for a conversation.
    Includes user effort estimation and fault tolerance for each turn.
    """
    if not item.conversation or not item.conversation.messages:
        return []

    analysis: List[TurnAnalysis] = []

    for i, msg in enumerate(item.conversation.messages):
        role = 'user' if isinstance(msg, HumanMessage) else 'agent'
        context = '\n'.join(
            f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content or ''}"
            for m in item.conversation.messages[:i]
        )

        try:
            # ---- Perform LLM-based turn analysis ----
            analysis_input = TurnAnalyzerInput(
                transcript_context=context,
                current_turn_role=role.capitalize(),
                current_turn_content=msg.content or '',
            )
            llm_result = await turn_analyzer.execute(analysis_input)

            # ---- Optional user friction analysis ----
            user_effort = None
            if isinstance(msg, HumanMessage) and friction_analyzer:
                try:
                    friction_input = UserFrictionAnalyzerInput(
                        transcript_context=context,
                        user_turn_content=msg.content or '',
                    )
                    friction_result = await friction_analyzer.execute(friction_input)
                    length_effort = min(1.0, len(msg.content or '') / 400.0)
                    user_effort = (length_effort * 0.4) + (
                        friction_result.friction_score * 0.6
                    )
                except Exception as e:
                    logger.warning(
                        f'Friction analysis failed at turn {i} for item {item.id}: {e}'
                    )
                    user_effort = min(1.0, len(msg.content or '') / 400.0)

            analysis.append(
                TurnAnalysis(
                    turn_index=i,
                    role=role,
                    state=llm_result.state,
                    dialogue_act=llm_result.dialogue_act,
                    user_effort_score=user_effort,
                )
            )

        except Exception as e:
            # ---- Fallback for individual turn errors ----
            logger.warning(
                f'Turn analysis failed at turn {i} for item {item.id}: {e}. Using fallback.'
            )
            user_effort = (
                min(1.0, len(msg.content or '') / 400.0)
                if isinstance(msg, HumanMessage)
                else None
            )
            analysis.append(
                TurnAnalysis(
                    turn_index=i,
                    role=role,
                    state='Error/Recovery',
                    dialogue_act='statement',
                    user_effort_score=user_effort,
                )
            )

    return analysis


async def get_or_compute_turn_analysis(
    item: DatasetItem,
    cache: Optional[AnalysisCache],
    turn_analyzer: TurnAnalyzer,
    friction_analyzer: Optional[UserFrictionAnalyzer] = None,
) -> List[TurnAnalysis]:
    """
    Retrieve or compute turn-by-turn analysis with concurrency-safe caching and validation.

    - Prevents redundant computations with async locking.
    - Validates cache consistency (turn count).
    - Falls back gracefully on errors.
    """
    cache_key = 'turn_by_turn_analysis'

    # Cache hit with validation
    if cache and cache.has(item.id, cache_key):
        cached_analysis = cache.get(item.id, cache_key)
        if cached_analysis and len(cached_analysis) == len(item.conversation.messages):
            logger.debug(f'Using valid cached turn analysis for item {item.id}')
            return cached_analysis
        else:
            logger.warning(
                f'Invalid or outdated cached analysis for item {item.id}. Recomputing.'
            )

    # ⚙Compute safely with concurrency lock
    if not cache:
        logger.debug(
            f'No cache provided — computing turn analysis directly for item {item.id}'
        )
        return await _compute_turn_analysis(item, turn_analyzer, friction_analyzer)

    lock = cache.get_lock(item.id, cache_key)
    async with lock:
        # Double-check inside the lock to avoid duplicate computation
        if cache.has(item.id, cache_key):
            cached_analysis = cache.get(item.id, cache_key)
            if cached_analysis and len(cached_analysis) == len(
                item.conversation.messages
            ):
                return cached_analysis

        logger.debug(f'Computing turn analysis for item {item.id} inside lock')
        analysis = await _compute_turn_analysis(item, turn_analyzer, friction_analyzer)
        cache.set(item.id, cache_key, analysis)
        return analysis


def _build_transcript(item: DatasetItem) -> str:
    """
    Build a simple, formatted transcript from conversation messages.

    Args:
        item: The DatasetItem containing the conversation.

    Returns:
        A formatted transcript string (e.g., "User: Hello\nAgent: Hi there").
    """
    if not item.conversation or not item.conversation.messages:
        return ''

    lines = []
    for msg in item.conversation.messages:
        # Determine the role, defaulting to a capitalized version of the role attribute
        if isinstance(msg, HumanMessage):
            role = 'User'
        elif isinstance(msg, AIMessage):
            role = 'Agent'
        else:
            # Fallback for other potential message types
            role = str(getattr(msg, 'role', 'System')).capitalize()

        content = msg.content or ''
        lines.append(f'{role}: {content}')

    return '\n'.join(lines)


def build_transcript_from_messages(
    messages: List[Union[HumanMessage, AIMessage]],
) -> str:
    """
    Build a formatted transcript from conversation messages.

    Args:
        messages: List of conversation messages

    Returns:
        Formatted transcript string
    """
    lines = []
    for msg in messages:
        role = 'User' if isinstance(msg, HumanMessage) else 'Agent'
        content = msg.content or ''
        lines.append(f'{role}: {content}')
    return '\n'.join(lines)


def extract_moment_transcript(
    messages: List[Union[HumanMessage, AIMessage]], moment: ConversationMoment
) -> str:
    """
    Extract the transcript for a specific conversation moment.

    Args:
        messages: Full list of conversation messages
        moment: ConversationMoment to extract

    Returns:
        Transcript string for just that moment
    """
    moment_messages = messages[moment.start_turn : moment.end_turn + 1]
    return build_transcript_from_messages(moment_messages)
