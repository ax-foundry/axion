from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import LLMRunnable, StrictBaseModel
from axion._handlers.llm.handler import LLMHandler
from axion.llm_registry import LLMRegistry

logger = get_logger(__name__)


def _get_first_dict(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """
    Return the first dict value found in priority order.

    Note: Uses key presence (`k in d`), not truthiness. This means if the primary
    key exists with value None, we return None rather than falling back to aliases.
    This is intentional - explicit None should not trigger fallback behavior.
    """
    for k in keys:
        if k in d:
            return d[k]
    return default


def _get_first_attr(obj: Any, attrs: Sequence[str], default: Any = None) -> Any:
    """
    Return the first attribute value found in priority order.

    Note: Uses attribute presence (`hasattr`), not truthiness. This means if the
    primary attribute exists with value None, we return None rather than falling
    back to aliases. This is intentional - explicit None should not trigger fallback.
    """
    for a in attrs:
        if hasattr(obj, a):
            return getattr(obj, a)
    return default


# ============================================================================
# Pydantic Models for Misalignment Analysis
# ============================================================================


class MisalignedCase(StrictBaseModel):
    """A single misaligned case for analysis input."""

    record_id: str = Field(description='Unique identifier for the record')
    query: str = Field(description='The original query/input')
    actual_output: str = Field(description='The LLM response being evaluated')
    human_score: int = Field(description='Human annotation score (0 or 1)')
    llm_score: int = Field(description='LLM judge score (0 or 1)')
    llm_reasoning: Optional[str] = Field(
        default=None, description='LLM reasoning for its judgment'
    )
    signals: Optional[Any] = Field(
        default=None,
        description='Optional structured signals from the LLM judge/metric (if available)',
    )


class MisalignmentInput(StrictBaseModel):
    """Input model for misalignment analysis."""

    false_positive_examples: List[MisalignedCase] = Field(
        description='Cases where LLM=1 but Human=0 (too lenient)'
    )
    false_negative_examples: List[MisalignedCase] = Field(
        description='Cases where LLM=0 but Human=1 (too strict)'
    )
    evaluation_criteria: str = Field(description='Current evaluation criteria text')


class PatternOutput(StrictBaseModel):
    """A single pattern identified in the misalignment."""

    pattern_type: str = Field(description='Either "false_positive" or "false_negative"')
    description: str = Field(description='Description of the pattern observed')
    example_ids: List[str] = Field(description='Record IDs that exhibit this pattern')


class MisalignmentOutput(StrictBaseModel):
    """Structured output from misalignment analysis LLM."""

    summary: str = Field(description='2-3 sentence summary of main patterns observed')
    patterns: List[PatternOutput] = Field(
        description='List of identified misalignment patterns'
    )
    recommendations: List[str] = Field(
        description='3-4 specific recommendations to improve alignment'
    )


class OptimizeInput(StrictBaseModel):
    """Input model for prompt optimization."""

    system_prompt: str = Field(description='Current system prompt for the evaluator')
    current_criteria: str = Field(description='Current evaluation criteria')
    false_positive_examples: List[MisalignedCase] = Field(
        description='Cases where LLM=1 but Human=0 (too lenient)'
    )
    false_negative_examples: List[MisalignedCase] = Field(
        description='Cases where LLM=0 but Human=1 (too strict)'
    )
    total_misaligned: int = Field(description='Total number of misaligned cases')
    false_positives: int = Field(description='Number of false positive cases')
    false_negatives: int = Field(description='Number of false negative cases')


class SuggestionOutput(StrictBaseModel):
    """A single optimization suggestion."""

    aspect: str = Field(
        description='Aspect being addressed (e.g., "criteria clarity", "edge cases")'
    )
    suggestion: str = Field(description='Specific change recommended')
    rationale: str = Field(description='Why this helps improve alignment')


class OptimizeOutput(StrictBaseModel):
    """Structured output from prompt optimization LLM."""

    optimized_criteria: str = Field(description='Improved evaluation criteria text')
    suggestions: List[SuggestionOutput] = Field(
        description='Specific suggestions for improvement'
    )
    expected_improvement: str = Field(
        description='Rationale for why these changes should improve alignment'
    )


# ============================================================================
# Shared Extraction Function
# ============================================================================


def _extract_misaligned_cases(
    results: Union[List[Dict[str, Any]], List[Any]],
) -> tuple[List[MisalignedCase], List[MisalignedCase]]:
    """
    Extract false positive and false negative cases from evaluation results.

    Supports multiple input formats:
    - Primary keys: record_id, human_score, llm_score, query, actual_output, llm_reasoning
    - Aliases for DatasetItem/MetricScore compatibility: id, judgment, score, explanation

    Args:
        results: List of evaluation results (dicts or objects with attributes)

    Returns:
        Tuple of (false_positives, false_negatives) where:
        - false_positives: LLM=1, Human=0 (LLM too lenient)
        - false_negatives: LLM=0, Human=1 (LLM too strict)
    """
    false_positives: List[MisalignedCase] = []
    false_negatives: List[MisalignedCase] = []

    for item in results:
        # Handle both dict and object access
        if isinstance(item, dict):
            # Primary keys (back-compat) + aliases (DatasetItem/MetricScore-friendly)
            record_id = _get_first_dict(item, ('record_id', 'id'), default='')
            query = _get_first_dict(item, ('query',), default='')
            actual_output = _get_first_dict(item, ('actual_output',), default='')

            human_score = _get_first_dict(
                item,
                ('human_score', 'judgment'),
                default=0,
            )
            llm_score = _get_first_dict(
                item,
                ('llm_score', 'score'),
                default=0,
            )
            llm_reasoning = _get_first_dict(
                item,
                ('llm_reasoning', 'explanation'),
                default=None,
            )
            signals = _get_first_dict(item, ('signals',), default=None)
        else:
            record_id = _get_first_attr(item, ('record_id', 'id'), default='')
            query = _get_first_attr(item, ('query',), default='')
            actual_output = _get_first_attr(item, ('actual_output',), default='')

            human_score = _get_first_attr(
                item,
                ('human_score', 'judgment'),
                default=0,
            )
            llm_score = _get_first_attr(
                item,
                ('llm_score', 'score'),
                default=0,
            )
            llm_reasoning = _get_first_attr(
                item,
                ('llm_reasoning', 'explanation'),
                default=None,
            )
            signals = _get_first_attr(item, ('signals',), default=None)

        # Check for misalignment
        if human_score != llm_score:
            case = MisalignedCase(
                record_id=str(record_id),
                query=str(query),
                actual_output=str(actual_output),
                human_score=int(human_score),
                llm_score=int(llm_score),
                llm_reasoning=str(llm_reasoning) if llm_reasoning else None,
                signals=signals,
            )

            if llm_score == 1 and human_score == 0:
                false_positives.append(case)
            elif llm_score == 0 and human_score == 1:
                false_negatives.append(case)

    return false_positives, false_negatives


# ============================================================================
# Data Classes for API Results
# ============================================================================


@dataclass
class MisalignmentPattern:
    """A discovered pattern in misalignment analysis."""

    pattern_type: str  # 'false_positive' or 'false_negative'
    description: str
    count: int
    example_ids: List[str]


@dataclass
class MisalignmentAnalysis:
    """Complete result from misalignment analysis."""

    total_misaligned: int
    false_positives: int  # LLM=1, Human=0 (too lenient)
    false_negatives: int  # LLM=0, Human=1 (too strict)
    patterns: List[MisalignmentPattern]
    summary: str
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptSuggestion:
    """A single suggestion for prompt improvement."""

    aspect: str
    suggestion: str
    rationale: str


@dataclass
class OptimizedPrompt:
    """Complete result from prompt optimization."""

    original_criteria: str
    optimized_criteria: str
    suggestions: List[PromptSuggestion]
    expected_improvement: str
    metadata: Dict[str, Any] = field(default_factory=dict)


DEFAULT_MISALIGNMENT_INSTRUCTION = """You are an expert at analyzing evaluation misalignment between LLM judges and human annotators.

Analyze the provided misalignment patterns to understand why the LLM judge disagrees with human annotations.

## Terminology
- **False Positives**: Cases where the LLM accepted (score=1) but human rejected (score=0). The LLM was too lenient.
- **False Negatives**: Cases where the LLM rejected (score=0) but human accepted (score=1). The LLM was too strict.

## Your Task
1. Identify common patterns in the misalignments
2. Understand why the LLM's judgment differs from human judgment
3. Provide actionable recommendations to improve alignment

## Guidelines
- Look for systematic biases (e.g., LLM is consistently strict/lenient on certain types of responses)
- Consider whether the evaluation criteria are ambiguous
- Identify edge cases that the criteria don't handle well
- Recommendations should be specific and actionable
"""


class MisalignmentAnalysisHandler(LLMHandler[MisalignmentInput, MisalignmentOutput]):
    """LLM handler for analyzing misalignment patterns."""

    input_model: Type[MisalignmentInput] = MisalignmentInput
    output_model: Type[MisalignmentOutput] = MisalignmentOutput
    instruction: str = DEFAULT_MISALIGNMENT_INSTRUCTION
    generation_fake_sample: bool = False

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        llm_provider: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the misalignment analysis handler.

        Args:
            model_name: Name of the LLM model (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
            llm: Pre-configured LLM instance
            llm_provider: LLM provider ('openai', 'anthropic')
            instruction: Custom instruction to override default
        """
        if instruction:
            self.instruction = instruction

        self._set_llm(llm, model_name, llm_provider)
        super().__init__(**kwargs)

    def _set_llm(
        self,
        llm: Optional[LLMRunnable],
        model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> None:
        """Set the LLM instance from provided params or registry."""
        if llm is not None:
            self.llm = llm
        elif model_name is not None or llm_provider is not None:
            registry = LLMRegistry(llm_provider)
            self.llm = registry.get_llm(model_name)
        else:
            registry = LLMRegistry(llm_provider)
            self.llm = registry.get_llm(model_name)

        self.model_name = model_name or getattr(self.llm, 'model', None)
        self.llm_provider = llm_provider or getattr(self.llm, '_provider', None)


DEFAULT_OPTIMIZE_INSTRUCTION = """You are an expert at calibrating LLM-as-a-judge evaluators to align with human judgment.

Your task is to improve the evaluation criteria to reduce misalignment between the LLM judge and human annotators.

## Current Misalignment Statistics
You will be provided with:
- The current system prompt and evaluation criteria
- Examples of false positives (LLM too lenient)
- Examples of false negatives (LLM too strict)
- Total counts of misaligned cases

## Your Task
1. Analyze why the current criteria leads to misalignment
2. Write improved evaluation criteria that would reduce misalignment
3. Provide specific suggestions for each aspect that needs improvement
4. Explain why your changes should improve alignment

## Guidelines
- Preserve the original intent and scope of the evaluation
- Add clarity where the original criteria are ambiguous
- Add specific guidance for edge cases
- Consider adding examples of borderline cases
- Make criteria more operational (easier for an LLM to apply consistently)
"""


class PromptOptimizationHandler(LLMHandler[OptimizeInput, OptimizeOutput]):
    """LLM handler for optimizing evaluation prompts."""

    input_model: Type[OptimizeInput] = OptimizeInput
    output_model: Type[OptimizeOutput] = OptimizeOutput
    instruction: str = DEFAULT_OPTIMIZE_INSTRUCTION
    generation_fake_sample: bool = False

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        llm_provider: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the prompt optimization handler.

        Args:
            model_name: Name of the LLM model (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
            llm: Pre-configured LLM instance
            llm_provider: LLM provider ('openai', 'anthropic')
            instruction: Custom instruction to override default
        """
        if instruction:
            self.instruction = instruction

        self._set_llm(llm, model_name, llm_provider)
        super().__init__(**kwargs)

    def _set_llm(
        self,
        llm: Optional[LLMRunnable],
        model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> None:
        """Set the LLM instance from provided params or registry."""
        if llm is not None:
            self.llm = llm
        elif model_name is not None or llm_provider is not None:
            registry = LLMRegistry(llm_provider)
            self.llm = registry.get_llm(model_name)
        else:
            registry = LLMRegistry(llm_provider)
            self.llm = registry.get_llm(model_name)

        self.model_name = model_name or getattr(self.llm, 'model', None)
        self.llm_provider = llm_provider or getattr(self.llm, '_provider', None)


class MisalignmentAnalyzer:
    """
    Analyzes misalignment between LLM judges and human annotators.

    This class leverages LLMHandler for structured output, automatic retries,
    and consistent LLM configuration with the rest of axion.

    Example:
        >>> from axion.align import MisalignmentAnalyzer
        >>>
        >>> results = [
        ...     {'record_id': 'r1', 'human_score': 1, 'llm_score': 0,
        ...      'query': '...', 'actual_output': '...', 'llm_reasoning': '...'},
        ...     {'record_id': 'r2', 'human_score': 0, 'llm_score': 1,
        ...      'query': '...', 'actual_output': '...', 'llm_reasoning': '...'},
        ... ]
        >>> criteria = "Evaluate whether the response is accurate and helpful."
        >>>
        >>> # Using model_name/provider (recommended)
        >>> analyzer = MisalignmentAnalyzer(model_name='gpt-4o', llm_provider='openai')
        >>> analysis = await analyzer.analyze(results, criteria)
        >>>
        >>> print(f'Summary: {analysis.summary}')
        >>> print(f'Recommendations: {analysis.recommendations}')
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        llm_provider: Optional[str] = None,
        instruction: Optional[str] = None,
        max_examples: int = 10,
    ):
        """
        Initialize MisalignmentAnalyzer.

        Args:
            model_name: Name of the LLM model (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
            llm: Pre-configured LLM instance
            llm_provider: LLM provider ('openai', 'anthropic')
            instruction: Custom instruction to override default analysis prompt
            max_examples: Max examples per category to include in analysis
        """
        self._model_name = model_name
        self._llm = llm
        self._llm_provider = llm_provider
        self._instruction = instruction
        self._max_examples = max_examples

        # Lazily initialized handler
        self._handler: Optional[MisalignmentAnalysisHandler] = None

    def _get_handler(self) -> MisalignmentAnalysisHandler:
        """Get or create the analysis handler."""
        if self._handler is None:
            self._handler = MisalignmentAnalysisHandler(
                model_name=self._model_name,
                llm=self._llm,
                llm_provider=self._llm_provider,
                instruction=self._instruction,
            )
        return self._handler

    async def analyze(
        self,
        results: Union[List[Dict[str, Any]], List[Any]],
        evaluation_criteria: str,
    ) -> MisalignmentAnalysis:
        """
        Analyze misalignment patterns asynchronously.

        Args:
            results: List of evaluation results with human_score, llm_score, etc.
            evaluation_criteria: The current evaluation criteria being used

        Returns:
            MisalignmentAnalysis with patterns, summary, and recommendations
        """
        # Extract misaligned cases
        fp_cases, fn_cases = self._extract_misaligned_cases(results)

        total_misaligned = len(fp_cases) + len(fn_cases)

        # Handle case with no misalignments
        if total_misaligned == 0:
            return MisalignmentAnalysis(
                total_misaligned=0,
                false_positives=0,
                false_negatives=0,
                patterns=[],
                summary='Perfect alignment! No misalignments detected between LLM and human judgments.',
                recommendations=[],
                metadata={'perfect_alignment': True},
            )

        # Sample if too many examples
        fp_sample = fp_cases[: self._max_examples]
        fn_sample = fn_cases[: self._max_examples]

        # Build structured input
        input_data = MisalignmentInput(
            false_positive_examples=fp_sample,
            false_negative_examples=fn_sample,
            evaluation_criteria=evaluation_criteria,
        )

        # Execute with LLMHandler
        handler = self._get_handler()
        output: MisalignmentOutput = await handler.execute(input_data)

        # Convert to result format
        patterns = [
            MisalignmentPattern(
                pattern_type=p.pattern_type,
                description=p.description,
                count=len(p.example_ids),
                example_ids=p.example_ids,
            )
            for p in output.patterns
        ]

        return MisalignmentAnalysis(
            total_misaligned=total_misaligned,
            false_positives=len(fp_cases),
            false_negatives=len(fn_cases),
            patterns=patterns,
            summary=output.summary,
            recommendations=output.recommendations,
            metadata={
                'model': handler.model_name,
                'provider': handler.llm_provider,
                'sampled_fp': len(fp_sample),
                'sampled_fn': len(fn_sample),
            },
        )

    def _extract_misaligned_cases(
        self, results: Union[List[Dict[str, Any]], List[Any]]
    ) -> tuple[List[MisalignedCase], List[MisalignedCase]]:
        """Extract false positive and false negative cases from results."""
        return _extract_misaligned_cases(results)


class PromptOptimizer:
    """
    Optimizes evaluation prompts based on misalignment analysis.

    This class leverages LLMHandler for structured output, automatic retries,
    and consistent LLM configuration with the rest of axion.

    Example:
        >>> from axion.align import PromptOptimizer
        >>>
        >>> results = [
        ...     {'record_id': 'r1', 'human_score': 1, 'llm_score': 0,
        ...      'query': '...', 'actual_output': '...', 'llm_reasoning': '...'},
        ... ]
        >>> criteria = "Evaluate whether the response is accurate."
        >>> system_prompt = "You are an evaluator..."
        >>>
        >>> optimizer = PromptOptimizer(model_name='gpt-4o', llm_provider='openai')
        >>> optimized = await optimizer.optimize(results, criteria, system_prompt)
        >>>
        >>> print(f'Optimized criteria: {optimized.optimized_criteria}')
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        llm_provider: Optional[str] = None,
        instruction: Optional[str] = None,
        max_examples: int = 10,
    ):
        """
        Initialize PromptOptimizer.

        Args:
            model_name: Name of the LLM model (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
            llm: Pre-configured LLM instance
            llm_provider: LLM provider ('openai', 'anthropic')
            instruction: Custom instruction to override default optimization prompt
            max_examples: Max examples per category to include in optimization
        """
        self._model_name = model_name
        self._llm = llm
        self._llm_provider = llm_provider
        self._instruction = instruction
        self._max_examples = max_examples

        # Lazily initialized handler
        self._handler: Optional[PromptOptimizationHandler] = None

    def _get_handler(self) -> PromptOptimizationHandler:
        """Get or create the optimization handler."""
        if self._handler is None:
            self._handler = PromptOptimizationHandler(
                model_name=self._model_name,
                llm=self._llm,
                llm_provider=self._llm_provider,
                instruction=self._instruction,
            )
        return self._handler

    async def optimize(
        self,
        results: Union[List[Dict[str, Any]], List[Any]],
        current_criteria: str,
        system_prompt: str = '',
    ) -> OptimizedPrompt:
        """
        Optimize evaluation criteria asynchronously.

        Args:
            results: List of evaluation results with human_score, llm_score, etc.
            current_criteria: The current evaluation criteria to improve
            system_prompt: The current system prompt (optional)

        Returns:
            OptimizedPrompt with improved criteria and suggestions
        """
        # Extract misaligned cases
        fp_cases, fn_cases = self._extract_misaligned_cases(results)

        total_misaligned = len(fp_cases) + len(fn_cases)

        # Handle case with no misalignments
        if total_misaligned == 0:
            return OptimizedPrompt(
                original_criteria=current_criteria,
                optimized_criteria=current_criteria,
                suggestions=[],
                expected_improvement='No optimization needed - criteria already achieves perfect alignment.',
                metadata={'perfect_alignment': True},
            )

        # Sample if too many examples
        fp_sample = fp_cases[: self._max_examples]
        fn_sample = fn_cases[: self._max_examples]

        # Build structured input
        input_data = OptimizeInput(
            system_prompt=system_prompt or 'You are an AI evaluator.',
            current_criteria=current_criteria,
            false_positive_examples=fp_sample,
            false_negative_examples=fn_sample,
            total_misaligned=total_misaligned,
            false_positives=len(fp_cases),
            false_negatives=len(fn_cases),
        )

        # Execute with LLMHandler
        handler = self._get_handler()
        output: OptimizeOutput = await handler.execute(input_data)

        # Convert to result format
        suggestions = [
            PromptSuggestion(
                aspect=s.aspect,
                suggestion=s.suggestion,
                rationale=s.rationale,
            )
            for s in output.suggestions
        ]

        return OptimizedPrompt(
            original_criteria=current_criteria,
            optimized_criteria=output.optimized_criteria,
            suggestions=suggestions,
            expected_improvement=output.expected_improvement,
            metadata={
                'model': handler.model_name,
                'provider': handler.llm_provider,
                'total_misaligned': total_misaligned,
                'false_positives': len(fp_cases),
                'false_negatives': len(fn_cases),
            },
        )

    def _extract_misaligned_cases(
        self, results: Union[List[Dict[str, Any]], List[Any]]
    ) -> tuple[List[MisalignedCase], List[MisalignedCase]]:
        """Extract false positive and false negative cases from results."""
        return _extract_misaligned_cases(results)
