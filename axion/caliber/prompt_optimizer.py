"""
Prompt optimization for CaliberHQ workflow.

Optimizes LLM-as-judge evaluation prompts based on misalignment analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import LLMRunnable, StrictBaseModel
from axion._handlers.llm.handler import LLMHandler
from axion.caliber.analysis import MisalignedCase, extract_misaligned_cases
from axion.llm_registry import LLMRegistry

logger = get_logger(__name__)


# =============================================================================
# Pydantic Models for Prompt Optimization
# =============================================================================


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


# =============================================================================
# Data Classes for API Results
# =============================================================================


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


class PromptOptimizer:
    """
    Optimizes evaluation prompts based on misalignment analysis.

    This class leverages LLMHandler for structured output, automatic retries,
    and consistent LLM configuration with the rest of axion.

    Example:
        >>> from axion.caliber import PromptOptimizer
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
        fp_cases, fn_cases = extract_misaligned_cases(results)

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
