"""
Optimizer Agent for generating revised prompts based on evaluation feedback.

This module implements the "brain" of the prompt learning system - an LLM-based
agent that analyzes failures and generates improved prompt revisions.
"""

from typing import List, Optional

from pydantic import Field

from axion._core.metadata.schema import ToolMetadata
from axion._core.schema import LLMRunnable, RichBaseModel
from axion._core.tracing import init_tracer
from axion._core.tracing.handlers import BaseTraceHandler
from axion._handlers.llm.handler import LLMHandler


class PromptRevisionInput(RichBaseModel):
    """Input model for the optimizer agent."""

    current_prompt: str = Field(
        description='The current system prompt being optimized.'
    )

    pass_rate: float = Field(description='Current pass rate (0.0 to 1.0).')

    best_pass_rate: float = Field(description='Best pass rate achieved so far.')

    iteration: int = Field(description='Current iteration number.')

    failure_analysis: str = Field(
        description='Formatted analysis of failure cases with metric signals.'
    )

    regression_warning: Optional[str] = Field(
        default=None,
        description='Warning message if the last iteration caused a regression.',
    )


class PromptRevisionOutput(RichBaseModel):
    """Output model for the optimizer agent."""

    revised_prompt: str = Field(description='The improved system prompt.')

    reasoning: str = Field(description='Explanation of what changes were made and why.')


class SingleCandidateOutput(RichBaseModel):
    """A single prompt candidate within a beam search."""

    revised_prompt: str = Field(description='The improved system prompt.')

    reasoning: str = Field(
        description='Explanation of what changes were made and why for this specific candidate.'
    )


class MultiPromptRevisionOutput(RichBaseModel):
    """Output model for beam search generating multiple candidates."""

    candidates: List[SingleCandidateOutput] = Field(
        description='List of candidate prompts with their reasoning.'
    )


OPTIMIZER_EXAMPLES = """
## Few-Shot Examples

### Example 1: Unsupported Claims → Add Sourcing Constraint
**Failure Signal**: "Claim 'The product was released in 2019' is not supported by the context"
**Root Cause**: The AI generated information not present in the provided context.
**Fix Applied**: Added "Only make claims that are explicitly stated in or directly inferable from the provided context. If information is not available, say so."

### Example 2: Incomplete Responses → Add Coverage Instruction
**Failure Signal**: "Response does not address the user's question about pricing"
**Root Cause**: The AI addressed some aspects but missed specific sub-questions.
**Fix Applied**: Added "Before responding, identify ALL questions or sub-questions in the user's query. Ensure your response addresses each one explicitly."

### Example 3: Tone Issues → Add Style Constraints
**Failure Signal**: "Response uses casual language inappropriate for professional context"
**Root Cause**: No guidance on expected communication style.
**Fix Applied**: Added "Maintain a professional, clear, and respectful tone. Avoid slang, casual expressions, or overly familiar language."

### Example 4: Hallucination → Add Explicit Fallback
**Failure Signal**: "Response contains fabricated statistics not in the source material"
**Root Cause**: AI invented plausible-sounding but false details.
**Fix Applied**: Added "If specific data, statistics, or facts are not available in the context, explicitly state 'This information is not available in the provided materials' rather than generating approximate values."
"""


OPTIMIZER_INSTRUCTION = (
    """You are an expert Prompt Engineer specializing in optimizing system prompts for AI applications. Your task is to analyze evaluation failures and generate improved prompts that prevent these errors.

## Your Goal
Improve the system prompt so that the AI produces responses that pass all evaluation metrics. Focus on:
1. **Faithfulness**: Ensure responses only contain claims that are supported by the provided context
2. **Relevancy**: Ensure responses directly address all aspects of the user's question
3. **Completeness**: Ensure responses cover all necessary information
4. **Accuracy**: Ensure factual correctness in all statements

## Analysis Process
1. Carefully read the failure cases and their metric explanations
2. Identify the ROOT CAUSE of each failure (not just the symptom)
3. Determine what instruction or constraint would have prevented the error
4. Add specific, actionable guidance to the prompt

## Revision Guidelines
- Be SPECIFIC: Instead of "be accurate", specify "verify all facts against the provided context"
- Be ACTIONABLE: Give clear instructions that can be followed
- Be CONSERVATIVE: Make targeted changes, don't rewrite everything
- PRESERVE: Keep working instructions from the original prompt
- ADD CONSTRAINTS: If the AI is making up information, add "Only use information from the provided context"
- ADD CHECKS: If responses are incomplete, add "Ensure your response addresses all parts of the question"

## Common Failure Patterns and Fixes
- **Claims not supported**: Add "Only make statements that are directly supported by the context"
- **Missing information**: Add "Check that your response covers [specific aspects]"
- **Irrelevant content**: Add "Focus only on what the user asked"
- **Contradictions**: Add "Verify your statements are consistent with each other"

## Output Format
Provide the complete revised prompt (not a diff) and explain your reasoning.
The revised prompt should be production-ready and self-contained."""
    + OPTIMIZER_EXAMPLES
)


OPTIMIZER_INSTRUCTION_BEAM_SUFFIX = """

## Multiple Candidate Generation Mode
You are in BEAM SEARCH mode. Instead of generating a single revised prompt, generate {beam_width} DISTINCT candidate prompts.

Each candidate should:
1. Address the same failure patterns, but with DIFFERENT approaches
2. Vary in specificity (e.g., one more general, one more specific)
3. Vary in strategy (e.g., one adds constraints, one restructures)
4. Be a complete, production-ready prompt

The candidates will be evaluated and the best one selected. Provide diverse options!
"""


class OptimizerAgent(LLMHandler[PromptRevisionInput, PromptRevisionOutput]):
    """
    LLM-based agent that generates improved prompts based on failure analysis.

    This agent uses the LLMHandler pattern to generate structured outputs,
    taking failure analysis and producing revised prompts with reasoning.
    """

    instruction = OPTIMIZER_INSTRUCTION
    input_model = PromptRevisionInput
    output_model = PromptRevisionOutput

    def __init__(
        self,
        llm: LLMRunnable,
        tracer: Optional[BaseTraceHandler] = None,
        **kwargs,
    ):
        """
        Initialize the optimizer agent.

        Args:
            llm: The LLM to use for generating prompt revisions.
            tracer: Optional tracer for observability.
            **kwargs: Additional arguments passed to LLMHandler.
        """
        self.tracer = init_tracer(
            'llm',
            ToolMetadata(
                name='OptimizerAgent',
                description='Generates improved prompts based on evaluation feedback',
                owner='AXION',
                version='1.0.0',
            ),
            tracer,
        )

        super().__init__(llm=llm, tracer=self.tracer, **kwargs)

    def get_instruction(self, input_data: PromptRevisionInput) -> str:
        """
        Generate dynamic instruction including context about the optimization state.

        Args:
            input_data: The input containing current state and failure analysis.

        Returns:
            The full instruction string for the LLM.
        """
        base_instruction = self.instruction

        # Add context about the current optimization state
        context = f"""
## Current Optimization State
- **Iteration**: {input_data.iteration}
- **Current Pass Rate**: {input_data.pass_rate:.1%}
- **Best Pass Rate**: {input_data.best_pass_rate:.1%}
"""

        # Add regression warning if applicable
        if input_data.regression_warning:
            context += f"""
## ⚠️ REGRESSION WARNING
{input_data.regression_warning}

Your previous changes CAUSED A REGRESSION. You must:
1. Revert problematic changes
2. Take a different approach
3. Make more conservative edits
"""

        # Add the current prompt section
        context += f"""
## Current System Prompt (To Improve)
```
{input_data.current_prompt}
```

## Failure Analysis
The following test cases failed evaluation. Study the errors carefully.

{input_data.failure_analysis}
"""

        return base_instruction + context

    async def execute_beam(
        self, input_data: PromptRevisionInput, beam_width: int
    ) -> MultiPromptRevisionOutput:
        """
        Generate multiple candidate prompts using beam search.

        This method generates `beam_width` distinct prompt candidates
        that will be evaluated via mini-eval to select the best one.

        Args:
            input_data: The input containing current state and failure analysis.
            beam_width: Number of candidate prompts to generate.

        Returns:
            MultiPromptRevisionOutput containing the list of candidates.
        """
        # Build instruction with beam suffix
        base_instruction = self.get_instruction(input_data)
        beam_instruction = base_instruction + OPTIMIZER_INSTRUCTION_BEAM_SUFFIX.format(
            beam_width=beam_width
        )

        # Use the LLM directly with the multi-output model
        messages = [
            {'role': 'system', 'content': beam_instruction},
            {'role': 'user', 'content': 'Generate the candidate prompts now.'},
        ]

        result = await self.llm.acompletion(
            messages=messages,
            response_model=MultiPromptRevisionOutput,
        )

        return result


def create_optimizer_agent(
    llm: LLMRunnable,
    model: Optional[str] = None,
    tracer: Optional[BaseTraceHandler] = None,
) -> OptimizerAgent:
    """
    Factory function to create an optimizer agent.

    Args:
        llm: The LLM to use for generating revisions.
        model: Optional model override (if different from llm's default).
        tracer: Optional tracer for observability.

    Returns:
        Configured OptimizerAgent instance.
    """
    return OptimizerAgent(llm=llm, tracer=tracer)
