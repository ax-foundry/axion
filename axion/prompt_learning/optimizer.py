"""
Optimizer Agent for generating revised prompts based on evaluation feedback.

This module implements the "brain" of the prompt learning system - an LLM-based
agent that analyzes failures and generates improved prompt revisions.
"""

from typing import Optional

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


OPTIMIZER_INSTRUCTION = """You are an expert Prompt Engineer specializing in optimizing system prompts for AI applications. Your task is to analyze evaluation failures and generate improved prompts that prevent these errors.

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
