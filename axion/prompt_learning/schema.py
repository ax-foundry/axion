"""
Pydantic models for the Prompt Learning module.

This module defines the data structures used throughout the prompt optimization workflow,
including configuration, state tracking, iteration history, and final results.
"""

from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import Field

from axion._core.schema import RichBaseModel


def _strftime() -> str:
    """Generate ISO timestamp string."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


class PromptOptimizationConfig(RichBaseModel):
    """
    Configuration for prompt optimization workflow.

    Controls the optimization loop parameters including convergence criteria,
    iteration limits, and the model used for generating prompt revisions.
    """

    target_pass_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description='Target pass rate to achieve before stopping optimization.',
    )

    max_iterations: int = Field(
        default=5,
        ge=1,
        le=50,
        description='Maximum number of optimization iterations.',
    )

    hard_negative_batch_size: int = Field(
        default=3,
        ge=1,
        le=20,
        description='Number of failure cases to analyze per iteration.',
    )

    optimizer_model: str = Field(
        default='gpt-4o',
        description='Model to use for generating prompt revisions.',
    )

    regression_tolerance: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description='Allowed regression in pass rate before triggering rollback warning.',
    )

    show_progress: bool = Field(
        default=True,
        description='Whether to show progress during optimization.',
    )

    prompt_key: str = Field(
        default='system_prompt',
        description='Keyword argument name for injecting prompt into task function.',
    )


class IterationRecord(RichBaseModel):
    """
    Record of a single optimization iteration.

    Captures the prompt, pass rate, and timestamp for each iteration,
    enabling analysis of the optimization trajectory.
    """

    iteration: int = Field(
        description='Iteration number (0-indexed).',
    )

    prompt: str = Field(
        description='The prompt used in this iteration.',
    )

    pass_rate: float = Field(
        description='Pass rate achieved with this prompt.',
    )

    num_passed: int = Field(
        default=0,
        description='Number of test cases that passed.',
    )

    num_total: int = Field(
        default=0,
        description='Total number of test cases.',
    )

    timestamp: str = Field(
        default_factory=_strftime,
        description='Timestamp when this iteration completed.',
    )


class OptimizationState(RichBaseModel):
    """
    Tracks the current state of the optimization workflow.

    This model maintains all state needed across iterations, including
    the current and best prompts, metrics, and convergence status.
    """

    iteration: int = Field(
        default=0,
        description='Current iteration number.',
    )

    current_prompt: str = Field(
        description='The prompt being evaluated in the current iteration.',
    )

    best_prompt: str = Field(
        description='The best performing prompt found so far.',
    )

    best_pass_rate: float = Field(
        default=-1.0,
        description='Highest pass rate achieved so far.',
    )

    pass_rate: float = Field(
        default=0.0,
        description='Pass rate from the latest evaluation.',
    )

    num_passed: int = Field(
        default=0,
        description='Number of test cases that passed in latest evaluation.',
    )

    num_total: int = Field(
        default=0,
        description='Total number of test cases in latest evaluation.',
    )

    hard_negatives: List[Any] = Field(
        default_factory=list,
        description='Test results for cases that failed (used for optimization feedback).',
    )

    history: List[IterationRecord] = Field(
        default_factory=list,
        description='History of all iterations.',
    )

    is_converged: bool = Field(
        default=False,
        description='Whether the optimization has converged.',
    )

    error: Optional[str] = Field(
        default=None,
        description='Error message if optimization failed.',
    )


class PromptLearningResult(RichBaseModel):
    """
    Final result of prompt optimization workflow.

    Contains the optimized prompt, performance metrics, and full history
    of the optimization process.
    """

    final_prompt: str = Field(
        description='The prompt from the last iteration.',
    )

    best_prompt: str = Field(
        description='The best performing prompt found during optimization.',
    )

    best_pass_rate: float = Field(
        description='Highest pass rate achieved.',
    )

    final_pass_rate: float = Field(
        description='Pass rate from the final iteration.',
    )

    total_iterations: int = Field(
        description='Total number of iterations completed.',
    )

    history: List[IterationRecord] = Field(
        default_factory=list,
        description='Complete history of all iterations.',
    )

    converged: bool = Field(
        default=False,
        description='Whether optimization reached the target pass rate.',
    )

    target_pass_rate: float = Field(
        description='The target pass rate that was configured.',
    )

    def summary(self) -> str:
        """Generate a human-readable summary of the optimization result."""
        status = 'Converged' if self.converged else 'Did not converge'
        return (
            f'{status} after {self.total_iterations} iterations.\n'
            f'Best pass rate: {self.best_pass_rate:.1%}\n'
            f'Final pass rate: {self.final_pass_rate:.1%}\n'
            f'Target: {self.target_pass_rate:.1%}'
        )
