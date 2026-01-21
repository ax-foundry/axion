"""
Pydantic models for the Prompt Learning module.

This module defines the data structures used throughout the prompt optimization workflow,
including configuration, state tracking, iteration history, and final results.
"""

from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from pydantic import Field

from axion._core.schema import RichBaseModel


def _strftime() -> str:
    """Generate ISO timestamp string."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


class BeamCandidate(RichBaseModel):
    """
    Represents a candidate prompt in beam search.

    During beam search, multiple candidate prompts are generated and
    evaluated with a mini-eval to select the most promising one.
    """

    prompt: str = Field(
        description='The candidate prompt text.',
    )

    reasoning: str = Field(
        description='Explanation of what changes were made and why.',
    )

    mini_eval_pass_rate: Optional[float] = Field(
        default=None,
        description='Pass rate from mini-evaluation (if performed).',
    )

    rank: Optional[int] = Field(
        default=None,
        description='Rank after mini-evaluation (1 = best).',
    )


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

    # Beam search configuration
    beam_width: int = Field(
        default=1,
        ge=1,
        le=10,
        description='Number of candidate prompts to generate per iteration. '
        'beam_width=1 means single candidate (original behavior).',
    )

    mini_eval_sample_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description='Number of samples to use for mini-evaluation when beam_width > 1.',
    )

    mini_eval_sample_strategy: Literal['random', 'hard_negatives', 'stratified'] = (
        Field(
            default='hard_negatives',
            description='Strategy for selecting mini-eval samples: '
            'random (uniform), hard_negatives (focus on failures), '
            'stratified (balanced mix of pass/fail).',
        )
    )

    # Auto-revert configuration
    auto_revert_on_regression: bool = Field(
        default=True,
        description='Whether to automatically revert to best prompt after consecutive regressions.',
    )

    max_consecutive_regressions: int = Field(
        default=2,
        ge=1,
        le=5,
        description='Number of consecutive regressions before auto-reverting to best prompt.',
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

    # Beam search fields
    beam_candidates: Optional[List['BeamCandidate']] = Field(
        default=None,
        description='All candidates considered during beam search (if beam_width > 1).',
    )

    selected_candidate_index: Optional[int] = Field(
        default=None,
        description='Index of the selected candidate in beam_candidates (if beam_width > 1).',
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

    # Auto-revert state tracking
    consecutive_regressions: int = Field(
        default=0,
        description='Number of consecutive iterations with regression.',
    )

    reverted_from_iteration: Optional[int] = Field(
        default=None,
        description='If auto-reverted, the iteration we reverted from.',
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
