"""
Prompt Learning Module for Axion.

This module provides tools for iteratively optimizing prompts using
evaluation feedback. It uses an active learning approach where metric
failures guide prompt improvements through an LLM-based optimizer.

Example usage:
    ```python
    from axion import EvaluationConfig, Dataset
    from axion.metrics import Faithfulness, AnswerRelevancy
    from axion.prompt_learning import (
        PromptLearningOrchestrator,
        PromptOptimizationConfig,
        optimize_prompt,
    )

    # Simple usage with convenience function
    result = await optimize_prompt(
        initial_prompt="You are a helpful assistant.",
        evaluation_config=eval_config,
        task=my_task_function,
        llm=llm,
        target_pass_rate=0.95,
    )

    # Or use the orchestrator directly for more control
    orchestrator = PromptLearningOrchestrator(
        evaluation_config=eval_config,
        optimization_config=PromptOptimizationConfig(
            target_pass_rate=0.95,
            max_iterations=5,
        ),
        task=my_task_function,
        llm=llm,
    )
    result = await orchestrator.optimize("You are a helpful assistant.")
    ```
"""

from axion.prompt_learning.optimizer import (
    OptimizerAgent,
    PromptRevisionInput,
    PromptRevisionOutput,
    create_optimizer_agent,
)
from axion.prompt_learning.orchestrator import (
    PromptLearningOrchestrator,
    optimize_prompt,
)
from axion.prompt_learning.runner_wrapper import (
    DynamicPromptTask,
    PromptTemplateTask,
    wrap_task_for_optimization,
)
from axion.prompt_learning.schema import (
    IterationRecord,
    OptimizationState,
    PromptLearningResult,
    PromptOptimizationConfig,
)
from axion.prompt_learning.signals import SignalFormatter

__all__ = [
    # Main orchestrator
    'PromptLearningOrchestrator',
    'optimize_prompt',
    # Configuration
    'PromptOptimizationConfig',
    # Results
    'PromptLearningResult',
    'OptimizationState',
    'IterationRecord',
    # Optimizer agent
    'OptimizerAgent',
    'PromptRevisionInput',
    'PromptRevisionOutput',
    'create_optimizer_agent',
    # Task wrappers
    'DynamicPromptTask',
    'PromptTemplateTask',
    'wrap_task_for_optimization',
    # Utilities
    'SignalFormatter',
]
