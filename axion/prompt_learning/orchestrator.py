"""
Prompt Learning Orchestrator - Main workflow coordinator.

This module provides the PromptLearningOrchestrator class which coordinates
the evaluate → analyze → optimize loop for iterative prompt improvement.
"""

import copy
from typing import Callable, List, Optional

from axion._core.logging import get_logger
from axion._core.metadata.schema import ToolMetadata
from axion._core.schema import LLMRunnable
from axion._core.tracing import init_tracer, trace
from axion._core.tracing.handlers import BaseTraceHandler
from axion.prompt_learning.optimizer import OptimizerAgent, PromptRevisionInput
from axion.prompt_learning.runner_wrapper import DynamicPromptTask
from axion.prompt_learning.schema import (
    IterationRecord,
    OptimizationState,
    PromptLearningResult,
    PromptOptimizationConfig,
)
from axion.prompt_learning.signals import SignalFormatter
from axion.runners import EvaluationConfig, EvaluationRunner
from axion.schema import EvaluationResult, TestResult

logger = get_logger(__name__)


class PromptLearningOrchestrator:
    """
    Orchestrates the prompt optimization workflow.

    This class coordinates the iterative process of:
    1. Evaluating a prompt with the configured metrics
    2. Analyzing failures to identify hard negatives
    3. Generating improved prompts using an LLM optimizer
    4. Repeating until convergence or max iterations

    Example usage:
        ```python
        from axion import EvaluationConfig
        from axion.metrics import Faithfulness, AnswerRelevancy
        from axion.prompt_learning import PromptLearningOrchestrator, PromptOptimizationConfig

        # Define your task
        async def generate_response(item, system_prompt):
            response = await llm.acompletion(
                model='gpt-4o',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': item.query}
                ]
            )
            return {'actual_output': response.choices[0].message.content}

        # Configure
        eval_config = EvaluationConfig(
            evaluation_name='optimization',
            evaluation_inputs=dataset,
            scoring_metrics=[Faithfulness(), AnswerRelevancy()],
        )

        opt_config = PromptOptimizationConfig(
            target_pass_rate=0.95,
            max_iterations=5,
        )

        # Run optimization
        orchestrator = PromptLearningOrchestrator(
            evaluation_config=eval_config,
            optimization_config=opt_config,
            task=generate_response,
            llm=llm,
        )

        result = await orchestrator.optimize("You are a helpful assistant.")
        print(f"Best prompt: {result.best_prompt}")
        ```
    """

    def __init__(
        self,
        evaluation_config: EvaluationConfig,
        optimization_config: PromptOptimizationConfig,
        task: Callable,
        llm: LLMRunnable,
        tracer: Optional[BaseTraceHandler] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            evaluation_config: Configuration for running evaluations (metrics, dataset, etc.)
                              Note: The 'task' field will be overridden by the orchestrator.
            optimization_config: Configuration for the optimization process.
            task: The task function that generates responses. Should have signature:
                  async def task(item: DatasetItem, system_prompt: str) -> dict
            llm: The LLM to use for the optimizer agent.
            tracer: Optional tracer for observability.
        """
        self.eval_config = evaluation_config
        self.opt_config = optimization_config
        self.llm = llm

        # Initialize tracer
        self.tracer = init_tracer(
            'base',
            ToolMetadata(
                name='PromptLearningOrchestrator',
                description='Orchestrates prompt optimization workflow',
                owner='AXION',
                version='1.0.0',
            ),
            tracer,
        )

        # Wrap the user's task for dynamic prompt injection
        self.task_wrapper = DynamicPromptTask(
            task, prompt_key=optimization_config.prompt_key
        )

        # Initialize the optimizer agent
        self.optimizer = OptimizerAgent(llm=llm, tracer=self.tracer)

    @trace(name='prompt_learning_optimize', capture_args=True, capture_result=True)
    async def optimize(self, initial_prompt: str) -> PromptLearningResult:
        """
        Run the prompt optimization workflow.

        Args:
            initial_prompt: The starting prompt to optimize.

        Returns:
            PromptLearningResult containing the optimized prompt and metrics.
        """
        # Initialize state
        state = OptimizationState(
            current_prompt=initial_prompt,
            best_prompt=initial_prompt,
        )

        logger.info(
            f'Starting Prompt Optimization (max iterations: {self.opt_config.max_iterations}, '
            f'target: {self.opt_config.target_pass_rate:.1%})'
        )

        # Main optimization loop
        for i in range(self.opt_config.max_iterations):
            logger.info(f'\n--- Iteration {i + 1}/{self.opt_config.max_iterations} ---')

            # 1. EVALUATE
            eval_result = await self._evaluate(state)

            # 2. ANALYZE
            state = self._analyze(state, eval_result)

            # Log progress
            logger.info(
                f'Pass rate: {state.pass_rate:.1%} '
                f'({state.num_passed}/{state.num_total} passed)'
            )

            # 3. CHECK CONVERGENCE
            if state.is_converged:
                logger.info(
                    f'Target pass rate achieved ({self.opt_config.target_pass_rate:.1%})'
                )
                break

            if i == self.opt_config.max_iterations - 1:
                logger.info('Max iterations reached.')
                break

            # 4. OPTIMIZE (generate new prompt)
            state = await self._optimize(state)

        # Build final result
        return PromptLearningResult(
            final_prompt=state.current_prompt,
            best_prompt=state.best_prompt,
            best_pass_rate=state.best_pass_rate,
            final_pass_rate=state.pass_rate,
            total_iterations=state.iteration + 1,
            history=state.history,
            converged=state.is_converged,
            target_pass_rate=self.opt_config.target_pass_rate,
        )

    @trace(name='evaluate_iteration')
    async def _evaluate(self, state: OptimizationState) -> EvaluationResult:
        """
        Run evaluation with the current prompt.

        Args:
            state: Current optimization state.

        Returns:
            EvaluationResult from running the evaluation.
        """
        logger.info(f'Evaluating prompt (length: {len(state.current_prompt)} chars)...')

        # Set the current prompt on the task wrapper
        self.task_wrapper.set_prompt(state.current_prompt)

        # Create a copy of the eval config with our wrapped task
        config = copy.copy(self.eval_config)
        config.task = self.task_wrapper
        config.evaluation_name = (
            f'{self.eval_config.evaluation_name}_iter_{state.iteration}'
        )

        # Run evaluation
        runner = EvaluationRunner(config, tracer=self.tracer)
        result = await runner.execute()

        return result

    def _analyze(
        self, state: OptimizationState, result: EvaluationResult
    ) -> OptimizationState:
        """
        Analyze evaluation results and update state.

        Calculates pass rate, identifies hard negatives (failures),
        and checks for convergence.

        Args:
            state: Current optimization state.
            result: Evaluation result from the latest iteration.

        Returns:
            Updated optimization state.
        """
        if not result or not result.results:
            state.pass_rate = 0.0
            state.num_passed = 0
            state.num_total = 0
            state.hard_negatives = []
            return state

        # Calculate pass rate
        # A test case passes if ALL its metrics pass
        passed = 0
        failures: List[TestResult] = []

        for test_result in result.results:
            all_passed = all(
                score.passed is True for score in test_result.score_results
            )
            if all_passed:
                passed += 1
            else:
                failures.append(test_result)

        total = len(result.results)
        state.pass_rate = passed / total if total > 0 else 0.0
        state.num_passed = passed
        state.num_total = total

        # Select hard negatives (worst failures)
        # Sort by number of failed metrics (more failures = harder)
        failures.sort(
            key=lambda x: sum(1 for s in x.score_results if s.passed is False),
            reverse=True,
        )
        state.hard_negatives = failures[: self.opt_config.hard_negative_batch_size]

        # Check if this is a new best
        if state.pass_rate > state.best_pass_rate:
            logger.info(f'New best pass rate: {state.pass_rate:.1%}')
            state.best_pass_rate = state.pass_rate
            state.best_prompt = state.current_prompt
        elif (
            state.pass_rate
            < state.best_pass_rate - self.opt_config.regression_tolerance
        ):
            logger.warning(
                f'Regression detected: {state.pass_rate:.1%} < {state.best_pass_rate:.1%}'
            )

        # Check convergence
        if state.pass_rate >= self.opt_config.target_pass_rate:
            state.is_converged = True

        # Record history
        state.history.append(
            IterationRecord(
                iteration=state.iteration,
                prompt=state.current_prompt,
                pass_rate=state.pass_rate,
                num_passed=state.num_passed,
                num_total=state.num_total,
            )
        )

        return state

    @trace(name='generate_revision')
    async def _optimize(self, state: OptimizationState) -> OptimizationState:
        """
        Generate a revised prompt using the optimizer agent.

        Args:
            state: Current optimization state with hard negatives.

        Returns:
            Updated state with new prompt and incremented iteration.
        """
        if state.is_converged or not state.hard_negatives:
            return state

        logger.info(f'Analyzing {len(state.hard_negatives)} failures...')

        # Format failure analysis
        failure_analysis = SignalFormatter.format_hard_negatives(
            state.hard_negatives, max_failures=self.opt_config.hard_negative_batch_size
        )

        # Check for regression warning
        regression_warning = None
        if len(state.history) > 0:
            prev_pass_rate = state.history[-1].pass_rate
            if state.pass_rate < prev_pass_rate - self.opt_config.regression_tolerance:
                regression_warning = (
                    f'Your previous change caused a regression: '
                    f'{prev_pass_rate:.1%} -> {state.pass_rate:.1%}. '
                    f'You must take a different approach.'
                )

        # Build input for optimizer
        optimizer_input = PromptRevisionInput(
            current_prompt=state.current_prompt,
            pass_rate=state.pass_rate,
            best_pass_rate=state.best_pass_rate,
            iteration=state.iteration,
            failure_analysis=failure_analysis,
            regression_warning=regression_warning,
        )

        # Generate revision
        try:
            revision_output = await self.optimizer.execute(optimizer_input)
            new_prompt = revision_output.revised_prompt

            logger.info(
                f'Generated revised prompt (reasoning: {revision_output.reasoning[:100]}...)'
            )

            # Update state
            state.current_prompt = new_prompt
            state.iteration += 1

        except Exception as e:
            logger.error(f'Failed to generate revision: {e}')
            state.error = str(e)
            state.iteration += 1

        return state


async def optimize_prompt(
    initial_prompt: str,
    evaluation_config: EvaluationConfig,
    task: Callable,
    llm: LLMRunnable,
    target_pass_rate: float = 1.0,
    max_iterations: int = 5,
    hard_negative_batch_size: int = 3,
    tracer: Optional[BaseTraceHandler] = None,
) -> PromptLearningResult:
    """
    Convenience function to run prompt optimization.

    Args:
        initial_prompt: The starting prompt to optimize.
        evaluation_config: Configuration for running evaluations.
        task: The task function that generates responses.
        llm: The LLM to use for the optimizer agent.
        target_pass_rate: Target pass rate to achieve (default: 1.0).
        max_iterations: Maximum optimization iterations (default: 5).
        hard_negative_batch_size: Number of failures to analyze per iteration.
        tracer: Optional tracer for observability.

    Returns:
        PromptLearningResult containing the optimized prompt and metrics.
    """
    opt_config = PromptOptimizationConfig(
        target_pass_rate=target_pass_rate,
        max_iterations=max_iterations,
        hard_negative_batch_size=hard_negative_batch_size,
    )

    orchestrator = PromptLearningOrchestrator(
        evaluation_config=evaluation_config,
        optimization_config=opt_config,
        task=task,
        llm=llm,
        tracer=tracer,
    )

    return await orchestrator.optimize(initial_prompt)
