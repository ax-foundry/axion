"""
Prompt Learning Orchestrator - Main workflow coordinator.

This module provides the PromptLearningOrchestrator class which coordinates
the evaluate → analyze → optimize loop for iterative prompt improvement.
"""

import copy
import random
from typing import Callable, List, Optional

from axion._core.logging import get_logger
from axion._core.metadata.schema import ToolMetadata
from axion._core.schema import LLMRunnable
from axion._core.tracing import init_tracer, trace
from axion._core.tracing.handlers import BaseTraceHandler
from axion.prompt_learning.optimizer import OptimizerAgent, PromptRevisionInput
from axion.prompt_learning.runner_wrapper import DynamicPromptTask
from axion.prompt_learning.schema import (
    BeamCandidate,
    IterationRecord,
    OptimizationState,
    PromptLearningResult,
    PromptOptimizationConfig,
)
from axion.prompt_learning.signals import SignalFormatter
from axion.runners import EvaluationConfig, EvaluationRunner
from axion.schema import DatasetItem, EvaluationResult, TestResult

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

    def _select_mini_eval_samples(
        self,
        state: OptimizationState,
        sample_size: int,
        strategy: str,
    ) -> List[DatasetItem]:
        """
        Select samples for mini-evaluation during beam search.

        Args:
            state: Current optimization state (contains hard_negatives from last eval).
            sample_size: Number of samples to select.
            strategy: Selection strategy ('random', 'hard_negatives', 'stratified').

        Returns:
            List of DatasetItem objects for mini-evaluation.
        """
        # Get the full dataset from eval config
        all_items = list(self.eval_config.evaluation_inputs)

        if len(all_items) <= sample_size:
            return all_items

        if strategy == 'random':
            return random.sample(all_items, sample_size)

        elif strategy == 'hard_negatives':
            # Prioritize items that failed in the last evaluation
            # Get IDs of failed items from hard_negatives
            failed_ids = set()
            for test_result in state.hard_negatives:
                if hasattr(test_result, 'item') and test_result.item:
                    if hasattr(test_result.item, 'id'):
                        failed_ids.add(test_result.item.id)

            # Separate failed items from passed items
            failed_items = [
                item for item in all_items if getattr(item, 'id', None) in failed_ids
            ]
            passed_items = [
                item
                for item in all_items
                if getattr(item, 'id', None) not in failed_ids
            ]

            # Take as many failed items as possible, fill rest with passed
            result = failed_items[:sample_size]
            remaining = sample_size - len(result)
            if remaining > 0 and passed_items:
                result.extend(
                    random.sample(passed_items, min(remaining, len(passed_items)))
                )

            return result

        elif strategy == 'stratified':
            # Balanced mix of pass/fail items
            failed_ids = set()
            for test_result in state.hard_negatives:
                if hasattr(test_result, 'item') and test_result.item:
                    if hasattr(test_result.item, 'id'):
                        failed_ids.add(test_result.item.id)

            failed_items = [
                item for item in all_items if getattr(item, 'id', None) in failed_ids
            ]
            passed_items = [
                item
                for item in all_items
                if getattr(item, 'id', None) not in failed_ids
            ]

            # Take half from each category
            half_size = sample_size // 2
            result = []

            if failed_items:
                result.extend(
                    random.sample(failed_items, min(half_size, len(failed_items)))
                )
            if passed_items:
                remaining = sample_size - len(result)
                result.extend(
                    random.sample(passed_items, min(remaining, len(passed_items)))
                )

            return result

        else:
            # Fallback to random
            return random.sample(all_items, sample_size)

    @trace(name='mini_evaluate_candidates')
    async def _mini_evaluate(
        self,
        candidates: List[BeamCandidate],
        sample_items: List[DatasetItem],
    ) -> List[BeamCandidate]:
        """
        Perform mini-evaluation on beam search candidates.

        Runs a quick evaluation on a subset of data for each candidate
        to determine which one performs best.

        Args:
            candidates: List of BeamCandidate objects with prompt text.
            sample_items: Subset of dataset items to evaluate on.

        Returns:
            List of BeamCandidate objects with mini_eval_pass_rate and rank filled in.
        """
        from axion import Dataset

        logger.info(
            f'Mini-evaluating {len(candidates)} candidates on {len(sample_items)} samples...'
        )

        for idx, candidate in enumerate(candidates):
            # Set the candidate prompt
            self.task_wrapper.set_prompt(candidate.prompt)

            # Create mini-eval config
            mini_config = copy.copy(self.eval_config)
            mini_config.task = self.task_wrapper
            mini_config.evaluation_inputs = Dataset(items=sample_items)
            mini_config.evaluation_name = f'mini_eval_candidate_{idx}'

            # Run mini evaluation
            runner = EvaluationRunner(mini_config, tracer=self.tracer)
            result = await runner.execute()

            # Calculate pass rate
            if result and result.results:
                passed = sum(
                    1
                    for test_result in result.results
                    if all(score.passed is True for score in test_result.score_results)
                )
                candidate.mini_eval_pass_rate = passed / len(result.results)
            else:
                candidate.mini_eval_pass_rate = 0.0

            logger.info(
                f'  Candidate {idx + 1}: {candidate.mini_eval_pass_rate:.1%} pass rate'
            )

        # Rank candidates by mini_eval_pass_rate (descending)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.mini_eval_pass_rate or 0.0,
            reverse=True,
        )

        for rank, candidate in enumerate(sorted_candidates, start=1):
            candidate.rank = rank

        return candidates

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
            # Reset consecutive regressions on improvement
            state.consecutive_regressions = 0
        elif (
            state.pass_rate
            < state.best_pass_rate - self.opt_config.regression_tolerance
        ):
            state.consecutive_regressions += 1
            logger.warning(
                f'Regression detected: {state.pass_rate:.1%} < {state.best_pass_rate:.1%} '
                f'(consecutive: {state.consecutive_regressions})'
            )

            # Auto-revert if enabled and threshold reached
            if (
                self.opt_config.auto_revert_on_regression
                and state.consecutive_regressions
                >= self.opt_config.max_consecutive_regressions
            ):
                logger.warning(
                    f'Auto-reverting to best prompt after {state.consecutive_regressions} '
                    f'consecutive regressions'
                )
                state.reverted_from_iteration = state.iteration
                state.current_prompt = state.best_prompt
                state.consecutive_regressions = 0
        else:
            # No regression (stayed same or within tolerance)
            state.consecutive_regressions = 0

        # Check convergence
        if state.pass_rate >= self.opt_config.target_pass_rate:
            state.is_converged = True

        # Record history
        iteration_record = IterationRecord(
            iteration=state.iteration,
            prompt=state.current_prompt,
            pass_rate=state.pass_rate,
            num_passed=state.num_passed,
            num_total=state.num_total,
        )

        # Add beam search info if available (from previous _optimize_beam call)
        if hasattr(state, '_beam_candidates') and state._beam_candidates is not None:  # type: ignore
            iteration_record.beam_candidates = state._beam_candidates  # type: ignore
            iteration_record.selected_candidate_index = state._selected_candidate_index  # type: ignore
            # Clear temporary attributes
            del state._beam_candidates  # type: ignore
            del state._selected_candidate_index  # type: ignore

        state.history.append(iteration_record)

        return state

    @trace(name='generate_revision')
    async def _optimize(self, state: OptimizationState) -> OptimizationState:
        """
        Generate a revised prompt using the optimizer agent.

        Supports beam search when beam_width > 1, generating multiple candidates
        and selecting the best via mini-evaluation.

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

        # Check for regression warning (including auto-revert context)
        regression_warning = None
        if len(state.history) > 0:
            prev_pass_rate = state.history[-1].pass_rate
            if state.pass_rate < prev_pass_rate - self.opt_config.regression_tolerance:
                regression_warning = (
                    f'Your previous change caused a regression: '
                    f'{prev_pass_rate:.1%} -> {state.pass_rate:.1%}. '
                    f'You must take a different approach.'
                )
                # Add auto-revert context if applicable
                if state.reverted_from_iteration is not None:
                    regression_warning += (
                        f' Note: The system auto-reverted to the best prompt '
                        f'from iteration {state.reverted_from_iteration}.'
                    )
                    state.reverted_from_iteration = None  # Clear after use

        # Build input for optimizer
        optimizer_input = PromptRevisionInput(
            current_prompt=state.current_prompt,
            pass_rate=state.pass_rate,
            best_pass_rate=state.best_pass_rate,
            iteration=state.iteration,
            failure_analysis=failure_analysis,
            regression_warning=regression_warning,
        )

        # Generate revision(s)
        try:
            beam_width = self.opt_config.beam_width

            if beam_width > 1:
                # Beam search: generate multiple candidates
                state = await self._optimize_beam(state, optimizer_input, beam_width)
            else:
                # Single candidate (original behavior)
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

    @trace(name='optimize_beam_search')
    async def _optimize_beam(
        self,
        state: OptimizationState,
        optimizer_input: PromptRevisionInput,
        beam_width: int,
    ) -> OptimizationState:
        """
        Perform beam search optimization with multiple candidates.

        Generates multiple prompt candidates, evaluates them via mini-eval,
        and selects the best performing one.

        Args:
            state: Current optimization state.
            optimizer_input: Input for the optimizer containing failure analysis.
            beam_width: Number of candidates to generate.

        Returns:
            Updated state with the best candidate selected.
        """
        logger.info(f'Generating {beam_width} candidate prompts (beam search)...')

        # Generate multiple candidates
        multi_output = await self.optimizer.execute_beam(optimizer_input, beam_width)

        # Convert to BeamCandidate objects
        candidates = [
            BeamCandidate(
                prompt=candidate.revised_prompt,
                reasoning=candidate.reasoning,
            )
            for candidate in multi_output.candidates
        ]

        logger.info(f'Generated {len(candidates)} candidates')

        # Select samples for mini-evaluation
        sample_items = self._select_mini_eval_samples(
            state,
            self.opt_config.mini_eval_sample_size,
            self.opt_config.mini_eval_sample_strategy,
        )

        # Mini-evaluate all candidates
        candidates = await self._mini_evaluate(candidates, sample_items)

        # Select the best candidate (rank 1)
        best_candidate = min(candidates, key=lambda c: c.rank or float('inf'))
        selected_index = candidates.index(best_candidate)

        logger.info(
            f'Selected candidate {selected_index + 1} with '
            f'{best_candidate.mini_eval_pass_rate:.1%} mini-eval pass rate'
        )
        logger.info(f'Reasoning: {best_candidate.reasoning[:100]}...')

        # Update state with selected candidate
        state.current_prompt = best_candidate.prompt
        state.iteration += 1

        # Store beam search info in the last history record (will be added in next _analyze)
        # Note: The IterationRecord for this iteration will be created in the next _analyze call
        # We store the beam info temporarily on state to be picked up later
        state._beam_candidates = candidates  # type: ignore
        state._selected_candidate_index = selected_index  # type: ignore

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
