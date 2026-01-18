# Prompt Learning

The **Prompt Learning** module provides tools for iteratively optimizing prompts using evaluation feedback. It implements an active learning approach where metric failures guide prompt improvements through an LLM-based optimizer.

## Overview

Prompt Learning uses English-language feedback from evaluation metrics to automatically improve system prompts. Instead of manual trial-and-error, the module:

1. **Evaluates** your prompt against a dataset using Axion metrics
2. **Analyzes** failures to extract actionable feedback from metric signals
3. **Optimizes** by generating improved prompts using an LLM agent
4. **Repeats** until the target pass rate is achieved or max iterations reached

## Installation

The prompt learning module is included in Axion. No additional installation required.

```python
from axion.prompt_learning import (
    PromptLearningOrchestrator,
    PromptOptimizationConfig,
    optimize_prompt,
)
```

## Quick Start

### Basic Usage

```python
import asyncio
from llama_index.llms.openai import OpenAI
from axion import Dataset, DatasetItem, EvaluationConfig
from axion.metrics import Faithfulness, AnswerRelevancy
from axion.prompt_learning import PromptLearningOrchestrator, PromptOptimizationConfig

# 1. Define your task function
async def generate_response(item: DatasetItem, system_prompt: str) -> dict:
    """
    Your task function must:
    - Accept a DatasetItem as the first argument
    - Accept system_prompt as a keyword argument
    - Return a dict with 'actual_output' key
    """
    llm = OpenAI(model='gpt-4o-mini')

    prompt = f"{system_prompt}\n\nContext: {item.retrieved_content}\n\nQuestion: {item.query}"
    response = await llm.acomplete(prompt)

    return {'actual_output': response.text}

# 2. Create your dataset
dataset = Dataset(
    name='my_dataset',
    items=[
        DatasetItem(
            id='1',
            query='What is the capital of France?',
            expected_output='Paris is the capital of France.',
            retrieved_content=['Paris is the capital and largest city of France.'],  # List of strings
        ),
        # ... more items
    ]
)

# 3. Configure evaluation
eval_config = EvaluationConfig(
    evaluation_name='prompt_optimization',
    evaluation_inputs=dataset,
    scoring_metrics=[Faithfulness(), AnswerRelevancy()],
    thresholds={'Faithfulness': 0.7, 'AnswerRelevancy': 0.7},
)

# 4. Configure optimization
opt_config = PromptOptimizationConfig(
    target_pass_rate=0.95,      # Stop when 95% of tests pass
    max_iterations=5,           # Maximum optimization rounds
    hard_negative_batch_size=3, # Failures to analyze per iteration
)

# 5. Run optimization
async def main():
    llm = OpenAI(model='gpt-4o')  # LLM for the optimizer agent

    orchestrator = PromptLearningOrchestrator(
        evaluation_config=eval_config,
        optimization_config=opt_config,
        task=generate_response,
        llm=llm,
    )

    result = await orchestrator.optimize("You are a helpful assistant.")

    print(f"Best prompt: {result.best_prompt}")
    print(f"Best pass rate: {result.best_pass_rate:.1%}")
    print(f"Converged: {result.converged}")

asyncio.run(main())
```

### Using the Convenience Function

For simpler use cases:

```python
from axion.prompt_learning import optimize_prompt

result = await optimize_prompt(
    initial_prompt="You are a helpful assistant.",
    evaluation_config=eval_config,
    task=generate_response,
    llm=llm,
    target_pass_rate=0.95,
    max_iterations=5,
)
```

## API Reference

### PromptLearningOrchestrator

The main class that coordinates the optimization workflow.

```python
class PromptLearningOrchestrator:
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
            evaluation_config: Configuration for running evaluations
            optimization_config: Configuration for optimization process
            task: Task function (see Task Function Signature below)
            llm: LLM for the optimizer agent (e.g., GPT-4)
            tracer: Optional tracer for observability
        """

    async def optimize(self, initial_prompt: str) -> PromptLearningResult:
        """
        Run the optimization workflow.

        Args:
            initial_prompt: Starting prompt to optimize

        Returns:
            PromptLearningResult with optimized prompt and metrics
        """
```

### PromptOptimizationConfig

Configuration for the optimization process.

```python
class PromptOptimizationConfig:
    target_pass_rate: float = 1.0
    """Target pass rate to achieve (0.0 to 1.0). Optimization stops when reached."""

    max_iterations: int = 5
    """Maximum number of optimization iterations."""

    hard_negative_batch_size: int = 3
    """Number of failure cases to analyze per iteration."""

    optimizer_model: str = 'gpt-4o'
    """Model to use for generating prompt revisions."""

    regression_tolerance: float = 0.05
    """Allowed regression before triggering warning."""

    prompt_key: str = 'system_prompt'
    """Keyword argument name for injecting prompt into task."""

    show_progress: bool = True
    """Whether to show progress during optimization."""
```

### PromptLearningResult

The result of an optimization run.

```python
class PromptLearningResult:
    final_prompt: str
    """The prompt from the last iteration."""

    best_prompt: str
    """The best performing prompt found during optimization."""

    best_pass_rate: float
    """Highest pass rate achieved."""

    final_pass_rate: float
    """Pass rate from the final iteration."""

    total_iterations: int
    """Total number of iterations completed."""

    history: List[IterationRecord]
    """Complete history of all iterations."""

    converged: bool
    """Whether optimization reached the target pass rate."""

    target_pass_rate: float
    """The target pass rate that was configured."""

    def summary(self) -> str:
        """Generate a human-readable summary."""
```

### Task Function Signature

Your task function must follow this signature:

```python
async def my_task(item: DatasetItem, system_prompt: str) -> Dict[str, Any]:
    """
    Args:
        item: DatasetItem with query, retrieved_content, etc.
        system_prompt: The prompt being optimized (injected by orchestrator)

    Returns:
        Dict containing at minimum: {'actual_output': str}
    """
```

The `system_prompt` parameter name can be customized via `PromptOptimizationConfig.prompt_key`.

## How It Works

### The Optimization Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION LOOP                         │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ EVALUATE │ -> │ ANALYZE  │ -> │ OPTIMIZE │ -> [REPEAT]  │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │                     │
│       v               v               v                     │
│  Run metrics     Find failures   Generate new              │
│  on dataset      & extract       prompt via                │
│                  signals         LLM agent                  │
│                                                             │
│  Exit when: target_pass_rate reached OR max_iterations      │
└─────────────────────────────────────────────────────────────┘
```

### Signal Extraction

The module extracts actionable feedback from metric signals:

- **Faithfulness**: Identifies claims not supported by context
- **AnswerRelevancy**: Identifies missing aspects
- **Completeness**: Identifies missing information
- **Other metrics**: Extracts score and explanation

Example feedback sent to the optimizer:

```
--- Failure 1 ---
Query: What is the capital of France?
Output: Paris is a beautiful city in Europe.

Metric 'Faithfulness' (score: 0.50):
  - [CLAIM NOT SUPPORTED] 'Paris is a beautiful city'
    Verdict: no_evidence
    Reason: Context does not mention beauty
  Verdict summary: no_evidence: 1
```

### The Optimizer Agent

The optimizer is an LLM agent (using `LLMHandler`) that:

1. Receives the current prompt and failure analysis
2. Identifies root causes of failures
3. Generates specific improvements
4. Returns a revised prompt with reasoning

The optimizer is instructed to:
- Be specific (not "be accurate" but "verify facts against context")
- Be conservative (targeted changes, not complete rewrites)
- Preserve working instructions
- Add constraints to prevent specific failure patterns

## Advanced Usage

### Custom Prompt Key

If your task uses a different parameter name:

```python
async def my_task(item: DatasetItem, instructions: str) -> dict:
    # Uses 'instructions' instead of 'system_prompt'
    ...

opt_config = PromptOptimizationConfig(
    prompt_key='instructions',  # Custom key
    ...
)
```

### With Tracing

Enable observability with Logfire or Langfuse:

```python
from axion._core.tracing import Tracer

tracer = Tracer(mode='logfire')

orchestrator = PromptLearningOrchestrator(
    evaluation_config=eval_config,
    optimization_config=opt_config,
    task=generate_response,
    llm=llm,
    tracer=tracer,  # Enable tracing
)
```

### Accessing Iteration History

```python
result = await orchestrator.optimize(initial_prompt)

for record in result.history:
    print(f"Iteration {record.iteration + 1}:")
    print(f"  Pass rate: {record.pass_rate:.1%}")
    print(f"  Passed: {record.num_passed}/{record.num_total}")
    print(f"  Prompt length: {len(record.prompt)} chars")
```

### Custom Signal Formatting

For custom metrics, you can use `SignalFormatter` directly:

```python
from axion.prompt_learning import SignalFormatter

# Format a single metric failure
feedback = SignalFormatter.format_metric_failure(metric_score)

# Format a test result with all failures
feedback = SignalFormatter.format_test_result_failures(test_result)

# Format multiple hard negatives
feedback = SignalFormatter.format_hard_negatives(
    hard_negatives,
    max_failures=5
)
```

## Best Practices

### 1. Dataset Quality

- Include diverse examples that cover edge cases
- Ensure `retrieved_content` is realistic
- Set appropriate `expected_output` values

### 2. Metric Selection

- Start with `Faithfulness` and `AnswerRelevancy`
- Add domain-specific metrics as needed
- Set realistic thresholds

### 3. Configuration

- Start with `target_pass_rate=0.8` and increase gradually
- Use `max_iterations=3-5` for most cases
- Set `hard_negative_batch_size=3-5` for focused feedback

### 4. Initial Prompt

- Start with a simple, clear prompt
- Don't over-engineer the initial prompt
- Let the optimizer add specific constraints

## Troubleshooting

### Low Pass Rates

- Check if your dataset has realistic expected outputs
- Verify retrieved_content is relevant to queries
- Lower thresholds if metrics are too strict

### Optimizer Not Improving

- Increase `hard_negative_batch_size` for more feedback
- Check if failures have clear patterns
- Try a more capable optimizer model (gpt-4o)

### Regression After Optimization

- The module tracks regressions and warns the optimizer
- If persistent, try more conservative `regression_tolerance`
- Review the optimizer's reasoning in logs

## Examples

See the full example at:
```
examples/prompt_learning_example.py
```

Run with:
```bash
# Real optimization (requires OPENAI_API_KEY)
python examples/prompt_learning_example.py

# Mock optimization (no API calls for optimizer)
python examples/prompt_learning_example.py --mock
```
