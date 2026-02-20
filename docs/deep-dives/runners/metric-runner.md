---
icon: custom/play
---
# Metric Runner

The Metric Runner orchestrates evaluation metrics across multiple frameworks — Axion, Ragas, and DeepEval — through a single interface. It handles concurrency, caching, error recovery, and progress tracking so you can focus on choosing the right metrics.

## What You'll Learn

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Batch Processing</p>
<p class="rule-card__desc">Run metrics across datasets with configurable concurrency, progress bars, and automatic result collection.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">Multi-Framework</p>
<p class="rule-card__desc">Mix Axion, Ragas, and DeepEval metrics in a single run &mdash; each is automatically routed to the correct executor.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">Custom Runners</p>
<p class="rule-card__desc">Register your own metric framework integrations with a decorator or manual registration.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">4</span>
<p class="rule-card__title">Standardized Output</p>
<p class="rule-card__desc">Every runner returns <code>TestResult</code> / <code>MetricScore</code> objects regardless of the underlying framework.</p>
</div>
</div>

## Quick Start

```python title="View Available Runners"
from axion.runners import MetricRunner

# Display all registered Metric runners with their options
MetricRunner.display()
```

## Usage Patterns

=== ":material-play: Basic Batch"

    Pass a list of metrics and dataset items. The runner handles concurrency and
    returns a `TestResult` per item with all metric scores attached.

    ```python
    from axion.runners import MetricRunner
    from axion.metrics import AnswerRelevancy
    from axion.dataset import DatasetItem

    # Initialize metric and runner
    metric = AnswerRelevancy()
    runner = MetricRunner(metrics=[metric], max_concurrent=5)

    # Prepare evaluation data
    data_items = [
        DatasetItem(
            query="How do I reset my password?",
            actual_output="To reset your password, click 'Forgot Password' on the login page and follow the email instructions.",
            expected_output="Navigate to login, click 'Forgot Password', and follow the reset link sent to your email.",
            retrieved_content=["Password reset is available via the login page. Users receive a reset link by email."]
        ),
        # Add more data items...
    ]

    # Execute batch evaluation
    results = await runner.execute_batch(data_items)

    # Process results
    for result in results:
        print(f"Test Case: {result.test_case.query}")
        for score in result.score_results:
            print(f"  {score.name}: {score.score} (passed: {score.passed})")
    ```

=== ":material-cog: Advanced Configuration"

    Configure caching, error handling, custom thresholds, and summary generation.
    The runner accepts `Dataset`, `List[DatasetItem]`, or a Pandas `DataFrame` as input.

    ```python
    from axion.runners import MetricRunner, CacheManager, ErrorConfig, CacheConfig
    from axion.runners.summary import MetricSummary
    from axion.metrics import AnswerRelevancy, Faithfulness

    import pandas as pd

    dataframe = pd.DataFrame({
        'id': '0000001',
        'query': "How do I reset my password?",
        'actual_output': "To reset your password, click 'Forgot Password' on the login page and follow the email instructions.",
        'expected_output': "Navigate to login, click 'Forgot Password', and follow the reset link sent to your email.",
        'retrieved_content': [["Password reset is available via the login page. Users receive a reset link by email."]]
    })

    # Initialize metrics
    metrics = [
        AnswerRelevancy(),
        Faithfulness()
    ]

    # Advanced configuration
    runner = MetricRunner(
        metrics=metrics,
        max_concurrent=10,                          # Concurrency limit
        cache_manager=CacheManager(                 # Optional Caching
            CacheConfig(cache_type='memory')),
        error_config=ErrorConfig(                   # Optional Error handling
            ignore_errors=True,
        ),
        thresholds={                                # Optional Custom thresholds
            'AnswerRelevancy': 0.75,
            'Faithfulness': 0.85
        },
        summary_generator=MetricSummary(),
    )

    # Execute with progress tracking
    results = await runner.execute_batch(
        evaluation_inputs=dataframe, # Can pass Dataset, List of DatasetItems or Pandas Dataframe
    )
    ```

=== ":material-arrow-right-circle: Direct Executor"

    For single-item evaluation, use an executor directly instead of the batch runner.

    ```python
    from axion.runners import AxionRunner
    from axion.metrics import AnswerRelevancy
    from axion.dataset import DatasetItem

    # Initialize specific executor directly
    metric = AnswerRelevancy()
    executor = AxionRunner(metric=metric, threshold=0.7)

    # Execute single evaluation
    data_item = DatasetItem(
        query="What is machine learning?",
        actual_output="Machine learning is a subset of AI...",
        expected_output="ML is a method of data analysis..."
    )

    result = await executor.execute(data_item)
    print(f"Score: {result.score}, Explanation: {result.explanation}")
    ```

## Available Runners

The `MetricRunner` automatically detects which framework a metric belongs to and routes
it to the correct executor. You can also mix metrics from different frameworks in a single run.

=== ":material-cube-outline: Axion"

    **Registry Key:** `axion` &middot; **Class:** `AxionRunner`

    Executes native metrics from the Axion framework. This is the default — any
    `BaseMetric` subclass is automatically routed here.

    ```python
    from axion.metrics import AnswerRelevancy

    metric = AnswerRelevancy()
    runner = MetricRunner(metrics=[metric])

    # The runner automatically detects this is an Axion metric
    # and uses AxionRunner internally
    ```

=== ":material-chart-bar: Ragas"

    **Registry Key:** `ragas` &middot; **Class:** `RagasRunner`

    Executes metrics from the [Ragas](https://docs.ragas.io/) evaluation framework.

    ```python
    from ragas.metrics import Faithfulness
    from axion.integrations.models import LiteLLMRagas

    # Ragas metrics are automatically detected and executed with RagasRunner
    metrics = [Faithfulness(llm=LiteLLMRagas())] # LiteLLMRagas() is optional
    runner = MetricRunner(metrics=metrics)

    # Requires actual_output and optionally retrieved_content
    data_item = DatasetItem(
        query="How do I reset my password?",
        actual_output="Response text...",
        retrieved_content=["Context 1", "Context 2"]
    )
    ```

=== ":material-flask: DeepEval"

    **Registry Key:** `deepeval` &middot; **Class:** `DeepEvalRunner`

    Executes metrics from the [DeepEval](https://docs.confident-ai.com/) framework.

    ```python
    from deepeval.metrics import AnswerRelevancyMetric
    from axion.integrations.models import LiteLLMDeepEval

    # DeepEval metrics are automatically detected
    metrics = [
        AnswerRelevancyMetric(model=LiteLLMDeepEval()), # LiteLLMDeepEval() is optional
    ]
    runner = MetricRunner(metrics=metrics)

    # Execute evaluation
    results = await runner.execute_batch(evaluation_data)
    ```

## Response Format

All metric runners return standardized `TestResult` objects:

| Field | Type | Description |
|-------|------|-------------|
| `test_case` | `DatasetItem` | Original evaluation input |
| `score_results` | `List[MetricScore]` | List of metric evaluation results |

Each `MetricScore` contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier for the test case |
| `name` | `str` | Metric name |
| `score` | `float` | Numerical score (0.0–1.0 typically) |
| `threshold` | `float` | Configured threshold for pass/fail |
| `passed` | `bool` | Whether score meets threshold |
| `explanation` | `str` | Detailed explanation (if available) |
| `source` | `str` | Framework source (axion, ragas, deepeval) |
| `timestamp` | `str` | ISO-formatted execution timestamp |

## Creating Custom Runners

Extend the Metric Runner system by registering your own framework integration.

=== ":material-at: Decorator Registration"

    The recommended approach — use `@MetricRunner.register()` to automatically register
    your runner class.

    ```python
    from axion.runners.metric import MetricRunner, BaseMetricRunner
    from axion.schema import MetricScore
    from axion.dataset import DatasetItem
    from typing import Union, Dict, Any

    @MetricRunner.register('custom_framework')
    class CustomFrameworkRunner(BaseMetricRunner):
        """Custom metric framework runner."""

        _name = 'custom_framework'

        async def execute(self, input_data: Union[DatasetItem, Dict[str, Any]]) -> MetricScore:
            """Execute metric using custom framework."""
            input_data = self.format_input(input_data)

            try:
                # Your custom framework integration here
                score = await self.metric.evaluate(
                    query=input_data.query,
                    response=input_data.actual_output,
                    reference=input_data.expected_output
                )

                return MetricScore(
                    id=input_data.id,
                    name=self.metric_name,
                    score=score,
                    threshold=self.threshold,
                    passed=self._has_passed(score),
                    source=self.source
                )

            except Exception as e:
                return self._create_error_score(input_data.id, e)
    ```

=== ":material-pencil: Manual Registration"

    For cases where decorators aren't suitable — define the class first, then register.

    ```python
    # Define your custom runner class
    class AnotherCustomRunner(BaseMetricRunner):
        _name = 'another_custom'

        async def execute(self, input_data: Union[DatasetItem, Dict[str, Any]]) -> MetricScore:
            # Implementation here
            pass

    # Manual registration
    MetricRunner.register('another_custom')(AnotherCustomRunner)
    ```

Once registered, custom metrics are automatically detected and routed:

```python
from your_custom_framework import CustomMetric

custom_metric = CustomMetric()
runner = MetricRunner(metrics=[custom_metric])

# The runner automatically uses your CustomFrameworkRunner
results = await runner.execute_batch(evaluation_data)
```

---

<div class="ref-nav" markdown="1">

[Runners API Reference :octicons-arrow-right-24:](../../reference/runners.md){ .md-button .md-button--primary }
[Evaluation Runner Deep Dive :octicons-arrow-right-24:](evaluation-runner.md){ .md-button }
[API Runner Deep Dive :octicons-arrow-right-24:](api-runner.md){ .md-button }

</div>
