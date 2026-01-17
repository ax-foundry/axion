# Metric Runner

The Metric Runner is an orchestration system for executing evaluation metrics across multiple libraries. It provides a unified interface for running single metrics or batches of metrics against evaluation datasets with built-in concurrency control, caching, error handling, and progress tracking.



### MetricRunner Class

The main orchestrator that manages multiple metric executors and provides batch processing capabilities.

**Key Features:**

- Registry-based metric framework management
- Concurrent execution with configurable limits
- Built-in progress tracking with tqdm integration
- Automatic caching support
- Flexible error handling configurations
- Standardized result summarization


## Available Metric Runners

To explore all available Metric runners and their configuration options, use the built-in discovery method:

```python title="View Available Runners"
from axion.runners import MetricRunner

# Display all registered Metric runners with their options
MetricRunner.display()
```


## Usage Patterns

### Basic Batch Evaluation

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

### Advanced Configuration

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

### Direct Executor Usage

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

## Available Metric Runners

### Axion Runner

**Registry Key:** `axion`
**Class:** `AxionRunner`
**Purpose:** Executes native metrics from the Axion framework


**Usage Example:**
```python
from axion.metrics import AnswerRelevancy

metric = AnswerRelevancy()
runner = MetricRunner(metrics=[metric])

# The runner automatically detects this is an Axion metric
# and uses AxionRunner internally
```

### Ragas Runner

**Registry Key:** `ragas`
**Class:** `RagasRunner`
**Purpose:** Executes metrics from the Ragas evaluation framework

**Usage Example:**
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

### DeepEval Runner

**Registry Key:** `deepeval`
**Class:** `DeepEvalRunner`
**Purpose:** Executes metrics from the DeepEval framework


**Usage Example:**
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

All metric runners return standardized `TestResult` objects containing:

| Field | Type | Description |
|-------|------|-------------|
| `test_case` | `DatasetItem` | Original evaluation input |
| `score_results` | `List[MetricScore]` | List of metric evaluation results |

Each `MetricScore` contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier for the test case |
| `name` | `str` | Metric name |
| `score` | `float` | Numerical score (0.0-1.0 typically) |
| `threshold` | `float` | Configured threshold for pass/fail |
| `passed` | `bool` | Whether score meets threshold |
| `explanation` | `str` | Detailed explanation (if available) |
| `source` | `str` | Framework source (axion, ragas, deepeval) |
| `timestamp` | `str` | ISO-formatted execution timestamp |

## Creating Custom Metric Runners

You can extend the Metric Runner system by creating custom metric framework integrations.

### Method 1: Decorator Registration

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

### Method 2: Manual Registration

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

### Using Custom Runners

```python
# Your custom metrics will be automatically detected and routed
# to the appropriate runner based on their module path

from your_custom_framework import CustomMetric

custom_metric = CustomMetric()
runner = MetricRunner(metrics=[custom_metric])

# The runner automatically uses your CustomFrameworkRunner
results = await runner.execute_batch(evaluation_data)
```


## API Reference

::: axion.runners.metric
