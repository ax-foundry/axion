# Evaluation Runner

The Evaluation Runner is an orchestration system for running end-to-end evaluation experiments. It  integrates task execution (model inference) with metric evaluation, providing a unified workflow for evaluating AI systems. The runner supports advanced features including caching, error handling, key remapping, custom thresholds, and detailed tracing.

## Overview

The Evaluation Runner combines three key phases into a single unified workflow:

1. **Task Execution** - Optional custom function to generate model predictions/outputs
2. **Metric Evaluation** - Automated scoring using multiple evaluation frameworks
3. **Result Aggregation** -  Result collection with metadata and summaries


## Available Metric Runners

To explore all inline usage examples and documentation for the evaluation runner and the configuration options, use the built-in discovery method:

```python title="View Available Runners"
from axion.runners import EvaluationRunner

# Display documentation and usage examples
EvaluationRunner.display()
```


## Configuration

The Evaluation Runner accepts configuration through the `EvaluationConfig` object or direct function parameters:

```python
from axion.runners import evaluation_runner, EvaluationConfig, ErrorConfig, CacheConfig, EvaluationRunner

# Method 1: Direct function call
results = evaluation_runner(
    evaluation_inputs=dataset,
    scoring_metrics=[AnswerRelevancy()],
    evaluation_name="My Experiment",
    task=my_task_function,
    max_concurrent=10,
    thresholds={"question_answer_relevance": 0.75}
)

# Method 2: Configuration object
config = EvaluationConfig(
    evaluation_name="Advanced Experiment",
    evaluation_inputs=dataset,
    scoring_metrics=metrics,
    task=generation_task,
    scoring_key_mapping={'actual_output': 'response'},
    evaluation_description="Evaluating new model version",
    evaluation_metadata={"model_version": "v2.1"},
    cache_config=CacheConfig(use_cache=True),
    max_concurrent=5,
    show_progress=True
)

runner = EvaluationRunner(config)
results = await runner.execute()  # Async execution
# To Pandas DataFrame
results.to_dataframe()
# Create Scorecard
results.to_scorecard()
```

## Usage Patterns

### Basic Evaluation (No Task)

For datasets that already contain model outputs:

```python
from axion.runners import evaluation_runner
from axion.metrics import AnswerRelevancy
from axion.dataset import DatasetItem

# Prepare dataset with existing outputs
dataset = [
    DatasetItem(
        query="How do I reset my password?",
        actual_output="To reset your password, click 'Forgot Password' on the login page...",
        expected_output="Navigate to login, click 'Forgot Password', and follow the reset link."
    ),
    # More items...
]

# Run evaluation only
results = evaluation_runner(
    evaluation_inputs=dataset,
    scoring_metrics=[AnswerRelevancy()],
    evaluation_name="Basic Evaluation"
)

print(f"Evaluation Name: {results.evaluation_name}")
print(f"Success rate: {results.success_rate}")
```

### End-to-End Evaluation (With Task)

For generating predictions and evaluating them in one workflow:

```python
# Task Setup Usage
from axion.runners import evaluation_runner, EvaluationConfig, CacheConfig
from axion.metrics import AnswerRelevancy, Latency
from axion.dataset import DatasetItem


data_item = DatasetItem(
    query = "How do I reset my password?",
    expected_output = "Navigate to login, click 'Forgot Password', and follow the reset link.",
)

# Task Option 1 - Task returns python dictionary (can be async or sync)
def dictionary_task_output(item):
    return {
        'response': "To reset your password, click 'Forgot Password' on the login page.",
        'latency': 1.3
    }

# Task Option 2 - Task returns pydantic BaseModel
from pydantic import BaseModel

class Output(BaseModel):
    response: str
    latency: float

def pydantic_task_output(item):
    return Output(
        response="To reset your password, click 'Forgot Password' on the login page.",
        latency=1.3
)

# Task Option 3 - Task can be API runners
from axion.runners.api import PromptTemplateAPIRunner
api_runner = PromptTemplateAPIRunner(config={'...'})

results = evaluation_runner(
    evaluation_inputs=[data_item],
    task=dictionary_task_output, # or pydantic_task_output, api_runner
    scoring_metrics=[
        AnswerRelevancy(model_name='gpt-4o'),
        Latency(threshold=1.5),
    ],
    max_concurrent=5,
    scoring_key_mapping={'actual_output': 'response'}, # Required if your `task` returns a different schema
    evaluation_name='Custom Setup',
    evaluation_metadata={"model_version": "v2.1", "data_slice": "test_set"},
    cache_config=CacheConfig(use_cache=True),
)
# To Pandas DataFrame
results.to_dataframe()
# Create Scorecard
results.to_scorecard()

```

### Advanced Configuration with Multiple Libraries

```python
# Advanced configuration
from axion.runners import (
    EvaluationRunner,
    EvaluationConfig,
    CacheConfig,
    ErrorConfig
)
from axion.metrics import AnswerCompleteness, Latency
from axion.integrations.models import LiteLLMRagas, LiteLLMDeepEval
from axion.runners.api import PromptTemplateAPIRunner

import pandas as pd
from ragas.metrics import Faithfulness
from deepeval.metrics import AnswerRelevancyMetric

# Create evaluation dataset
dataframe = pd.DataFrame([
    {
        'id': '01',
        'query': "How do I reset my password?",
        'expected_output': "Navigate to login, click 'Forgot Password', and follow the reset link.",
    },
    {
        'id': '02',
        'query': "How do I update my billing information?",
        'expected_output': "Go to Account Settings, select Billing, and update your payment method.",
    }
])

# Configure LLM models for third-party metrics
deepeval_model = LiteLLMDeepEval(model='gpt-4') # Optional
ragas_model = LiteLLMRagas(model='gpt-4') # Optional

# Configure the task to evaluate
api_runner = PromptTemplateAPIRunner(config='...')

# Define evaluation metrics
metrics = [
    Faithfulness(llm=ragas_model),
    AnswerCompleteness(model_name='gpt-4'),
    AnswerRelevancyMetric(model=deepeval_model),
    Latency(threshold=8)
]

# Configure the evaluation
config = EvaluationConfig(
    evaluation_name="Advanced Configuration Eval",
    evaluation_inputs=dataframe,
    scoring_metrics=metrics,
    task=api_runner,
    max_concurrent=10,
    cache_config=CacheConfig(use_cache=True, cache_type='memory'),
    error_config=ErrorConfig(skip_on_missing_params=True),
    thresholds={"faithfulness": 0.8, "answer_completeness": 0.7},
    evaluation_metadata={"model_version": "v2.1"},
    dataset_name='Advanced Configuration Dataset'
)

# Run the evaluation
runner = EvaluationRunner(config)
results = await runner.execute()  # For async execution
# To Pandas DataFrame
results.to_dataframe()
# Create Scorecard
results.to_scorecard()

```


## Task Functions

Task functions are custom callables that generate model outputs for evaluation. They must accept a `DatasetItem` and return output data as either a Pydantic `BaseModel` or python `Dict`. Supports both async and sync functions.

### Simple Task Function

```python
def simple_task(item: DatasetItem) -> Dict[str, Any]:
    """Simple synchronous task function."""
    response = your_model.generate(item.query)

    return {
        'actual_output': response,
        'generation_time': time.time()
    }
```

## Key Remapping

Use scoring key mapping to adapt between task outputs and metric input requirements:

```python
# Your task returns this structure
task_output = {
    'generated_text': "The answer is...",
    'source_documents': ["doc1", "doc2"],
    'model_confidence': 0.95
}

# But metrics expect this structure
metric_expected = {
    'actual_output': "The answer is...",
    'retrieved_content': ["doc1", "doc2"]
}

# Use key mapping to bridge the gap
scoring_key_mapping = {
    'actual_output': 'generated_text',
    'retrieved_content': 'source_documents',
}

results = evaluation_runner(
    evaluation_inputs=dataset,
    task=my_task,
    scoring_metrics=metrics,
    scoring_key_mapping=scoring_key_mapping,
    evaluation_name="Mapped Evaluation"
)
```

## Response Format

The Evaluation Runner returns an `EvaluationResult` object containing:

| Field | Type | Description                                |
|-------|------|--------------------------------------------|
| `run_id` | `str` | Unique identifier for this evaluation run  |
| `evaluation_name` | `str` | Name of the evaluation                     |
| `timestamp` | `str` | ISO-formatted execution timestamp          |
| `results` | `List[TestResult]` | Detailed results for each evaluation input |
| `summary` | `Dict[str, Any]` | Summary of the TestResult objects          |
| `metadata` | `Dict[str, Any]` | Evaluation metadata and configuration      |

Each `TestResult` contains the same structure as described in the Metric Runner documentation.


----

## Advanced Features

### Caching Configuration

Type of caching backend to use.
- 'memory': Uses in-memory dictionary for caching (fast, but non-persistent). Must use class model for evaluations.
- 'disk': Writes cache to disk (persistent across runs).

```python
from axion.runners import CacheConfig

cache_config = CacheConfig(
    use_cache=True,          # Enable caching
    cache_task=True,         # Cache task outputs
    cache_type='disk',
    cache_dir='cache/',
)

results = evaluation_runner(
    evaluation_inputs=dataset,
    task=expensive_task,
    scoring_metrics=metrics,
    cache_config=cache_config,
    evaluation_name="Cached Evaluation"
)
```

### Evaluation Tracking and Metadata

```python
results = evaluation_runner(
    evaluation_inputs=dataset,
    task=task,
    scoring_metrics=metrics,
    evaluation_name="Model Comparison v2.1",
    evaluation_description="Comparing new model against baseline",
    evaluation_metadata={
        'model_version': 'v2.1.0',
        'baseline_version': 'v1.9.2',
        'dataset_version': 'eval_set_march_2024',
        'environment': 'staging',
        'researcher': 'data_science_team',
        'tags': ['comparison', 'monthly_eval', 'production_candidate']
    },
    run_id="evaluation_2024_03_15_001"
)

# Access metadata in results
print(f"Model version: {results.metadata['model_version']}")
print(f"Environment: {results.metadata['environment']}")
```


## Tracing Integration

The Evaluation Runner automatically integrates with Axion's tracing system for observability. When tracing is enabled (e.g., with Langfuse), you get detailed visibility into evaluation execution.

### Trace Names

The `evaluation_name` parameter is used as the trace name in your observability platform:

```python
results = evaluation_runner(
    evaluation_inputs=dataset,
    scoring_metrics=[AnswerRelevancy(), Faithfulness()],
    evaluation_name="RAG Quality Check v2"  # Appears as trace name in Langfuse
)
```

This makes it easy to identify and filter evaluations in Langfuse by their purpose.

### Captured Data

The evaluation runner captures input/output at multiple levels:

| Span | Input Captured | Output Captured |
|------|----------------|-----------------|
| Evaluation (root) | Config summary, input count, metrics list | Total items, metrics evaluated, status |
| MetricRunner_Batch | Input count, metric names | Results count, pass rate |
| Metric execution | Query, actual/expected output, context | Score, passed status, explanation |

### Example Langfuse Output

```
RAG Quality Check v2                          # evaluation_name
├─ MetricRunner_Batch                         # Batch processing span
│  ├─ (Axion) Answer Relevancy               # Individual metric spans
│  │  └─ litellm_structured_execution        # LLM call
│  └─ (Axion) Faithfulness
│     └─ litellm_structured_execution
```

### Enabling Tracing

```python
import os
os.environ['TRACING_MODE'] = 'langfuse'
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-...'
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-...'
os.environ['LANGFUSE_BASE_URL'] = 'https://us.cloud.langfuse.com'  # or EU

from axion._core.tracing import configure_tracing
configure_tracing()

# Now run your evaluation - traces are automatically captured
results = evaluation_runner(
    evaluation_inputs=dataset,
    scoring_metrics=metrics,
    evaluation_name="My Evaluation"
)
```

See the [Tracing Documentation](../internals/tracing.md) for more details on configuration options.

---

## API Reference

::: axion.runners.evaluate
