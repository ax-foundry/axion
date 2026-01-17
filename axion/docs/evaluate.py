from typing import Any


def basic_usage_template(key: str, item: Any) -> str:
    return """# Basic Usage
from axion.runners import evaluation_runner
from axion.metrics import AnswerRelevancy
from axion.dataset import DatasetItem

data_item = DatasetItem(
    query = "What is the infield fly rule in baseball?",
    actual_output = "The infield fly rule prevents the defense from intentionally dropping a fly ball to turn a double play.",
    expected_output = "The infield fly rule protects baserunners by declaring the batter out on certain easy pop-ups.",
)

# Simple evaluation without task function (expects "actual_output" to be provided in Dataset)
results = evaluation_runner(
    evaluation_inputs=[data_item],
    scoring_metrics=[AnswerRelevancy()],
    evaluation_name="Basic Evaluation"
)
# Access results
results.to_dataframe()

# Create Scorecard
results.to_scorecard()
"""


def hierarchical_usage_template(key: str, item: Any) -> str:
    return """# Hierarchical Scoring Usage
from axion.runners import evaluation_runner
from axion.metrics import AnswerRelevancy
from axion.dataset import DatasetItem

# Option 1 – Saved this to `config.yaml`
# ```
# metric:
#   Relevance:
#     class: 'answer_relevancy'
#     metric_name: 'Relevancy'
#     model_name: 'gpt-4.1'
#     relevancy_mode: 'task'
# model:
#   ANSWER_QUALITY:
#     Relevance: 1.0
# weights:
#   ANSWER_QUALITY: 1.0
# ```

# Option 2 – Define a python dictionary
config = {
    'metric': {
        'Relevance': AnswerRelevancy(
            metric_name='Relevancy',
            model_name='gpt-4.1',
            relevancy_mode='task',
        ),
    },
    'model': {
        'ANSWER_QUALITY': {
            'Relevance': 1.0,
        },
    },
    'weights': {
        'ANSWER_QUALITY': 1.0,
    }
}

data_item = DatasetItem(
    query = "What is the infield fly rule in baseball?",
    actual_output = "The infield fly rule prevents the defense from intentionally dropping a fly ball to turn a double play.",
    expected_output = "The infield fly rule protects baserunners by declaring the batter out on certain easy pop-ups.",
)

results = evaluation_runner(
    evaluation_inputs=[data_item],
    scoring_config=config, # (Can also pass file path of `config.yaml')
    evaluation_name="Hierarchical Evaluation"
)

# Access results
results.to_dataframe()

# Create Scorecard
results.to_scorecard()
"""


def task_usage_template(key: str, item: Any) -> str:
    return """# Task Setup Usage
from axion.runners import evaluation_runner, EvaluationConfig, CacheConfig
from axion.metrics import AnswerRelevancy, Latency
from axion.dataset import DatasetItem


data_item = DatasetItem(
    query = "What is the infield fly rule in baseball?",
    expected_output = "The infield fly rule protects baserunners by declaring the batter out on certain easy pop-ups.",
)

# Task Option 1 - Task returns python dictionary
def dictionary_task_output(item):
    return {
        'response': "The infield fly rule prevents the defense from intentionally dropping a fly ball to turn a double play.",
        'latency': 1.3
    }

# Task Option 2 - Task returns pydantic BaseModel
from pydantic import BaseModel

class Output(BaseModel):
    response: str
    latency: float

def pydantic_task_output(item):
    return Output(
        response="The infield fly rule prevents the defense from intentionally dropping a fly ball to turn a double play.",
        latency=1.3
)

# Task Option 3 - Task can be API runners
# api_runner = ExampleAPIRunner(config={'...'})

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
# Access results
results.to_dataframe()

# Create Scorecard
results.to_scorecard()
    """


def advanced_usage_template(key: str, item: Any) -> str:
    return """# Advanced configuration
from axion.runners import (
    EvaluationRunner,
    EvaluationConfig,
    CacheConfig,
    ErrorConfig
)
from axion.metrics import AnswerCompleteness
from axion.integrations.models import LiteLLMRagas, LiteLLMDeepEval

import pandas as pd
from ragas.metrics import Faithfulness
from deepeval.metrics import AnswerRelevancyMetric

# Create evaluation dataset
dataframe = pd.DataFrame([
    {
        'id': '01',
        'query': "What is the infield fly rule in baseball?",
        'expected_output': "The infield fly rule protects baserunners by declaring the batter out on certain easy pop-ups.",
    },
    {
        'id': '02',
        'query': "What is a balk in baseball?",
        'expected_output': "A balk is an illegal motion by the pitcher that deceives baserunners, resulting in all runners advancing one base.",
    }
])

# Configure LLM models for third-party metrics
deepeval_model = LiteLLMDeepEval(model='gpt-4') # Optional
ragas_model = LiteLLMRagas(model='gpt-4') # Optional

# Configure the task to evaluate
api_runner = ExampleAPIRunner(config='...')

# Define evaluation metrics
metrics = [
    Faithfulness(llm=ragas_model),
    AnswerCompleteness(model_name='gpt-4.1'),
    AnswerRelevancyMetric(model=deepeval_model),
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

# For memory metrics and task runs are cached to
cache = runner.cache_manager.cache

results.to_dataframe()
results.to_scorecard()
"""


def documentation_template(key: str, item: Any) -> str:
    """Documentation template for EvaluationRunner."""
    return """# EvaluationRunner Documentation

## Overview
EvaluationRunner orchestrates evaluation experiments by running tasks and scoring metrics across datasets with support for caching, error handling, and concurrent processing.

## Key Features
- ✅ **Task Execution**: Run custom generation/transformation tasks
- ✅ **Metric Scoring**: Evaluate outputs with multiple metrics
- ✅ **Caching**: Cache task outputs and metric results
- ✅ **Error Handling**: Robust error management and recovery
- ✅ **Concurrency**: Parallel processing for efficiency
- ✅ **Progress Tracking**: Real-time progress monitoring

## Core Components

### EvaluationConfig
Configuration dataclass for evaluation parameters. Includes required fields for evaluation name, evaluation inputs, and scoring metrics, plus optional settings for tasks, concurrency, caching, and error handling.

### evaluation_runner() Function
Synchronous wrapper for running evaluations. Takes your dataset, metrics, and evaluation name as main parameters, with additional options for customization.

## Input Types
- **Dataset**: High-level dataset object
- **List[DatasetItem]**: Individual dataset items
- **pd.DataFrame**: Pandas DataFrame

## Key Parameters

### Required
- evaluation_inputs: Your dataset to evaluate
- scoring_metrics: List of metric objects/callables
- evaluation_name: Unique evaluation identifier

### Optional
- task: Custom function for predictions/transformations
- scoring_key_mapping: Map metric inputs to dataset columns
- max_concurrent: Concurrency limit (default: 5)
- cache_config: Caching configuration
- error_config: Error handling settings
- thresholds: Performance thresholds per metric

## Return Value
Returns EvaluationResult containing:
- run_id: Unique run identifier
- evaluation_name: evaluation name
- results: List of metric results
- metadata: Evaluation metadata
- timestamp: Execution timestamp

## Best Practices
1. **Use caching** for expensive computations
2. **Set appropriate concurrency** based on your resources
3. **Configure error handling** for production workflows
4. **Use key mapping** to adapt metrics to your data schema
5. **Set thresholds** for automated quality checks

## Common Patterns

### Evaluation with Generation
Define a task function that returns a dictionary with your model's response, then pass it to the evaluation runner along with your dataset and metrics.

### Metric Adaptation
Use the scoring_key_mapping parameter to map your dataset column names to what your metrics expect. This allows you to use metrics with different schemas without modifying your data.

## Quick Start
1. Import the evaluation_runner function from axion.runners
2. Prepare your dataset and metrics
3. Call evaluation_runner with your data, metrics, and evaluation name
4. Access results through the returned EvaluationResult object
"""
