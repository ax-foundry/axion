from typing import Any


def python_template(key: str, item: Any) -> str:
    return """# Python Usage
from axion.metrics import AnswerRelevancy
from ragas.metrics import Faithfulness
from deepeval.metrics import ContextualRecallMetric
from axion.integrations.models import RagasLLM, DeepEvalLLM

config = {
    'metric': {
        'answer_relevancy': AnswerRelevancy(),
        'faithfulness': Faithfulness(llm=RagasLLM()),
        'contextual_recall': ContextualRecallMetric(model=DeepEvalLLM()),
    },
    'model': {
        'ANSWER_QUALITY': {
            'answer_relevancy': 1.0,
        },
        'KNOWLEDGE_QUALITY': {
            'faithfulness': 0.6,
            'contextual_recall': 0.4
        }
    },
    'weights': {
        'ANSWER_QUALITY': 0.5,
        'KNOWLEDGE_QUALITY': 0.5,
    }
}

data_item = DatasetItem(
    query = "What is Data Cloud?",
    actual_output = "Data Cloud is a hyperscale data platform to unlock value built on the Salesforce Platform.",
    expected_output = "Data Cloud is a hyperscale data platform built directly into Salesforce.",
    retrieved_content = ["built on the Salesforce Platform", "Unlocks Enterprise Value"],
)

model = EvalTree(config)
result = await model.execute(data_item) # Single Evaluation
model.visualize(result)
# results = await model.batch_execute([data_item]) # Batch Evaluation
# model.visualize(results)
# Access Results
results.to_dataframe()
"""


def yaml_template(key: str, item: Any) -> str:
    return """# Save this as: my_tree.yaml
metric: # Definition of metrics and parameters
  answer_relevancy:
    model_name: gpt-4o
    threshold: 0.5
  contextual_relevancy:
    model_name: gpt-4o
    threshold: 0.5
  citation_presence:
    resource_similarity_threshold: 0.8
  faithfulness:
    model_name: gpt-4o
    class: "deepeval.metrics.FaithfulnessMetric"
    threshold: 0.5

model: # Define your hierarchy
  ANSWER_QUALITY:
    RESPONSE:
      answer_relevancy: 1.0
    STRUCTURAL:
      citation_presence: 1.0
  KNOWLEDGE_QUALITY:
      answer_relevancy: 0.5
      faithfulness: 0.5

weights: # Weights for each Component Node
  ANSWER_QUALITY: 0.5
  KNOWLEDGE_QUALITY: 0.5
  RESPONSE: 0.9
  STRUCTURAL: 0.1


##  Usage in Python:
# from axion.dataset import DatasetItem
# from axion.eval_tree import EvalTree

# data_item = DatasetItem(
#    query = "What is Data Cloud?",
#    actual_output = "Data Cloud (https://www.salesforce.com/data/) is a hyperscale data platform to unlock value built on the Salesforce Platform.",
#    retrieved_content = ["built on the Salesforce Platform", "Unlocks Enterprise Value"],
# )

# model = EvalTree("my_tree.yaml")
# result = await model.execute(data_item) # Single Evaluation
# model.visualize(result)
# results = await model.batch_execute([data_item]) # Batch Evaluation
# model.visualize(results)
# # Access Results
# results.to_dataframe()
"""


def documentation_template(key: str, item: Any) -> str:
    """Documentation template for Eval Tree."""
    return """# Eval Tree Documentation

## Overview
EvalTree orchestrates complex, multi-faceted evaluations by building a hierarchical scoring model from a configuration. It runs a tree of metrics against datasets, aggregates scores from the bottom up, and provides powerful visualization tools.

## Key Features
- ✅ **Hierarchical Scoring**: Define complex scoring logic with nested components.
- ✅ **Flexible Configuration**: Configure evaluations from YAML files or directly in Python.
- ✅ **Metric Integration**: Supports Ragas, DeepEval, BEAM, and custom AI Toolkit metrics.
- ✅ **Concurrent Execution**: Parallel processing for efficient, large-scale evaluations.
- ✅ **Visualization**: Interactive tree diagrams to inspect scores and weights at every level.
- ✅ **Optimized Batching**: Intelligently groups API calls to maximize throughput.

## Core Configuration Components

The `EvalTree` is driven by a single configuration object (often a dictionary) with three top-level keys:

### 1. metric
This section defines all the individual metrics (the leaves of the tree) that can be used in the model. There are two ways to define a metric:

- **Declarative (from file/dict)**: Provide a dictionary with a class key pointing to the metric's import path.
- **Programmatic (in code)**: Pass a pre-instantiated metric object. This is useful for dynamic workflows.

### 2. model
This section defines the hierarchical structure of your evaluation.
- **Component Nodes**: Represented by dictionaries. They group other components or metrics (e.g., ANSWER_QUALITY).
- **Metric Nodes**: Represented by non-dictionary values (e.g., a float). The key is the metric name (which must match a name in the metric section), and the value is its weight within the parent component.

### 3. weights
This section defines the weights for the **component nodes** in a flat structure.

## Execution Methods
- **execute(data_item)**: Runs the full evaluation tree for a single data item.
- **batch_execute(data_items)**: Runs the evaluation for a list of data items using an optimized, parallelized approach.

## Return Value
- **execute()** returns a **TestResult** object containing the detailed score breakdown for one item.
- **batch_execute()** returns an **EvaluationResult** object containing results for all items.
"""
