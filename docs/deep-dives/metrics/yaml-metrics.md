# Creating YAML-Based Metrics

Axion supports creating evaluation metrics directly from YAML configuration files, providing a simple alternative to defining metrics as code. YAML metrics support both LLM-powered evaluation and custom heuristic functions.

## Overview

YAML metrics offer two evaluation approaches:

- **LLM-powered metrics** using instruction prompts
- **Heuristic-based metrics** using custom Python functions

## Basic Structure

Every YAML metric requires either an `instruction` (for LLM evaluation) OR a `heuristic` (for algorithmic evaluation), but not both:

```yaml
# Option 1: LLM-powered metric
name: 'My Metric'
instruction: |
  Your evaluation prompt here...

# Option 2: Heuristic-based metric
name: 'My Metric'
heuristic: |
  def evaluate(item):
      # Your logic here
      return MetricEvaluationResult(score=1.0, explanation="...")

# Optional configuration
model_name: "gpt-4"
threshold: 0.7
required_fields: [...]
optional_fields: [...]
examples: [...]
```

## LLM-Powered Metrics

Use `instruction` to define metrics that leverage language models for evaluation.

### Basic LLM Metric

```yaml
# answer_quality.yaml
name: 'Answer Quality'
instruction: |
  Evaluate the quality of the given answer based on clarity, completeness, and accuracy.
  Provide a score either of 0 or 1 based on .... and explain your reasoning.


# Optional configuration
model_name: "gpt-4"
threshold: 0.7
required_fields:
  - "query"
  - "actual_output"

examples:
  - input:
      query: "What is photosynthesis?"
      actual_output: "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
    output:
      score: 1
      explanation: "Excellent answer that clearly explains photosynthesis with key components. Very clear and complete."

  - input:
      query: "How do you bake a cake?"
      actual_output: "Mix ingredients and bake."
    output:
      score: 0
      explanation: "Poor answer lacking detail. Doesn't specify ingredients, quantities, or timing."
```


## Heuristic-Based Metrics

Use `heuristic` to define metrics with custom Python logic for fast, deterministic evaluation.

### String Matching Heuristic

```yaml
# contains_match.yaml
heuristic: |
  def evaluate(item):
      expected = item.expected_output.strip()
      is_contained = expected in item.actual_output

      return MetricEvaluationResult(
          score=1.0 if is_contained else 0.0,
          explanation=f"Expected text {'found' if is_contained else 'not found'} in actual output"
      )
```


## Loading and Using YAML Metrics

```python
from axion.metrics.yaml_metrics import load_metric_from_yaml
from axion.dataset import DatasetItem

# Load metric from YAML file
MetricClass = load_metric_from_yaml("answer_quality.yaml")

# Create instance
metric = MetricClass(
    field_mapping={"actual_output": "additional_output.summary"}
)
# Evaluate
data_item = DatasetItem(
    query="What is machine learning?",
    actual_output="Machine learning is a subset of AI that enables computers to learn from data.",
    expected_output="Machine learning allows computers to learn patterns from data without explicit programming."
)

result = await metric.execute(data_item)
```

### Field Mapping for Nested Inputs

YAML metrics don't define field mappings in the YAML file yet, but you can pass
`field_mapping` when creating the metric instance to map canonical fields to nested
paths or alternate field names.

---

<div class="ref-nav" markdown="1">

[Metrics API Reference :octicons-arrow-right-24:](../../reference/metrics.md){ .md-button .md-button--primary }
[Creating Custom Metrics :octicons-arrow-right-24:](creating-metrics.md){ .md-button }

</div>
