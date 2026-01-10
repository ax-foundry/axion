# Creating Custom Metrics

Axion provides a flexible framework for creating custom evaluation metrics. You can build metrics that leverage LLMs, traditional algorithms, or hybrid approaches, all while maintaining consistency with the broader evaluation ecosystem.

## Overview

Custom metrics in Axion inherit from `BaseMetric` and can be registered automatically using decorators or manually through the registry. The framework supports:

- **LLM-powered metrics** using structured prompts and examples
- **Algorithm-based metrics** using traditional computation
- **Hybrid metrics** combining multiple evaluation approaches
- **Complex multi-step metrics** with intermediate processing

## Core Components

### BaseMetric Class

All custom metrics inherit from `BaseMetric`, which provides:

- **LLM Integration** - Built-in LLM handler with configurable models
- **Structured I/O** - Type-safe input/output with Pydantic models
- **Execution Framework** - Async execution with tracing and logging
- **Configuration Management** - Threshold and parameter handling
- **Validation** - Automatic field validation for dataset items

### Metric Decorator

The `@metric` decorator provides declarative configuration:

```python
@metric(
    name="Human-readable metric name",
    description="Detailed description of what the metric measures",
    required_fields=["field1", "field2"],
    optional_fields=["field3"],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=["category", "domain"]
)
```

## Creating Simple Metrics

### Basic LLM-Powered Metric

```python
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.dataset import DatasetItem

@metric(
    name="Answer Quality",
    description="Evaluates the overall quality of an answer based on clarity, completeness, and accuracy",
    required_fields=["actual_output"],
    optional_fields=["expected_output", "query"],
    default_threshold=0.7,
    score_range=(0, 1),
    tags=["quality", "general"]
)
class AnswerQuality(BaseMetric):
    """Evaluates answer quality across multiple dimensions."""

    instruction = """
    Evaluate the quality of the given answer based on the following criteria:
    1. Clarity - Is the answer clear and easy to understand?
    2. Completeness - Does the answer fully address the question?
    3. Accuracy - Is the information provided correct?
    Provide a score from 0 to 1, where:
    ....
    Provide a brief explanation for your score.
    """

    examples = [
        (
            DatasetItem(
                query="What is photosynthesis?",
                actual_output="Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs in chloroplasts and is essential for plant survival and the oxygen we breathe."
            ),
            MetricEvaluationResult(
                score=0.9,
                explanation="Excellent answer that clearly explains photosynthesis, includes key components (sunlight, CO2, water, glucose, oxygen), mentions location (chloroplasts), and explains significance. Very clear and complete."
            )
        ),
        (
            DatasetItem(
                query="How do you bake a cake?",
                actual_output="Mix ingredients and bake."
            ),
            MetricEvaluationResult(
                score=0.2,
                explanation="Poor answer that lacks detail and completeness. Doesn't specify ingredients, quantities, temperatures, or timing. Too vague to be useful for someone wanting to bake a cake."
            )
        )
    ]
```

### Algorithm-Based Metric

```python
import re
from typing import Set

@metric(
    name="Keyword Coverage",
    description="Measures how many expected keywords appear in the actual output",
    required_fields=["actual_output", "expected_keywords"],
    default_threshold=0.6,
    score_range=(0, 1),
    tags=["coverage", "keywords", "algorithmic"]
)
class KeywordCoverage(BaseMetric):
    """Calculates the percentage of expected keywords found in the output."""

    async def execute(self, item: DatasetItem) -> MetricEvaluationResult:
        """Calculate keyword coverage score."""
        self._validate_required_metric_fields(item)

        actual_output = item.actual_output.lower()
        expected_keywords = item.expected_keywords

        if isinstance(expected_keywords, str):
            expected_keywords = [kw.strip() for kw in expected_keywords.split(',')]

        # Find keywords in output
        found_keywords = []
        missing_keywords = []

        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in actual_output:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

        # Calculate score
        score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.0

        # Generate explanation
        explanation = f"Found {len(found_keywords)}/{len(expected_keywords)} expected keywords. "
        if found_keywords:
            explanation += f"Found: {', '.join(found_keywords)}. "
        if missing_keywords:
            explanation += f"Missing: {', '.join(missing_keywords)}."

        return MetricEvaluationResult(
            score=score,
            explanation=explanation.strip()
        )
```

## Registration Methods

### Automatic Registration with Decorator

The `@metric` decorator automatically registers metrics:

```python
@metric(
    name="Custom Metric",
    description="Description of the metric",
    required_fields=["field1", "field2"],
    default_threshold=0.5
)
class CustomMetric(BaseMetric):
    # Metric implementation
    pass

# Metric is automatically available in registry
from axion.metrics import metric_registry
metric_class = metric_registry.get("custom_metric")
```

### Manual Registration

For dynamic registration or when decorators aren't suitable:

```python
from axion.metrics.base import MetricConfig
from axion.metrics import metric_registry

class DynamicMetric(BaseMetric):
    """Dynamically configured metric."""
    pass

# Create configuration
config = MetricConfig(
    key="dynamic_metric",
    name="Dynamic Metric",
    description="A dynamically registered metric",
    required_fields=["actual_output"],
    optional_fields=[],
    default_threshold=0.6,
    score_range=(0, 1),
    tags=["dynamic"]
)

# Attach config and register
DynamicMetric.config = config
metric_registry.register(DynamicMetric)
```

## Usage Examples

### Using Custom Metrics

```python
from axion.dataset import DatasetItem

# Initialize your custom metric
metric = AnswerQuality()

# Prepare test data
data_item = DatasetItem(
    query="What is Data Cloud?",
    actual_output="Data Cloud is a hyperscale data platform to unlock value built on the Salesforce Platform.",
    expected_output="Data Cloud is a hyperscale data platform built directly into Salesforce.",
    retrieved_content=["built on the Salesforce Platform", "Unlocks Enterprise Value"]
)

# Execute evaluation
result = await metric.execute(data_item)
print(f"Score: {result.score}")
print(f"Explanation: {result.explanation}")

# Pretty print results
print(result.pretty())
```


## Best Practices

### Metric Design
- **Single Responsibility** - Each metric should evaluate one specific aspect
- **Clear Scoring** - Use consistent scoring scales and document ranges
- **Robust Validation** - Validate inputs thoroughly and provide helpful error messages
- **Comprehensive Examples** - Include diverse examples that cover edge cases

### Error Handling
- **Graceful Degradation** - Provide fallback scores when computation fails
- **Informative Messages** - Return helpful error messages and explanations
- **Input Validation** - Validate inputs early and provide clear requirements
- **Logging** - Use appropriate logging levels for debugging and monitoring
