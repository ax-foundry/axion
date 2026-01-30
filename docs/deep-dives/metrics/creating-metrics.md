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
    metric_category=MetricCategory.SCORE,  # SCORE, ANALYSIS, or CLASSIFICATION
    default_threshold=0.5,
    score_range=(0, 1),
    tags=["category", "domain"]
)
```

### Metric Categories

Axion supports three metric categories based on output type. Use `MetricCategory` to specify what kind of output your metric produces.

```python
from axion._core.types import MetricCategory
```

#### Decision Tree

```
Does your metric produce a numeric score?
├── YES → MetricCategory.SCORE
└── NO → Does it output a SINGLE label from a FIXED set?
    ├── YES → MetricCategory.CLASSIFICATION
    └── NO → MetricCategory.ANALYSIS
```

#### Category Comparison

| Aspect | `SCORE` | `CLASSIFICATION` | `ANALYSIS` |
|--------|---------|------------------|------------|
| **Output** | Numeric value (0-1) | Single label | Structured object |
| **Options** | Continuous range | Fixed/predefined set | Open-ended insights |
| **Pass/Fail** | Yes (threshold-based) | No | No |
| **Aggregation** | Average, percentile | Count by label | Custom |
| **Example output** | `0.85` | `"Property Condition"` | `{ category, reasons[], citations[] }` |

#### When to Use Each Category

| Category | Use When | Examples |
|----------|----------|----------|
| **`SCORE`** | Metric produces a numeric value that can be compared to a threshold | `Faithfulness`, `AnswerRelevancy`, `Latency`, `acceptance_rate` |
| **`CLASSIFICATION`** | Metric produces a single categorical label from a fixed set | `SentimentClassification` → positive/negative/neutral |
| **`ANALYSIS`** | Metric extracts structured insights without scoring | `ReferralReasonAnalysis` → reasons, citations, categories |

#### Examples

**SCORE Metric** (default):
```python
@metric(
    name="Answer Quality",
    metric_category=MetricCategory.SCORE,  # Default, can be omitted
    default_threshold=0.7,
    score_range=(0, 1),
    ...
)
class AnswerQuality(BaseMetric):
    async def execute(self, item, **kwargs) -> MetricEvaluationResult:
        score = await self._evaluate(item)
        return MetricEvaluationResult(
            score=score,  # Numeric score required
            explanation="Quality assessment complete"
        )
```

**CLASSIFICATION Metric**:
```python
@metric(
    name="Sentiment Classification",
    metric_category=MetricCategory.CLASSIFICATION,
    required_fields=["actual_output"],
    ...
)
class SentimentClassification(BaseMetric):
    async def execute(self, item, **kwargs) -> MetricEvaluationResult:
        label = await self._classify(item.actual_output)
        return MetricEvaluationResult(
            score=None,  # No score for classification
            explanation=f"Classified as: {label}",
            signals={"label": label}  # Single categorical output
        )
```

**ANALYSIS Metric**:
```python
@metric(
    name="Referral Reason Analysis",
    metric_category=MetricCategory.ANALYSIS,
    required_fields=["actual_output"],
    ...
)
class ReferralReasonAnalysis(BaseMetric):
    async def execute(self, item, **kwargs) -> MetricEvaluationResult:
        result = await self._analyze(item)
        return MetricEvaluationResult(
            score=None,  # No score for analysis
            explanation=f"Extracted {len(result.reasons)} reasons",
            signals={  # Rich structured output
                "primary_category": result.primary_category,
                "all_reasons": result.reasons,
                "citations": result.citations,
                "actionable_type": result.actionable_type,
            }
        )
```

#### Downstream Behavior

| Behavior | `SCORE` | `CLASSIFICATION` | `ANALYSIS` |
|----------|---------|------------------|------------|
| `score` field | Required numeric | `None` → `np.nan` | `None` → `np.nan` |
| `passed` field | `True`/`False` based on threshold | `None` | `None` |
| `threshold` field | Set from config | `None` | `None` |
| Summary reports | Included in averages | Excluded from averages | Excluded from averages |
| Metadata | `metric_category: "score"` | `metric_category: "classification"` | `metric_category: "analysis"` |

### Field Mapping for Nested Inputs

If your dataset stores canonical fields under different names or nested paths, you can
map them at runtime using `field_mapping`. Mapped fields are used for validation and
lookup, so required fields still pass when they are sourced from alternate locations.

```python
from axion.metrics import AnswerCompleteness

metric = AnswerCompleteness(
    field_mapping={
        "actual_output": "additional_output.summary",
        "expected_output": "additional_input.reference",
    }
)
```

Paths use dot notation and traverse attributes or dict keys. For example, the mapping
above resolves `actual_output` from `item.additional_output["summary"]`.

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
    query="How do I reset my password?",
    actual_output="To reset your password, click 'Forgot Password' on the login page and follow the email instructions.",
    expected_output="Navigate to login, click 'Forgot Password', and follow the reset link sent to your email.",
    retrieved_content=["Password reset available via login page", "Reset link sent by email"]
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
