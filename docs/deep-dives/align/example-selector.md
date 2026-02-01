# Example Selector

The `ExampleSelector` class provides intelligent selection of few-shot examples for LLM-as-judge calibration with CaliberHQ. Instead of naive slicing (`examples[:n]`), it offers strategies that improve calibration quality by balancing accept/reject cases, prioritizing misaligned examples, or covering discovered patterns.

## Overview

When calibrating an LLM judge, the few-shot examples you provide significantly impact evaluation quality. ExampleSelector offers three strategies:

| Strategy | Best For | Requires |
|----------|----------|----------|
| **BALANCED** | General use, initial calibration | Just records + annotations |
| **MISALIGNMENT_GUIDED** | Iterative improvement after eval | Evaluation results |
| **PATTERN_AWARE** | Targeting specific failure modes | Pattern Discovery results |

## Quick Start

```python
from axion.caliber import ExampleSelector, SelectionStrategy

# Initialize with seed for reproducibility
selector = ExampleSelector(seed=42)

# Basic balanced selection
result = selector.select(records, annotations, count=6)

print(f"Selected {len(result.examples)} examples")
print(f"Strategy: {result.strategy_used}")
print(f"Metadata: {result.metadata}")
```

## Selection Strategies

### BALANCED (Default)

Selects a 50/50 mix of accept (score=1) and reject (score=0) cases with random sampling. This prevents the LLM judge from being biased toward one outcome.

```python
result = selector.select(
    records=records,
    annotations=annotations,  # {record_id: 0 or 1}
    count=6,
    strategy=SelectionStrategy.BALANCED
)

# Metadata includes counts
# {'accepts': 3, 'rejects': 3}
```

**When to use:**

- Initial calibration with no prior evaluation data
- General-purpose few-shot selection
- When you want unbiased baseline examples

**Behavior with imbalanced data:**

- If all annotations are accepts, returns only accepts
- If all annotations are rejects, returns only rejects
- Fills remaining slots from the larger pool when one side is exhausted

### MISALIGNMENT_GUIDED

Prioritizes cases where the LLM judge disagreed with human annotations. This focuses calibration on the hardest cases.

```python
# Requires evaluation results from a prior run
eval_results = [
    {'id': 'rec_1', 'score': 1},  # LLM predicted 1
    {'id': 'rec_2', 'score': 0},  # LLM predicted 0
    # ...
]

result = selector.select(
    records=records,
    annotations=annotations,
    count=6,
    strategy=SelectionStrategy.MISALIGNMENT_GUIDED,
    eval_results=eval_results
)

# Metadata shows misalignment stats
# {
#     'false_positives_selected': 2,
#     'false_negatives_selected': 1,
#     'total_fp_available': 5,
#     'total_fn_available': 3,
# }
```

**When to use:**

- After running an initial evaluation and seeing low agreement
- To improve calibration on specific failure modes
- Iterative refinement workflow

**How it works:**

1. Identifies false positives (LLM=1, Human=0) and false negatives (LLM=0, Human=1)
2. Allocates ~1/3 of slots each to FP and FN cases
3. Fills remaining slots with balanced aligned examples

### PATTERN_AWARE

Samples from discovered patterns to ensure coverage across failure categories. Requires results from Pattern Discovery.

```python
from axion.caliber import PatternDiscovery

# First, discover patterns in your annotations
discovery = PatternDiscovery(model_name='gpt-4o')
patterns_result = await discovery.discover(annotated_items)

# Then select examples covering those patterns
result = selector.select(
    records=records,
    annotations=annotations,
    count=6,
    strategy=SelectionStrategy.PATTERN_AWARE,
    patterns=patterns_result.patterns
)

# Metadata shows pattern coverage
# {
#     'patterns_covered': ['Missing Context', 'Factual Errors', 'Too Brief'],
#     'total_patterns': 5,
# }
```

**When to use:**

- After Pattern Discovery reveals distinct failure categories
- To ensure few-shot examples represent all failure types
- When certain patterns are underrepresented in random selection

**How it works:**

1. Takes one example from each discovered pattern (up to `count`)
2. Fills remaining slots with balanced selection
3. Handles patterns with unknown record IDs gracefully

## Data Format

### Records

Records should be a list of dicts with an `id` or `record_id` field:

```python
records = [
    {
        'id': 'rec_1',  # or 'record_id'
        'query': 'What is Python?',
        'actual_output': 'A programming language',
        # ... other fields
    },
    # ...
]
```

### Annotations

Annotations map record IDs to binary scores:

```python
annotations = {
    'rec_1': 1,  # Accept
    'rec_2': 0,  # Reject
    'rec_3': 1,
    # ...
}
```

### Evaluation Results (for MISALIGNMENT_GUIDED)

Evaluation results should include the LLM's score:

```python
eval_results = [
    {'id': 'rec_1', 'score': 1},      # 'score' field
    {'id': 'rec_2', 'llm_score': 0},  # or 'llm_score' field
    # ...
]
```

## Metric Integration

ExampleSelector operates at the record selection level and returns generic dicts. The caller is responsible for converting to the format expected by their metric.

### With CaliberHQ (Dict Format)

```python
from axion.caliber import ExampleSelector, CaliberMetric

selector = ExampleSelector(seed=42)
result = selector.select(records, annotations, count=6)

# Format for CaliberHQ
examples = [
    {
        'input': {
            'query': r['query'],
            'actual_output': r['actual_output'],
            'expected_output': r.get('expected_output'),
        },
        'output': {
            'score': annotations[r['id']],
            'reason': r.get('human_reasoning', ''),
        }
    }
    for r in result.examples
]

evaluator = CaliberMetric(criteria="...", examples=examples)
```

### With Pydantic-Based Metrics

```python
from axion.caliber import ExampleSelector
from axion.metrics import Faithfulness
from axion.metrics.faithfulness import FaithfulnessInput
from axion.schema import MetricEvaluationResult

selector = ExampleSelector(seed=42)
result = selector.select(records, annotations, count=6)

# Convert to metric-specific Pydantic format
examples = [
    (
        FaithfulnessInput(
            query=r['query'],
            actual_output=r['actual_output'],
            retrieval_context=r['retrieval_context'],
        ),
        MetricEvaluationResult(
            score=annotations[r['id']],
            reason=r.get('human_reasoning', ''),
        )
    )
    for r in result.examples
]

metric = Faithfulness(examples=examples)
```

## Best Practices

### Reproducibility

Always set a seed for reproducible results:

```python
selector = ExampleSelector(seed=42)
```

### Strategy Selection

Use this decision tree:

```
Do you have evaluation results from a prior run?
├── YES → Do you have Pattern Discovery results?
│   ├── YES → PATTERN_AWARE (targets specific failure modes)
│   └── NO → MISALIGNMENT_GUIDED (focuses on hard cases)
└── NO → BALANCED (unbiased baseline)
```

### Auto-Selection Helper

```python
def auto_select_strategy(eval_results=None, patterns=None):
    """Automatically choose the best strategy based on available data."""
    if patterns:
        return SelectionStrategy.PATTERN_AWARE
    elif eval_results:
        return SelectionStrategy.MISALIGNMENT_GUIDED
    else:
        return SelectionStrategy.BALANCED
```

### Example Count

- Start with 4-6 examples for initial calibration
- Increase to 8-10 for complex criteria
- More examples increase cost but may improve calibration

## API Reference

### ExampleSelector

::: axion.caliber.ExampleSelector
    options:
      show_root_heading: false
      members:
        - __init__
        - select

### SelectionStrategy

::: axion.caliber.SelectionStrategy
    options:
      show_root_heading: false

### SelectionResult

::: axion.caliber.SelectionResult
    options:
      show_root_heading: false
