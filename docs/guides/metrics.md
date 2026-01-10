# Metrics & Evaluation

Axion provides 98+ metrics for evaluating AI agents across multiple dimensions.

## Quick Start

```python
from axion import Dataset, metric_registry
from axion.metrics import Faithfulness, AnswerRelevancy

# Load your dataset
dataset = Dataset.from_json("eval_data.json")

# Select metrics
metrics = [Faithfulness(), AnswerRelevancy()]

# Run evaluation
from axion.runners import evaluation_runner
results = await evaluation_runner(dataset, metrics)
```

## Metric Categories

### Composite Metrics (LLM-based)

For nuanced evaluation requiring reasoning:

| Metric | What it Measures |
|--------|-----------------|
| `Faithfulness` | Is the answer grounded in retrieved context? |
| `AnswerRelevancy` | Does the answer address the question? |
| `FactualAccuracy` | Is the answer factually correct vs. expected? |
| `AnswerCompleteness` | Are all parts of the question answered? |
| `AnswerCriteria` | Does the answer meet specific business rules? |

### Heuristic Metrics (Non-LLM)

Fast, deterministic checks:

| Metric | What it Measures |
|--------|-----------------|
| `ExactStringMatch` | Exact match between actual and expected |
| `CitationPresence` | Are citations/references present? |
| `Latency` | Response time (pass/fail threshold) |
| `ContainsMatch` | Does output contain required phrases? |

### Retrieval Metrics

For RAG pipeline evaluation:

| Metric | What it Measures |
|--------|-----------------|
| `HitRateAtK` | Is the right doc in top K results? |
| `MeanReciprocalRank` | Position of first relevant result |
| `ContextualRelevancy` | Are retrieved chunks relevant? |
| `ContextualSufficiency` | Do chunks contain the answer? |

### Conversational Metrics

For multi-turn agents:

| Metric | What it Measures |
|--------|-----------------|
| `GoalCompletion` | Did user achieve their goal? |
| `ConversationEfficiency` | Were there unnecessary loops? |
| `ConversationFlow` | Is the dialogue logical? |

## Using the Metric Registry

```python
from axion.metrics import metric_registry

# List all available metrics
print(metric_registry.list_metrics())

# Get metric by name
metric = metric_registry.get("Faithfulness")

# Filter by category
composite_metrics = metric_registry.filter(category="composite")
```

## Customizing Metrics

```python
from axion.metrics import Faithfulness

# Adjust threshold
metric = Faithfulness(threshold=0.8)

# Custom instructions
metric = AnswerCriteria(
    criteria_key="my_criteria",
    scoring_strategy="aspect"
)
```

## Next Steps

- [Running Evaluations](evaluation.md) - Scale up with evaluation runners
- [Creating Custom Metrics](../deep-dives/metrics/creating-metrics.md) - Build your own metrics
- [API Reference: Metrics](../reference/metrics.md) - Full API documentation
