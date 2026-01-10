# Running Evaluations

Axion provides evaluation runners for batch processing with caching and parallel execution.

## Quick Start

```python
from axion import Dataset
from axion.metrics import Faithfulness, AnswerRelevancy
from axion.runners import evaluation_runner

# Load dataset
dataset = Dataset.from_json("eval_data.json")

# Run evaluation
results = await evaluation_runner(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy()]
)

# View results
print(results.summary())
df = results.to_dataframe()
```

## Evaluation Runners

### evaluation_runner

The main entry point for running evaluations:

```python
from axion.runners import evaluation_runner, EvaluationConfig

config = EvaluationConfig(
    max_concurrency=10,
    cache_enabled=True,
    cache_dir=".cache/evaluations"
)

results = await evaluation_runner(
    dataset=dataset,
    metrics=metrics,
    config=config
)
```

### MetricRunner

For running individual metrics with more control:

```python
from axion.runners import MetricRunner

runner = MetricRunner(metric=Faithfulness())
result = await runner.run(dataset_item)
```

## Caching

Avoid re-running expensive LLM evaluations:

```python
from axion._core.cache import CacheManager, CacheConfig

cache = CacheManager(CacheConfig(
    cache_type="disk",
    cache_dir=".cache/metrics"
))

# Results are cached by input hash
results = await evaluation_runner(dataset, metrics, cache=cache)
```

## Understanding Results

```python
# Overall summary
print(results.summary())

# Per-metric scores
for metric_result in results.metric_results:
    print(f"{metric_result.name}: {metric_result.score}")

# Detailed breakdown
df = results.to_dataframe()
df[df['score'] < 0.5]  # Find failures
```

## Best Practices

1. **Start small** - Test with a few items before running full dataset
2. **Enable caching** - Avoid re-running on unchanged inputs
3. **Use appropriate concurrency** - Balance speed vs. API rate limits
4. **Review failures** - Low scores need human analysis, not just numbers

## Next Steps

- [Metric Runner Deep Dive](../deep-dives/runners/metric-runner.md) - Advanced runner usage
- [Evaluation Runner Deep Dive](../deep-dives/runners/evaluation-runner.md) - Configuration options
- [API Reference: Runners](../reference/runners.md) - Full API documentation
