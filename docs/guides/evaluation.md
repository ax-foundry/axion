# Running Evaluations

Axion provides evaluation runners for batch processing with caching and parallel execution.

## Quick Start

```python
from axion import Dataset
from axion.metrics import Faithfulness, AnswerRelevancy
from axion.runners import evaluation_runner

# Load dataset
dataset = Dataset.from_csv("eval_data.csv")

# Run evaluation
results = await evaluation_runner(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy()]
)

# View results
df = results.to_dataframe()
results.to_scorecard(display_in_notebook=True)
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
# Convert to DataFrame for analysis
df = results.to_dataframe()

# Generate detailed metric summary report
from axion.runners.summary import MetricSummary
MetricSummary().execute(results.results, total_time=100)

# Per-test results
for test_result in results.results:
    for score in test_result.score_results:
        print(f"{score.name}: {score.score}")

# Find failures
df[df['metric_score'] < 0.5]

# Visual scorecard (Jupyter notebooks)
results.to_scorecard(display_in_notebook=True)

# Latency analysis
results.to_latency_plot()
```

### Normalized DataFrames for Database Loads

When loading evaluation results into a database, use `to_normalized_dataframes()` instead of `to_dataframe()` to avoid data duplication. This returns two separate tables following data engineering best practices:

- **Dataset Items Table**: One row per test case (inputs/ground truth)
- **Metric Results Table**: One row per metric score, with foreign key to dataset item

```python
# Get normalized tables
dataset_df, metrics_df = results.to_normalized_dataframes()

# Load into database (example with pandas)
dataset_df.to_sql('dataset_items', engine, if_exists='append', index=False)
metrics_df.to_sql('metric_results', engine, if_exists='append', index=False)

# The 'dataset_id' column in metrics_df is a foreign key to dataset_df['dataset_id']
# This enables efficient joins and prevents duplicating dataset fields
```

**Why use normalized tables?**

| Approach | Rows for 100 items × 5 metrics | Dataset fields duplicated? |
|----------|-------------------------------|---------------------------|
| `to_dataframe()` | 500 rows | Yes (5× per item) |
| `to_normalized_dataframes()` | 100 + 500 rows | No |

**Merging back to denormalized view:**

```python
# If you need the denormalized view later (by_alias=True is the default)
merged_df = metrics_df.merge(dataset_df, on='dataset_id', how='left')
# This produces the same columns as to_dataframe(), just different column order
```

**Column naming with `by_alias`:**

```python
# by_alias=True (default): Uses descriptive aliases to avoid conflicts
# - dataset_id: DatasetItem's unique identifier
# - dataset_metadata: DatasetItem's metadata
# - metric_id: MetricScore's unique identifier
# - metric_metadata: MetricScore's metadata
dataset_df, metrics_df = results.to_normalized_dataframes(by_alias=True)

# by_alias=False: Uses original field names
# - id: Used for both DatasetItem FK (MetricScore's id is removed to avoid conflict)
dataset_df, metrics_df = results.to_normalized_dataframes(by_alias=False)
merged_df = metrics_df.merge(dataset_df, on='id', how='left')
```

### Summary Classes

Axion provides summary classes for different reporting needs:

| Class | Description |
|-------|-------------|
| `MetricSummary` | Detailed metric analysis with performance insights and distribution charts |
| `SimpleSummary` | High-level KPIs and business impact dashboard |
| `HierarchicalSummary` | Summary for hierarchical evaluation trees |

```python
from axion.runners.summary import MetricSummary, SimpleSummary

# Detailed analysis
MetricSummary(show_distribution=True).execute(results.results, total_time=100)

# Simple KPI dashboard
SimpleSummary().execute(results.results, total_time=100)
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
