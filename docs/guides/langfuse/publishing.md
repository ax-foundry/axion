# Publishing Evaluation Results

After running evaluations, publish results back to Langfuse. Axion provides two methods depending on whether you're scoring existing traces or creating new experiments.

## Two Publishing Paths

| Method | Use Case | Requires Existing Traces? |
|--------|----------|---------------------------|
| `publish_to_observability()` | Attach scores to **existing** production traces | Yes - `trace_id` required |
| `publish_as_experiment()` | Create a complete experiment from scratch | No - creates everything |

---

## Publishing to Existing Traces

Use `publish_to_observability()` when you have existing traces in Langfuse and want to attach evaluation scores to them.

### Basic Usage

```python
# Publish with default settings
stats = result.publish_to_observability()
print(f"Uploaded: {stats['uploaded']}, Skipped: {stats['skipped']}")
```

### With Tags

```python
stats = result.publish_to_observability(
    tags=['experiment-v1', 'automated']
)
```

### Trace-Level Only

```python
# Scores attach to traces, not observations
stats = result.publish_to_observability(observation_id_field=None)
```

### Using LangfuseTraceLoader Directly

For more control, use the loader's method:

```python
from axion.tracing import LangfuseTraceLoader

loader = LangfuseTraceLoader(default_tags=['evaluation'])

stats = loader.push_scores_to_langfuse(
    evaluation_result=result,
    observation_id_field='observation_id',
    flush=True,
    tags=['prod', 'v1.0']  # Merged with default_tags
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loader` | `BaseTraceLoader` | `None` | Loader instance (creates one if None) |
| `observation_id_field` | `str` | `'observation_id'` | Field for granular scoring |
| `flush` | `bool` | `True` | Flush client after uploading |
| `tags` | `list[str]` | `None` | Tags to attach to scores |

### Return Statistics

```python
stats = result.publish_to_observability()
# stats = {
#     'uploaded': 45,  # Successfully uploaded scores
#     'skipped': 5,    # Skipped (missing trace_id, NaN scores)
# }
```

### Granular vs Trace-Level Scoring

**Trace-level scoring** attaches scores to the entire trace:

```python
stats = result.publish_to_observability(observation_id_field=None)
```

**Observation-level scoring** attaches scores to specific spans:

```python
# Ensure DatasetItems have observation_id set
stats = result.publish_to_observability(observation_id_field='observation_id')
```

---

## Publishing as Experiments

Use `publish_as_experiment()` for evaluation workflows that don't start with existing traces. This creates a complete experiment in Langfuse: datasets, dataset items, experiment runs, and scores.

### Basic Usage

```python
from axion.runners import evaluation_runner
from axion.metrics import Faithfulness, AnswerRelevancy

# Run evaluation on a local dataset
result = await evaluation_runner(
    evaluation_inputs=dataset,
    scoring_metrics=[Faithfulness(), AnswerRelevancy()],
    evaluation_name='Offline RAG Evaluation',
)

# Upload as a Langfuse experiment
stats = result.publish_as_experiment(
    dataset_name='my-rag-eval-dataset',
    run_name='experiment-v1',
    run_metadata={'model': 'gpt-4o', 'version': '2.0'},
    tags=['offline', 'baseline'],
)

print(f"Dataset: {stats['dataset_name']}")
print(f"Run: {stats['run_name']}")
print(f"Items created: {stats['items_created']}")
print(f"Scores uploaded: {stats['scores_uploaded']}")
```

### How It Works

```
+---------------------------------------------------------------------+
|                        publish_as_experiment()                       |
+---------------------------------------------------------------------+
|                                                                     |
|  1. DETERMINE NAMES                                                 |
|     +- dataset_name: provided OR evaluation_name OR auto-generated  |
|     +- run_name: provided OR "{dataset_name}-{run_id[:8]}"          |
|                                                                     |
|  2. CREATE/GET DATASET (upserts - safe if exists)                   |
|     +- client.create_dataset(name=dataset_name)                     |
|                                                                     |
|  3. PHASE 1: CREATE DATASET ITEMS                                   |
|     For each TestResult:                                            |
|     +- Serialize input (query, retrieved_content, etc.)             |
|     +- Serialize expected_output                                    |
|     +- create_dataset_item(id=item.id, ...)                         |
|                                                                     |
|  4. PHASE 2: CREATE EXPERIMENT RUNS                                 |
|     For each dataset_item:                                          |
|     +- Create trace with input/output                               |
|     +- Link trace to dataset item                                   |
|     +- Attach scores to trace                                       |
|                                                                     |
|  5. FINAL FLUSH                                                     |
|                                                                     |
+---------------------------------------------------------------------+
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loader` | `LangfuseTraceLoader` | `None` | Loader instance (creates one if None) |
| `dataset_name` | `str` | `None` | Name for the Langfuse dataset |
| `run_name` | `str` | `None` | Name for the experiment run |
| `run_metadata` | `dict` | `None` | Metadata for the experiment run |
| `flush` | `bool` | `True` | Flush client after uploading |
| `tags` | `list[str]` | `None` | Tags to attach to scores |
| `score_on_runtime_traces` | `bool` | `False` | Skip creating runs, just add scores to existing traces |
| `link_to_traces` | `bool` | `False` | Link experiment runs to existing traces instead of creating new ones |

### Experiment Modes

`publish_as_experiment()` supports three modes for different use cases:

| Mode | Creates Runs | Links to Traces | Use Case |
|------|-------------|-----------------|----------|
| **Default** (both `False`) | :white_check_mark: new traces | :x: | Standard experiment - creates fresh traces for each run |
| `score_on_runtime_traces=True` | :x: | N/A (scores only) | Just add scores to existing traces without creating experiment runs |
| `link_to_traces=True` | :white_check_mark: linked | :white_check_mark: | Experiment UI linked to original evaluation traces |

!!! tip "When to use each mode"
    - **Default**: Use when you don't have existing traces and want a self-contained experiment
    - **score_on_runtime_traces**: Use when you only care about scores on existing traces, not the experiment UI
    - **link_to_traces**: Use when you ran evaluations with tracing enabled and want experiment runs to link back to those original traces

#### Default Mode

Creates new traces for each dataset item run:

```python
stats = result.publish_as_experiment(
    dataset_name='my-dataset',
    run_name='experiment-v1',
)
# Creates: dataset items + new "Dataset run" traces + scores
```

#### Score on Runtime Traces

Attaches scores directly to existing traces without creating experiment runs:

```python
stats = result.publish_as_experiment(
    dataset_name='my-dataset',
    run_name='experiment-v1',
    score_on_runtime_traces=True,
)
# Creates: dataset items + scores on existing traces
# Does NOT create: experiment runs
```

This is useful when:

- You only need scores visible on your production traces
- You don't need the experiment comparison UI
- Your `DatasetItem` objects have `trace_id` set from prior tracing

#### Link to Traces

Creates experiment runs that link to your existing evaluation traces:

```python
stats = result.publish_as_experiment(
    dataset_name='my-dataset',
    run_name='experiment-v1',
    link_to_traces=True,
)
# Creates: dataset items + experiment runs linked to existing traces + scores
```

This is useful when:

- You ran `evaluation_runner` with Langfuse tracing enabled
- You want experiment runs in the Langfuse UI to link back to those original traces
- You need both the experiment comparison view AND visibility into the original trace details

!!! note "Fallback behavior"
    When `link_to_traces=True` but a `DatasetItem` doesn't have a `trace_id`, that item falls back to default mode (creates a new trace).

!!! warning "Precedence"
    If both `score_on_runtime_traces=True` and `link_to_traces=True` are set, `score_on_runtime_traces` takes precedence.

### Return Statistics

```python
stats = result.publish_as_experiment(...)
# stats = {
#     'dataset_name': 'my-rag-eval-dataset',
#     'run_name': 'experiment-v1',
#     'items_created': 50,
#     'runs_created': 50,
#     'scores_uploaded': 100,
#     'scores_skipped': 0,
#     'errors': [],
# }
```

### Behavior with Existing Names

Understanding how the method handles existing datasets and runs:

| Scenario | Behavior |
|----------|----------|
| **Dataset already exists** | `create_dataset()` upserts - retrieves existing, no error |
| **Item ID already exists** | Caught as "already exists" error, item is reused |
| **Run name already exists** | Creates a new run under the same name (distinguished by timestamp) |

### Key Design Decisions

1. **Item IDs come from Axion's `DatasetItem.id`** - Enables deduplication. Running the same evaluation twice won't duplicate items.

2. **Runs are always created fresh** - Each call creates new experiment runs, even with the same `run_name`. This lets you compare multiple runs.

3. **Dataset items are append-only** - New items are added, existing items (by ID) are reused.

### Example Scenarios

=== "First Run"

    ```python
    # Creates: dataset "my-rag-eval", 100 items, run "baseline-v1"
    result.publish_as_experiment(
        dataset_name='my-rag-eval',
        run_name='baseline-v1'
    )
    ```

=== "Same Dataset, Different Run"

    ```python
    # Reuses: dataset "my-rag-eval", existing items (by ID)
    # Creates: new run "improved-v2"
    result.publish_as_experiment(
        dataset_name='my-rag-eval',  # Same dataset
        run_name='improved-v2'       # New run name
    )
    ```

=== "Re-running Same Experiment"

    ```python
    # Reuses: dataset, items
    # Creates: NEW run also named "baseline-v1" (Langfuse shows both)
    result.publish_as_experiment(
        dataset_name='my-rag-eval',
        run_name='baseline-v1'  # Same run name
    )
    ```

!!! tip "Comparing Experiments"
    Use the same `dataset_name` with different `run_name` values to compare multiple experiments (different models, prompts, or configurations) in Langfuse's experiment comparison view.

---

## Choosing the Right Method

!!! info "See Also: Evaluation Workflows"
    The choice of publishing method depends on your evaluation workflow. See [Evaluation Workflows](overview.md#evaluation-workflows) for guidance on choosing between API-Driven, Trace-Based, and Online Production workflows.

### By Workflow

| Workflow | Primary Method | Alternative |
|----------|---------------|-------------|
| **[API-Driven](overview.md#offline-api-driven)** (black-box testing) | `publish_as_experiment()` | - |
| **[Trace-Based](overview.md#offline-trace-based)** (white-box testing) | `publish_as_experiment(link_to_traces=True)` | `publish_to_observability()` |
| **[Online Production](overview.md#online-production)** (monitoring) | `publish_to_observability()` | - |

### By Scenario

| Scenario | Use This Method |
|----------|-----------------|
| Scoring production traces | `publish_to_observability()` |
| A/B testing with existing traces | `publish_to_observability()` |
| Offline evaluation (no traces) | `publish_as_experiment()` |
| Comparing model versions | `publish_as_experiment()` |
| Creating baseline datasets | `publish_as_experiment()` |
| Continuous monitoring | `publish_to_observability()` |
| Experiment UI + link to evaluation traces | `publish_as_experiment(link_to_traces=True)` |
| Scores on traces + no experiment runs | `publish_as_experiment(score_on_runtime_traces=True)` |

### Quick Reference

```python
# For existing traces (from production):
result.publish_to_observability()  # Attaches scores to existing traces

# For new experiments (no existing traces):
result.publish_as_experiment()  # Creates everything from scratch

# For experiments linked to evaluation traces:
result.publish_as_experiment(link_to_traces=True)  # Links runs to existing traces

# For scores only (no experiment runs):
result.publish_as_experiment(score_on_runtime_traces=True)  # Scores only
```

---

## Troubleshooting

### Scores Not Appearing

If scores don't appear in the Langfuse UI:

1. **Check return stats:**
   ```python
   stats = result.publish_to_observability()
   print(f"Uploaded: {stats['uploaded']}, Skipped: {stats['skipped']}")
   ```

2. **Ensure flush completed:**
   ```python
   stats = result.publish_to_observability(flush=True)
   ```

3. **Verify trace_id matches:**
   ```python
   # trace_id must match an existing trace
   for item in dataset.items:
       print(f"Item {item.id} -> trace_id: {item.trace_id}")
   ```

4. **Check for NaN scores (these are skipped):**
   ```python
   import math
   for test_result in result.results:
       for score in test_result.score_results:
           if score.score is None or math.isnan(score.score):
               print(f"Invalid score: {score.name}")
   ```

### Missing trace_id Warnings

Scores are skipped if `trace_id` is missing:

```python
# Ensure trace_id is preserved during conversion
items.append(DatasetItem(
    id=trace.id,
    query=query,
    actual_output=output,
    trace_id=trace.id,  # Required!
))
```

Check your dataset items:
```python
for item in dataset.items:
    if not item.trace_id:
        print(f"Missing trace_id: {item.id}")
```

### Rate Limiting

For large evaluations, consider batching:

```python
# Increase delay between requests
loader = LangfuseTraceLoader(request_pacing=0.1)
stats = result.publish_to_observability(loader=loader)
```

---

## Next Steps

- **[Overview](overview.md)**: Complete workflow example
- **[Tracing](tracing.md)**: Creating traces to score
- **[Configuration](configuration.md)**: Advanced configuration
- **[Evaluation Guide](../evaluation.md)**: Running evaluations
