# Using Custom Tags and Environment Variables with Langfuse

This guide explains how to use custom tags and environment variables when working with Langfuse in Axion.

## Quick Start

```python
import os
from axion._core.tracing import Tracer, configure_tracing
from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader

# Set environment variables
os.environ['LANGFUSE_TAGS'] = 'prod,v1.0'
os.environ['LANGFUSE_ENVIRONMENT'] = 'production'
os.environ['LANGFUSE_DEFAULT_TAGS'] = 'evaluation,automated'

# Configure tracing
configure_tracing()
tracer = Tracer('my-service')

# Use tracer - tags and environment applied automatically
with tracer.span('my-operation'):
    # Your code
    pass

# Loader uses tags env var automatically when pushing scores
loader = LangfuseTraceLoader()
result = evaluation_runner(...)
stats = loader.push_scores_to_langfuse(result)  # Uses LANGFUSE_TAGS env var
# Note: Environment cannot be set when pushing scores - it must be set at tracer initialization
```

## Overview

Langfuse supports two main ways to organize and filter traces:
1. **Tags** - Labels attached to traces and observations for filtering and organization
2. **Environment** - Environment identifier (e.g., "production", "staging") set at the trace level

## Using Tags

### In LangfuseTraceLoader (for fetching traces and pushing scores)

Tags are fully supported when fetching traces and pushing evaluation scores back to Langfuse.

#### Setting Default Tags via Environment Variable

```bash
export LANGFUSE_DEFAULT_TAGS="prod,v1.0,experiment"
```

#### Setting Default Tags in Code

```python
from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader

# Set default tags that will be applied to all scores
loader = LangfuseTraceLoader(default_tags=['prod', 'v1.0'])

# Fetch traces filtered by tags
traces = loader.fetch_traces(limit=100, tags=['prod'])

# Push scores with additional tags
result = evaluation_runner(...)
stats = loader.push_scores_to_langfuse(
    result,
    tags=['experiment', 'custom-tag']  # Merged with default_tags
)

# Or rely on environment variables (fallback if not provided)
# os.environ['LANGFUSE_TAGS'] = 'prod,v1.0'
stats = loader.push_scores_to_langfuse(result)  # Uses LANGFUSE_TAGS env var automatically
```

#### How Tags Work in the Loader

- **Default tags**: Set via `default_tags` parameter or `LANGFUSE_DEFAULT_TAGS` env var
- **Per-call tags**: Passed to `push_scores_to_langfuse()` or `fetch_traces()`
- **Merging**: Per-call tags are merged with default tags
- **Storage**: Tags are stored in the `metadata.tags` field of scores

#### Important Note About Environment

**Environment cannot be set when pushing scores to existing traces.** Langfuse SDK v3 doesn't support updating trace environment after creation.

Environment must be set at **client initialization** when creating traces:
- Via `LANGFUSE_ENVIRONMENT` or `LANGFUSE_TRACING_ENVIRONMENT` environment variables
- Via the `environment` parameter when creating `LangfuseTracer`

**Tags Parameter Fallback**: The `tags` parameter falls back to `LANGFUSE_TAGS` environment variable if not provided:

```python
import os

# Set environment variable for tags
os.environ['LANGFUSE_TAGS'] = 'prod,v1.0'

# Tags will be used automatically if parameter is not provided
stats = loader.push_scores_to_langfuse(result)  # Uses LANGFUSE_TAGS env var
stats = loader.push_scores_to_langfuse(result, tags=['override'])  # Overrides env var
```

### In LangfuseTracer (for creating traces)

Tags can be set at the tracer level or per-span level.

#### Setting Tags via Environment Variable

```bash
export LANGFUSE_TAGS="prod,v1.0"
```

#### Setting Tags in Code

```python
from axion._core.tracing import Tracer, configure_tracing

# Method 1: Via environment variable
import os
os.environ['LANGFUSE_TAGS'] = 'prod,v1.0'
configure_tracing()
tracer = Tracer('my-service')

# Method 2: Directly on tracer (if using LangfuseTracer directly)
from axion._core.tracing.langfuse.tracer import LangfuseTracer

tracer = LangfuseTracer(tags=['prod', 'v1.0'])

# Method 3: Per-span tags
with tracer.span('my-operation', tags=['custom-tag', 'another-tag']):
    # Your code here
    pass
```

#### Tag Inheritance

- Tracer-level tags apply to all traces created by that tracer
- Span-level tags (on root span) override/extend tracer-level tags
- Tags are set at the **trace level** using `update_current_trace()`, not on individual observations
- Only tags set on the root span (first span in a trace) are applied to the trace

## Using Environment Variables

Environment variables help identify which environment a trace came from (production, staging, development, etc.).

### Setting Environment via Environment Variable

```bash
export LANGFUSE_ENVIRONMENT="production"
```

### Setting Environment in Code

```python
from axion._core.tracing import Tracer, configure_tracing

# Method 1: Via environment variable
import os
os.environ['LANGFUSE_ENVIRONMENT'] = 'production'
configure_tracing()
tracer = Tracer('my-service')

# Method 2: Directly on tracer
from axion._core.tracing.langfuse.tracer import LangfuseTracer

tracer = LangfuseTracer(environment='production')

# Method 3: Per-span (only affects root trace)
with tracer.span('my-operation', environment='staging'):
    # Your code here
    pass
```

### How Environment Works

- Environment is set at **client initialization**, not when pushing scores to existing traces
- The environment applies to all traces created by that client instance
- Environment **cannot** be updated after a trace is created
- Environment can be set via:
  - `LANGFUSE_ENVIRONMENT` env var (Axion custom)
  - `LANGFUSE_TRACING_ENVIRONMENT` env var (Langfuse SDK standard)
  - `environment` parameter when creating `LangfuseTracer`
- Environment helps filter and organize traces in Langfuse UI

**Important**: If you need to set environment for traces, configure it before creating the traces. The `push_scores_to_langfuse()` method does not accept an `environment` parameter because it cannot update existing traces.

## Complete Example

```python
import os
from axion._core.tracing import Tracer, configure_tracing
from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader

# Set up environment variables
os.environ['TRACING_MODE'] = 'langfuse'
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-...'
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-...'
os.environ['LANGFUSE_TAGS'] = 'prod,v1.0'
os.environ['LANGFUSE_ENVIRONMENT'] = 'production'
os.environ['LANGFUSE_DEFAULT_TAGS'] = 'evaluation,automated'

# Configure tracing
configure_tracing()

# Create tracer with tags and environment
tracer = Tracer('my-service')

# Create spans with additional tags
with tracer.span('llm-call', tags=['custom-operation'], model='gpt-4'):
    # Your LLM call here
    response = llm.generate("Hello")

# Later, fetch traces and push scores
loader = LangfuseTraceLoader()

# Fetch traces filtered by tags
traces = loader.fetch_traces(limit=100, tags=['prod'])

# Convert to dataset and evaluate
dataset = loader.to_dataset(name='my-eval', limit=100)
result = evaluation_runner(
    evaluation_inputs=dataset,
    scoring_metrics=[Faithfulness(), AnswerRelevancy()]
)

# Push scores back with tags
stats = loader.push_scores_to_langfuse(
    result,
    tags=['experiment-123']  # Merged with default_tags
)
print(f"Uploaded {stats['uploaded']} scores")

# Note: Environment cannot be set when pushing scores.
# It must be configured at client initialization when creating traces.
```

## Environment Variables Reference

| Variable | Description | Used By |
|----------|-------------|---------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | Tracer, Loader |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | Tracer, Loader |
| `LANGFUSE_BASE_URL` | Langfuse API endpoint | Tracer, Loader |
| `LANGFUSE_TAGS` | Comma-separated tags for traces | Tracer |
| `LANGFUSE_ENVIRONMENT` | Environment name (e.g., "production") - Axion custom | Tracer |
| `LANGFUSE_TRACING_ENVIRONMENT` | Environment name - Langfuse SDK standard | Tracer |
| `LANGFUSE_DEFAULT_TAGS` | Default tags for scores | Loader |

## Best Practices

1. **Use consistent tag naming**: Use lowercase with hyphens (e.g., `prod-v1`, `experiment-123`)
2. **Set environment at tracer initialization**: Environment must be set when creating traces, not when pushing scores. Use environment variables or tracer initialization for consistency
3. **Use tags for filtering**: Tags are great for filtering traces in Langfuse UI
4. **Combine default and per-call tags**: Set common tags as defaults, add specific ones per call
5. **Environment for deployments**: Set environment based on your deployment (production, staging, dev) at client initialization
6. **Tags can be set when pushing scores**: Unlike environment, tags can be added when pushing scores via the `tags` parameter

## Filtering Traces by Tags

In Langfuse UI, you can filter traces by:
- Tags (set on traces/observations)
- Environment (set on traces)
- Name
- Time range

The `fetch_traces()` method supports filtering by tags:

```python
# Fetch only traces with 'prod' tag
prod_traces = loader.fetch_traces(limit=100, tags=['prod'])

# Fetch traces with multiple tags (AND logic)
traces = loader.fetch_traces(limit=100, tags=['prod', 'v1.0'])
```
