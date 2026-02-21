---
icon: custom/analytics
---
# Trace Collection

`TraceCollection` provides a rich exploration layer between fetching raw traces and converting them to a `Dataset` for evaluation. It wraps raw Langfuse traces with dot-notation access, step-based navigation, filtering, serialization, and dataset conversion.

- **Dot-notation access**: Navigate nested trace data with attribute syntax
- **Step-based navigation**: Access named observations (spans, generations) as logical steps
- **Prompt variable extraction**: Extract structured variables from generation prompts via regex patterns
- **Filtering and serialization**: Filter traces by attributes, save/load JSON snapshots
- **Dataset conversion**: Convert directly to an axion `Dataset` with custom extraction logic

---

## Quick Start

```python
from axion.tracing import TraceCollection, LangfuseTraceLoader

loader = LangfuseTraceLoader()

# Fetch and wrap traces in one step
collection = TraceCollection.from_langfuse(
    trace_ids=['abc123', 'def456'],
    loader=loader,
)

# Explore traces with dot-notation
trace = collection[0]
print(trace.id)
print(trace.name)
print(trace.step_names)
```

---

## Loading Traces

### From Langfuse

The `from_langfuse()` factory fetches traces and wraps them in a single call:

```python
from axion.tracing import TraceCollection, LangfuseTraceLoader

loader = LangfuseTraceLoader(
    public_key='pk-lf-...',
    secret_key='sk-lf-...',
)

# Fetch by trace IDs
collection = TraceCollection.from_langfuse(
    trace_ids=['abc123', 'def456'],
    loader=loader,
)

# Or fetch by filters
collection = TraceCollection.from_langfuse(
    limit=100,
    days_back=7,
    tags=['production'],
    name='my-workflow',
    loader=loader,
)
```

### From Pre-fetched Traces

If you already have raw traces from `loader.fetch_traces()` or `loader.fetch_trace()`:

```python
traces = loader.fetch_traces(limit=50)
collection = TraceCollection.from_raw_traces(traces)

# Or wrap a single trace
trace = loader.fetch_trace('abc123')
collection = TraceCollection([trace])
```

### From JSON

Load a previously saved snapshot:

```python
collection = TraceCollection.load_json('traces.json')
```

---

## Exploring Traces

### Dot-Notation Access

`TraceCollection` wraps each trace in a `Trace` object that supports attribute-style access to both trace-level fields and named observation steps:

```python
trace = collection[0]

# Trace-level attributes
trace.id            # Trace ID
trace.name          # Trace name
trace.tags          # Tags list
trace.latency       # Response latency

# Fuzzy matching: snake_case and camelCase resolve to the same key
trace.created_at    # Resolves to 'createdAt' if present
```

### Step Navigation

Observations are grouped by name into steps. Access them by name directly on the trace:

```python
# List all step names
trace.step_names    # ['recommendation', 'grounding', 'location-extraction']

# Access a step by name
step = trace.recommendation

# Step properties
step.count          # Number of observations in this step
step.first          # First observation
step.last           # Last observation

# Access specific observation types within a step
step.generation     # The GENERATION observation
step.context        # Alias for the SPAN observation
step.generation.input   # Generation prompt input
step.generation.output  # Generation output
```

Bracket access also works, which is useful for step names with special characters:

```python
step = trace['location-extraction']
step.generation.output
```

### Fuzzy Step Names

Step name resolution is case and separator insensitive:

```python
# All of these resolve to the same step
trace.recommendation        # Exact match
trace.Recommendation        # Case-insensitive
trace.case_assessment       # Matches 'caseAssessment'
```

---

## Prompt Variable Extraction

For traces with structured prompts, `PromptPatternsBase` lets you define regex patterns to extract variables from generation inputs.

### Defining Patterns

Subclass `PromptPatternsBase` and define `_patterns_<step_name>()` methods:

```python
import re
from typing import Dict
from axion.tracing import PromptPatternsBase
from axion._core.tracing.collection import create_extraction_pattern

class MyPromptPatterns(PromptPatternsBase):
    @classmethod
    def _patterns_recommendation(cls) -> Dict[str, str]:
        h_assessment = 'CASE ASSESSMENT (from previous analysis)'
        h_context = 'FULL CONTEXT DATA'
        h_flags = 'UNDERWRITING FLAGS (Current Human Rules)'
        return {
            'case_assessment': create_extraction_pattern(
                h_assessment, re.escape(h_context)
            ),
            'context_data': create_extraction_pattern(
                h_context, re.escape(h_flags)
            ),
            'underwriting_flags': create_extraction_pattern(
                h_flags, r'$'
            ),
        }
```

`create_extraction_pattern(start_text, end_pattern)` builds a regex that captures text between a labelled start and a terminating pattern: `StartText:\s*(.*?)\s*(?:EndPattern)`.

### Using Patterns

Pass your patterns class when creating the collection:

```python
collection = TraceCollection.from_langfuse(
    trace_ids=['abc123'],
    loader=loader,
    prompt_patterns=MyPromptPatterns,
)

# Extract variables from a step
variables = collection[0].recommendation.extract_variables()
# {'case_assessment': 'patient is stable', 'context_data': '...', ...}

# Or access via dot-notation
collection[0].recommendation.variables
```

!!! note "Hyphenated Step Names"
    Step names with hyphens or special characters are normalized when looking up pattern methods. For example, `location-extraction` maps to `_patterns_location_extraction()`.

---

## Filtering

### Lambda Filter

```python
# Filter by any condition
prod_traces = collection.filter(lambda t: 'production' in (t.tags or []))
long_traces = collection.filter(lambda t: t.latency > 5.0)
```

### Attribute Filter

```python
# Simple attribute equality
by_name = collection.filter_by(name='my-workflow')
```

Both methods return a new `TraceCollection`.

---

## Serialization

### Save and Load JSON

```python
# Save to disk
collection.save_json('traces/snapshot.json')

# Load later (with optional patterns)
loaded = TraceCollection.load_json(
    'traces/snapshot.json',
    prompt_patterns=MyPromptPatterns,
)
```

### Raw Access

```python
# Get the underlying raw trace objects
raw_list = collection.to_list()
```

---

## Converting to Dataset

`to_dataset()` converts a `TraceCollection` into an axion `Dataset` for evaluation.

### Default Conversion

Without a transform, `to_dataset()` extracts `query` from trace input and `actual_output` from trace output using standard key detection (`query`, `question`, `input`, `response`, `answer`, etc.):

```python
dataset = collection.to_dataset(name='my-eval')
```

### Custom Transform

For complex trace structures, pass a transform function. The transform receives a `Trace` and returns either a `DatasetItem` or a `dict` of fields:

=== "Returning DatasetItem"

    ```python
    from axion.dataset import DatasetItem

    def extract_recommendation(trace):
        step = trace.recommendation
        gen = step.generation

        return DatasetItem(
            id=str(trace.id),
            query=f'Assess risk for {trace.id}',
            actual_output=gen.output.get('brief_recommendation', ''),
            trace_id=str(trace.id),
            observation_id=str(gen.id),
            additional_output={
                'detailed': gen.output.get('detailed_recommendation', ''),
            },
        )

    dataset = collection.to_dataset(
        name='recommendations',
        transform=extract_recommendation,
    )
    ```

=== "Returning dict"

    ```python
    def simple_transform(trace):
        return {
            'query': str(trace.input),
            'actual_output': str(trace.output),
            'trace_id': str(trace.id),
        }

    dataset = collection.to_dataset(
        name='simple-eval',
        transform=simple_transform,
    )
    ```

---

## End-to-End Example

This example demonstrates the full workflow: fetch traces, explore with patterns, convert to dataset, evaluate, and publish.

```python
import re
from typing import Dict

from axion.tracing import (
    LangfuseTraceLoader,
    PromptPatternsBase,
    TraceCollection,
)
from axion._core.tracing.collection import create_extraction_pattern
from axion.dataset import DatasetItem
from axion.metrics import AnswerRelevancy, Faithfulness
from axion.runners import evaluation_runner


# 1. Define prompt patterns for variable extraction
class WorkflowPatterns(PromptPatternsBase):
    @classmethod
    def _patterns_recommendation(cls) -> Dict[str, str]:
        h_assessment = 'CASE ASSESSMENT'
        h_context = 'CONTEXT DATA'
        return {
            'case_assessment': create_extraction_pattern(
                h_assessment, re.escape(h_context)
            ),
            'context_data': create_extraction_pattern(
                h_context, r'$'
            ),
        }


# 2. Fetch traces
loader = LangfuseTraceLoader()
collection = TraceCollection.from_langfuse(
    trace_ids=['abc123', 'def456'],
    loader=loader,
    prompt_patterns=WorkflowPatterns,
)

# 3. Explore
for trace in collection:
    print(trace.id, trace.step_names)
    if 'recommendation' in trace.step_names:
        print('  variables:', trace.recommendation.variables)


# 4. Define extraction transform
def extract(trace):
    return DatasetItem(
        id=str(trace.id),
        query=str(trace.input),
        actual_output=str(trace.output),
        trace_id=str(trace.id),
    )


# 5. Convert to Dataset and evaluate
dataset = collection.to_dataset(name='workflow-eval', transform=extract)

result = await evaluation_runner(
    evaluation_inputs=dataset,
    scoring_metrics=[AnswerRelevancy(), Faithfulness()],
    evaluation_name='Workflow Evaluation',
)

# 6. Publish scores back to Langfuse
result.publish_to_observability()
```

---

## API Reference

### TraceCollection

| Method | Description |
|--------|-------------|
| `from_langfuse(trace_ids, limit, days_back, tags, name, loader, prompt_patterns)` | Fetch from Langfuse and wrap |
| `from_raw_traces(raw_traces, prompt_patterns)` | Wrap pre-fetched trace objects |
| `load_json(path, prompt_patterns)` | Load from a JSON file |
| `filter(condition)` | Filter by lambda, returns new `TraceCollection` |
| `filter_by(**kwargs)` | Filter by attribute equality |
| `to_dataset(name, transform)` | Convert to axion `Dataset` |
| `save_json(path)` | Serialize to JSON file |
| `to_list()` | Return raw trace objects |
| `len(collection)` | Number of traces |
| `collection[i]` | Access by index |

### Trace

| Property / Method | Description |
|-------------------|-------------|
| `trace.step_names` | List of observation group names |
| `trace.steps` | Dict of step name to `TraceStep` |
| `trace.observations` | Flat list of all observations |
| `trace.raw` | Underlying raw trace object |
| `trace.<step_name>` | Access a step by name (fuzzy matching) |
| `trace.<attribute>` | Access trace-level attributes (fuzzy matching) |

### TraceStep

| Property / Method | Description |
|-------------------|-------------|
| `step.count` | Number of observations |
| `step.first` | First observation |
| `step.last` | Last observation |
| `step.generation` | GENERATION observation |
| `step.context` | SPAN observation (alias) |
| `step.extract_variables()` | Extract prompt variables via patterns |
| `step.variables` | Shorthand for `extract_variables()` |

### PromptPatternsBase

| Method | Description |
|--------|-------------|
| `get_for(step_name)` | Look up extraction patterns for a step |
| `_patterns_<name>()` | Override in subclass to define patterns |

### create_extraction_pattern

```python
create_extraction_pattern(start_text: str, end_pattern: str) -> str
```

Builds a regex: `StartText:\s*(.*?)\s*(?:EndPattern)`

---

## Next Steps

- **[Tracing](tracing.md)**: Creating traces with `@trace` and `fetch_traces()`
- **[Publishing](publishing.md)**: Publish evaluation scores back to Langfuse
- **[Overview](overview.md)**: Complete evaluation workflow examples
