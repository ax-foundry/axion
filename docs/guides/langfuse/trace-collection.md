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
    name='baseball-rules-agent',
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
trace.step_names    # ['rule-lookup', 'play-analysis', 'ruling']

# Access a step by name
step = trace.ruling

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
step = trace['rule-lookup']
step.generation.output
```

### Fuzzy Step Names

Step name resolution is case and separator insensitive:

```python
# All of these resolve to the same step
trace.ruling            # Exact match
trace.Ruling            # Case-insensitive
trace.play_analysis     # Matches 'playAnalysis'
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

class BaseballRulesPatterns(PromptPatternsBase):
    @classmethod
    def _patterns_ruling(cls) -> Dict[str, str]:
        h_situation = 'GAME SITUATION'
        h_rules = 'APPLICABLE RULES'
        h_precedents = 'HISTORICAL PRECEDENTS'
        return {
            'game_situation': create_extraction_pattern(
                h_situation, re.escape(h_rules)
            ),
            'applicable_rules': create_extraction_pattern(
                h_rules, re.escape(h_precedents)
            ),
            'historical_precedents': create_extraction_pattern(
                h_precedents, r'$'
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
    prompt_patterns=BaseballRulesPatterns,
)

# Extract variables from a step
variables = collection[0].ruling.extract_variables()
# {'game_situation': 'runners on first and second, one out', 'applicable_rules': '...', ...}

# Or access via dot-notation
collection[0].ruling.variables
```

!!! note "Hyphenated Step Names"
    Step names with hyphens or special characters are normalized when looking up pattern methods. For example, `rule-lookup` maps to `_patterns_rule_lookup()`.

---

## Observation Tree

While steps group observations by name, the **observation tree** reconstructs the parent/child hierarchy from `parent_observation_id` fields -- the same hierarchy visible in the Langfuse timeline UI.

### Accessing the Tree

```python
trace = collection[0]

# All root nodes (list, may have multiple roots)
roots = trace.tree_roots

# Convenience: single root when exactly one exists, else None
root = trace.tree

# Walk the entire tree (pre-order depth-first)
for node in root.walk():
    print("  " * node.depth + node.name)
```

### ObservationNode Properties

Each node wraps an observation and adds tree structure:

```python
node = trace.tree

# Tree structure
node.parent          # Parent node (None for roots)
node.children        # List of child nodes (sorted by start_time)
node.is_root         # True if no parent
node.is_leaf         # True if no children
node.depth           # Distance from root (0 for roots)

# Timing
node.start_time      # datetime (parsed from ISO string if needed)
node.end_time        # datetime
node.duration        # timedelta (end_time - start_time)

# Observation data (SmartAccess delegation)
node.name            # Observation name
node.type            # SPAN, GENERATION, etc.
node.input           # Observation input
node.output          # Observation output
```

### Searching and Navigating

`find()` searches the subtree for the first matching descendant:

```python
root = trace.tree

# Find by name
gen = root.find(name='recommendation:ai.generateText')

# Find by type
first_gen = root.find(type='GENERATION')

# Find by both (AND)
specific = root.find(name='ruling', type='GENERATION')

# Returns None when no match
root.find(name='nonexistent')  # None
```

Bracket access searches descendants by name first, then falls back to observation field lookup:

```python
# Find a descendant node by name
gen_node = root['recommendation:ai.generateText']

# Falls back to observation field when no descendant matches
trace_id = root['id']

# Raises KeyError when neither found
root['nonexistent']  # KeyError
```

### Iteration and Containment

Nodes support standard Python iteration protocols:

```python
root = trace.tree

# Iterate direct children
for child in root:
    print(child.name, child.type)

# Number of direct children
len(root)            # 3

# Check if a name exists anywhere in the subtree
'recommendation:ai.generateText' in root  # True
```

### Collection-Level Access

```python
# Get tree_roots for every trace in the collection
for roots in collection.trees:
    for root in roots:
        print(root.name, [c.name for c in root.children])
```

!!! note "`from_langfuse()` filter behavior"
    When using `from_langfuse()`, the `name=` filter parameter is ignored when `trace_ids` is provided. Trace IDs take precedence and bypass all other filters.

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
by_name = collection.filter_by(name='baseball-rules-agent')
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
    prompt_patterns=BaseballRulesPatterns,
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

    def extract_ruling(trace):
        step = trace.ruling
        gen = step.generation

        return DatasetItem(
            id=str(trace.id),
            query=f'What is the correct ruling for play {trace.id}?',
            actual_output=gen.output.get('ruling', ''),
            trace_id=str(trace.id),
            observation_id=str(gen.id),
            additional_output={
                'explanation': gen.output.get('explanation', ''),
                'rule_citations': gen.output.get('rule_citations', []),
            },
        )

    dataset = collection.to_dataset(
        name='rulings-eval',
        transform=extract_ruling,
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

This example demonstrates the full workflow: fetch traces from a baseball rules agent, explore with patterns, convert to dataset, evaluate, and publish.

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
class BaseballRulesPatterns(PromptPatternsBase):
    @classmethod
    def _patterns_ruling(cls) -> Dict[str, str]:
        h_situation = 'GAME SITUATION'
        h_rules = 'APPLICABLE RULES'
        return {
            'game_situation': create_extraction_pattern(
                h_situation, re.escape(h_rules)
            ),
            'applicable_rules': create_extraction_pattern(
                h_rules, r'$'
            ),
        }


# 2. Fetch traces
loader = LangfuseTraceLoader()
collection = TraceCollection.from_langfuse(
    trace_ids=['abc123', 'def456'],
    loader=loader,
    prompt_patterns=BaseballRulesPatterns,
)

# 3. Explore
for trace in collection:
    print(trace.id, trace.step_names)
    if 'ruling' in trace.step_names:
        print('  variables:', trace.ruling.variables)


# 4. Define extraction transform
def extract_ruling(trace):
    return DatasetItem(
        id=str(trace.id),
        query=str(trace.input),
        actual_output=str(trace.output),
        trace_id=str(trace.id),
    )


# 5. Convert to Dataset and evaluate
dataset = collection.to_dataset(name='baseball-rules-eval', transform=extract_ruling)

result = await evaluation_runner(
    evaluation_inputs=dataset,
    scoring_metrics=[AnswerRelevancy(), Faithfulness()],
    evaluation_name='Baseball Rules Evaluation',
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
| `trace.tree_roots` | List of root `ObservationNode`s (hierarchy) |
| `trace.tree` | Single root node if exactly one root, else `None` |
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

### ObservationNode

| Property / Method | Description |
|-------------------|-------------|
| `node.observation` | Underlying `ObservationsView` |
| `node.parent` | Parent node (`None` for roots) |
| `node.children` | Child nodes (sorted by `start_time`) |
| `node.is_root` | `True` if no parent |
| `node.is_leaf` | `True` if no children |
| `node.depth` | Distance from root |
| `node.start_time` | Parsed `datetime` |
| `node.end_time` | Parsed `datetime` |
| `node.duration` | `timedelta` (end - start) |
| `node.walk()` | Pre-order depth-first generator |
| `node.find(name, type)` | First descendant matching name and/or type, or `None` |
| `node['name']` | Descendant by name; falls back to observation field; raises `KeyError` |
| `for child in node` | Iterate direct children |
| `len(node)` | Number of direct children |
| `'name' in node` | `True` if any descendant has that name |

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
