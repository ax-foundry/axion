---
icon: custom/analytics
---
# Creating Traces

This guide covers how to create and manage traces in Langfuse using Axion's tracing system.

## Using the @trace Decorator

The `@trace` decorator automatically captures function inputs and outputs:

```python
from axion._core.tracing import init_tracer, trace

class RAGService:
    def __init__(self):
        self.tracer = init_tracer('llm')

    @trace(name='rag-query', capture_args=True, capture_result=True)
    async def query(self, question: str, context: list[str]) -> str:
        # Your RAG logic here
        response = await self.llm.generate(question, context)
        return response

# Usage
service = RAGService()
answer = await service.query(
    question='What is the return policy?',
    context=['Returns accepted within 30 days...']
)
service.tracer.flush()
```

### Decorator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Function name | Name for the span |
| `capture_args` | `bool` | `False` | Capture function arguments as input |
| `capture_result` | `bool` | `False` | Capture return value as output |

## Manual Span Creation

For more control over what gets captured, create spans manually:

### Synchronous Context Manager

```python
from axion._core.tracing import Tracer
from openai import OpenAI

tracer = Tracer('llm')
client = OpenAI()

with tracer.span('my-operation') as span:
    span.set_input({'query': 'How do I upgrade my plan?'})

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'user', 'content': 'How do I upgrade my plan?'}]
    )

    span.set_output({'response': response.choices[0].message.content})
    span.set_attribute('model', 'gpt-4o')
    span.set_attribute('tokens', response.usage.total_tokens)

tracer.flush()
```

### Asynchronous Context Manager

For async code, use `async_span`:

```python
async with tracer.async_span('async-operation') as span:
    span.set_input({'query': user_query})
    result = await process_query(user_query)
    span.set_output({'result': result})
```

## Input/Output Capture

### set_input() and set_output()

Use these methods to explicitly capture data on spans:

```python
with tracer.span('llm-call') as span:
    # Set input data (dict)
    span.set_input({
        'query': user_question,
        'context': retrieved_chunks,
        'model': 'gpt-4o'
    })

    # Your LLM call
    response = llm.generate(...)

    # Set output data (dict)
    span.set_output({
        'response': response.text,
        'tokens_used': response.usage.total_tokens
    })
```

### set_attribute()

Add additional metadata to spans:

```python
with tracer.span('retrieval') as span:
    span.set_attribute('num_chunks', 5)
    span.set_attribute('search_type', 'semantic')
    span.set_attribute('latency_ms', 142)
```

## Nested Spans

Create hierarchical traces with nested spans:

```python
with tracer.span('rag-pipeline') as parent:
    parent.set_input({'query': query})

    # Child span for retrieval
    with tracer.span('retrieval') as retrieval_span:
        chunks = retriever.search(query)
        retrieval_span.set_output({'chunks': len(chunks)})

    # Child span for generation
    with tracer.span('generation') as gen_span:
        gen_span.set_input({'context': chunks})
        response = llm.generate(query, chunks)
        gen_span.set_output({'response': response})

    parent.set_output({'answer': response})
```

## Flushing Traces

!!! tip "Always Flush Before Exit"
    Call `tracer.flush()` before your application exits to ensure all traces are sent to Langfuse. This is especially important in scripts and short-lived processes.

```python
tracer = Tracer('llm')

# Your tracing operations
with tracer.span('operation') as span:
    span.set_input({'query': 'test'})
    # ...

# Ensure traces are sent
tracer.flush()
```

## Fetching Traces

Use `LangfuseTraceLoader` to retrieve traces from Langfuse:

```python
from axion.tracing import LangfuseTraceLoader

loader = LangfuseTraceLoader()

# Fetch recent traces
traces = loader.fetch_traces(
    limit=100,          # Maximum traces to fetch
    days_back=7,        # Time window in days
    tags=['prod'],      # Filter by tags (optional)
    name='rag-query',   # Filter by trace name (optional)
)
```

### fetch_traces() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | `int` | `50` | Maximum number of traces to fetch |
| `mode` | `str` | `'days_back'` | Time window mode: `days_back`, `hours_back`, `absolute` |
| `days_back` | `int` | `7` | Number of days to look back (days_back mode) |
| `hours_back` | `int` | `24` | Number of hours to look back (hours_back mode) |
| `from_timestamp` | `datetime \| str \| None` | `None` | Start timestamp (absolute mode, ISO string supported) |
| `to_timestamp` | `datetime \| str \| None` | `None` | End timestamp (absolute mode, ISO string supported) |
| `tags` | `list[str]` | `None` | Filter by specific tags |
| `name` | `str` | `None` | Filter by trace name |
| `fetch_full_traces` | `bool` | `True` | Fetch full details vs. summaries |
| `**trace_list_kwargs` | `dict` | `{}` | Extra kwargs passed to `langfuse_client.api.trace.list(...)` |

### Filtering Examples

```python
# Filter by multiple tags (AND logic)
prod_traces = loader.fetch_traces(
    limit=100,
    tags=['production', 'v2.0']
)

# Filter by trace name
rag_traces = loader.fetch_traces(
    limit=100,
    name='rag-query'
)

# Combine filters
traces = loader.fetch_traces(
    limit=50,
    days_back=3,
    tags=['production'],
    name='chat-completion'
)
```

### Absolute Window Example

```python
from datetime import datetime, timezone

traces = loader.fetch_traces(
    mode='absolute',
    from_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    to_timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
    tags=['prod'],
)
```

## Converting Traces to Dataset

Traces must be converted to `DatasetItem` objects for evaluation. The key is to preserve `trace_id` and optionally `observation_id` for score publishing.

!!! tip "Use TraceCollection for Rich Exploration"
    For step-based navigation, dot-notation access, prompt variable extraction, and built-in dataset conversion, see **[Trace Collection](trace-collection.md)**. The manual approach below works for simple cases.

### Understanding Trace Structure

Langfuse traces contain:

- `id`: Unique trace identifier
- `input`: The input data (dict or string)
- `output`: The output data (dict or string)
- `observations`: List of spans within the trace
- `tags`: Associated tags
- `metadata`: Additional metadata

### Manual Conversion

```python
from axion import Dataset, DatasetItem

items = []
for trace in traces:
    # Extract query from input
    query = ''
    if trace.input:
        if isinstance(trace.input, dict):
            query = trace.input.get('query', trace.input.get('question', ''))
        else:
            query = str(trace.input)

    # Extract response from output
    actual_output = ''
    if trace.output:
        if isinstance(trace.output, dict):
            actual_output = trace.output.get('response', trace.output.get('answer', ''))
        else:
            actual_output = str(trace.output)

    # Create DatasetItem with trace_id for score publishing
    items.append(DatasetItem(
        id=trace.id,
        query=query,
        actual_output=actual_output,
        trace_id=trace.id,  # Required for publish_to_observability()
    ))

dataset = Dataset(items=items)
```

### Using DataFrame Conversion

For more complex conversions, use `Dataset.read_dataframe()`:

```python
import pandas as pd
from axion import Dataset

# Convert traces to DataFrame
data = []
for trace in traces:
    data.append({
        'id': trace.id,
        'query': trace.input.get('query', '') if trace.input else '',
        'actual_output': trace.output.get('response', '') if trace.output else '',
        'trace_id': trace.id,
        'retrieved_content': trace.input.get('context', []) if trace.input else [],
    })

df = pd.DataFrame(data)

# Convert to Dataset
dataset = Dataset.read_dataframe(df, ignore_extra_keys=True)
```

### Preserving Observation IDs

For granular scoring at the span level, extract observation IDs:

```python
items = []
for trace in traces:
    # Find the generation span for granular scoring
    obs_id = None
    for obs in trace.observations or []:
        if obs.type == 'GENERATION':
            obs_id = obs.id
            break

    items.append(DatasetItem(
        id=trace.id,
        query=extract_query(trace.input),
        actual_output=extract_output(trace.output),
        trace_id=trace.id,
        observation_id=obs_id,  # Scores attach to this span
    ))
```

## Performance Tips

!!! tip "Fetching Large Volumes"
    Set `fetch_full_traces=False` when fetching large volumes of traces. This returns trace summaries instead of full details, significantly reducing API calls and avoiding rate limits.

```python
# Fast fetch for large volumes
traces = loader.fetch_traces(
    limit=1000,
    fetch_full_traces=False  # Returns summaries only
)
```

### Empty Traces

If `fetch_traces()` returns an empty list:

1. **Extend time window:**
   ```python
   traces = loader.fetch_traces(days_back=30)
   ```

2. **Verify tags exist:**
   ```python
   # Fetch without tag filter first
   all_traces = loader.fetch_traces(limit=10, tags=None)
   print(f"All traces: {len(all_traces)}")
   ```

3. **Ensure traces were flushed:**
   ```python
   tracer.flush()  # Call after tracing operations
   ```

## Next Steps

- **[Trace Collection](trace-collection.md)**: Rich trace exploration with dot-notation, step navigation, and dataset conversion
- **[Publishing](publishing.md)**: Publish evaluation scores to Langfuse
- **[Configuration](configuration.md)**: Advanced configuration options
- **[Overview](overview.md)**: Complete workflow example
