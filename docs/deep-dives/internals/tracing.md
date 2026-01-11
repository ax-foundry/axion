# Axion Tracing System

Simple observability for AI applications with automatic context management with Logfire integration for OpenTelemetry backend.

### Why Use Axion Tracing?

- **Zero setup** - Configure once, trace everywhere
- **Automatic context** - No manual tracer passing between functions
- **AI-optimized** - Built-in support for LLM, evaluation, and knowledge operations
- **Production ready** - NOOP mode for zero overhead, Logfire for rich observability

## Quick Start

```python
from axion._core.tracing import init_tracer, trace

class MyService:
    def __init__(self):
        self.tracer = init_tracer('llm')

    @trace(name='internal_span', capture_result=True)
    async def process(self, data: dict):
        return 100

    @trace(name='span', capture_result=True)
    async def run(self):
        # Set manual span for tracing
        async with self.tracer.async_span("manual_span") as span:
            # set attribute on span
            span.set_attribute("output_status", "success")
            return await self.process({"key": "value"})

await MyService().run()
```

---

## Configuration

Configuration is managed by a single, unified environment variable that is read by your Pydantic settings object.

### Environment Variable

Set `TRACING_MODE` (in standalone mode) to one of the following values:

| tracing_mode | Description                                                                                                          | Additional Env Vars Needed |
|---|----------------------------------------------------------------------------------------------------------------------|---|
| `noop` | Disables all tracing. Methods are available but do nothing. Ideal for tests or performance-critical tasks. | None |
| `logfire_local` | For local development. Sends traces to a local UI started with the `logfire dev` command.                            | None |
| `logfire_hosted` | For local development. Sends traces to the Logfire cloud service.                                                    | `LOGFIRE_TOKEN` |
| `logfire_otel` | For our production via custom environments. Sends traces to a specified OpenTelemetry (OTEL) endpoint.               | `OTEL_ENDPOINT` |

### Programmatic Configuration

You can override the global setting by passing an argument to `configure_tracing`.

```python
from axion._core.tracing import configure_tracing, TracingMode

# Configure based on the Pydantic settings object (standard way)
configure_tracing()

# Use force=True to override any previously set configuration,
# which is especially useful in testing environments.
configure_tracing(tracing_mode=TracingMode.NOOP, force=True)
```

## Patterns

### Decorator Tracing (Recommended)

The tracer attribute is required for `@trace` decorator.
The `@trace` decorator automatically looks for a tracer attribute on the class instance (`self`) to create and manage spans.

```python
from axion._core.tracing import init_tracer, trace

class DecoratorService:
    def __init__(self):

        self.tracer = init_tracer('base')

    @trace(name="process_data", capture_args=True, capture_result=True)
    async def process(self, data: dict):
        await asyncio.sleep(0.1)
        return {"status": "processed", "items": len(data)}

    @trace  # Simple usage without arguments
    async def run(self):
        result = await self.process({"id": 123, "items": ["a", "b"]})
        return result

# Usage
service = DecoratorService()
await service.run()
```

### Context-Aware Function Tracing

Use `init_tracer` at the top level of a service or class to start a new trace context. Use `get_current_tracer` in downstream functions or services that you expect to be called within an existing trace, allowing them to add child spans without needing the tracer to be passed in manually.

```python
from axion._core.tracing import get_current_tracer

class ServiceA:
    def __init__(self):
        self.tracer = init_tracer('llm')

    async def process(self):
        async with self.tracer.async_span("service_a_process"):
            # Context automatically propagates to ServiceB
            service_b = ServiceB()
            await service_b.process()

class ServiceB:
    async def process(self):
        # Get tracer from context - no manual passing needed!
        tracer = get_current_tracer()
        async with tracer.async_span("service_b_process"):
            return "processed"

# Usage
service = ServiceA()
await service.process()
```

### Complete Example

```python
import asyncio
from axion._core.tracing import init_tracer, trace
from axion.metrics import AnswerRelevancy
from axion.dataset import DatasetItem
from axion._core.metadata.schema import ToolMetadata

class MetricService:
    def __init__(self):
        # Set Context on Span
        tool_metadata = ToolMetadata(
            name="MetricService",
            description='My Service',
            version='1.0.1',
        )
        self.tracer = init_tracer(
            metadata_type='llm',
            tool_metadata=tool_metadata,
        )

    @trace(capture_result=True)
    async def run_metric(self):
        # AnswerRelevancy has its own tracer and auto-captured as a child span
        metric = AnswerRelevancy()
        data_item = DatasetItem(
            query="What is Data Cloud?",
            actual_output="Data Cloud is a hyperscale data platform to unlock value built on the Salesforce Platform.",
            expected_output="Data Cloud is a hyperscale data platform built directly into Salesforce.",
        )
        return await metric.execute(data_item)

    @trace(name="doing_work")
    async def do_some_work(self):
        await asyncio.sleep(0.5)
        return "Work done!"

    @trace(name='run_main_task')
    async def run_main_task(self):
        # Can also set manual spans
        async with self.tracer.async_span("metric_evaluation") as span:
            _ = await self.do_some_work()
            result = await self.run_metric()
            span.set_attribute("operation_status", "success")
        return result

# Usage
service = MetricService()
result = await service.run_main_task()
```

## Metadata Types

Choose the right type for automatic specialized handling:

- **`'base'`** - General operations
- **`'llm'`** - Language model calls (captures tokens, model info)
- **`'knowledge'`** - Search and retrieval (captures queries, results)
- **`'database'`** - Database operations (captures performance)
- **`'evaluation'`** - Evaluation metrics (captures scores)

## Integration

The tracing system automatically works with other Axion components:

- **Evaluation metrics** are automatically traced
- **API calls** include retry and performance data
- **LLM operations** capture token usage and model info

Just initialize your tracer and everything else traces automatically.
