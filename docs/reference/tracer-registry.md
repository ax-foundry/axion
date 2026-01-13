# Tracer Registry API Reference

Registry pattern for tracing providers with support for NoOp, Logfire, Langfuse, and Opik backends.

## TracerRegistry

::: axion._core.tracing.registry.TracerRegistry
    options:
      show_root_heading: true

## BaseTracer

::: axion._core.tracing.registry.BaseTracer
    options:
      show_root_heading: true

## Built-in Tracers

### NoOpTracer

::: axion._core.tracing.noop.tracer.NoOpTracer
    options:
      show_root_heading: true
      members:
        - create
        - span
        - async_span

### LogfireTracer

::: axion._core.tracing.logfire.tracer.LogfireTracer
    options:
      show_root_heading: true
      members:
        - create
        - span
        - async_span
        - flush
        - log_llm_call
        - log_retrieval_call
        - log_evaluation
        - log_database_query
        - get_statistics
        - display_statistics

### LangfuseTracer

::: axion._core.tracing.langfuse.tracer.LangfuseTracer
    options:
      show_root_heading: true
      members:
        - create
        - span
        - async_span
        - flush
        - log_llm_call
        - log_evaluation

### OpikTracer

::: axion._core.tracing.opik.tracer.OpikTracer
    options:
      show_root_heading: true
      members:
        - create
        - span
        - async_span
        - flush
        - shutdown
        - log_llm_call
        - log_evaluation
