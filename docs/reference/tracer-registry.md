# Tracer Registry API Reference

Registry pattern for tracing providers with support for NoOp, Logfire, Langfuse, and Opik backends.

<div class="ref-import" markdown>

```python
from axion._core.tracing.registry import TracerRegistry, BaseTracer
from axion._core.tracing.noop.tracer import NoOpTracer
from axion._core.tracing.logfire.tracer import LogfireTracer
from axion._core.tracing.langfuse.tracer import LangfuseTracer
from axion._core.tracing.opik.tracer import OpikTracer
```

</div>

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">N</span>
<p class="rule-card__title">NoOpTracer</p>
<p class="rule-card__desc">Zero-overhead tracer for tests and production when observability is not needed.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">L</span>
<p class="rule-card__title">LogfireTracer</p>
<p class="rule-card__desc">OpenTelemetry-based observability with Logfire integration for detailed span tracing.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">F</span>
<p class="rule-card__title">LangfuseTracer</p>
<p class="rule-card__desc">LLM-specific observability with cost tracking, prompt management, and evaluation logging.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">O</span>
<p class="rule-card__title">OpikTracer</p>
<p class="rule-card__desc">Comet Opik integration for experiment tracking, LLM call logging, and evaluation traces.</p>
</div>
</div>

---

## TracerRegistry

::: axion._core.tracing.registry.TracerRegistry
    options:
      show_root_heading: true

## BaseTracer

::: axion._core.tracing.registry.BaseTracer
    options:
      show_root_heading: true

---

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

---

<div class="ref-nav" markdown>

[Tracing Deep Dive :octicons-arrow-right-24:](../deep-dives/internals/tracing.md){ .md-button .md-button--primary }
[Langfuse Guide :octicons-arrow-right-24:](../guides/langfuse/overview.md){ .md-button }

</div>
