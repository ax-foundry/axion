---
icon: custom/play
---
# Runners API Reference

Evaluation execution engines for parallel and batch processing.

<div class="ref-import" markdown>

```python
from axion.runners import (
    evaluation_runner,
    EvaluationRunner,
    EvaluationConfig,
    MetricRunner,
)
from axion._core.cache import CacheManager, CacheConfig
```

</div>

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">E</span>
<p class="rule-card__title">evaluation_runner</p>
<p class="rule-card__desc">High-level function for running complete evaluations across datasets with multiple metrics in parallel.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">M</span>
<p class="rule-card__title">MetricRunner</p>
<p class="rule-card__desc">Lower-level runner for executing a single metric across dataset items with concurrency control.</p>
</div>
</div>

---

## evaluation_runner

::: axion.runners.evaluation_runner
    options:
      show_root_heading: true

---

## EvaluationRunner

::: axion.runners.EvaluationRunner
    options:
      show_root_heading: true

---

## EvaluationConfig

::: axion.runners.EvaluationConfig
    options:
      show_root_heading: true

---

## MetricRunner

::: axion.runners.MetricRunner
    options:
      show_root_heading: true

---

## CacheManager

::: axion._core.cache.CacheManager
    options:
      show_root_heading: true

---

## CacheConfig

::: axion._core.cache.CacheConfig
    options:
      show_root_heading: true

---

<div class="ref-nav" markdown>

[Running Evaluations Guide :octicons-arrow-right-24:](../guides/evaluation.md){ .md-button .md-button--primary }
[Evaluation Runner Deep Dive :octicons-arrow-right-24:](../deep-dives/runners/evaluation-runner.md){ .md-button }
[Metric Runner Deep Dive :octicons-arrow-right-24:](../deep-dives/runners/metric-runner.md){ .md-button }

</div>
