---
icon: custom/sliders
---
# Metrics API Reference

Evaluation metrics for AI agents — composable, LLM-powered and heuristic scoring.

<div class="ref-import" markdown>

```python
from axion import metric_registry
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics import (
    Faithfulness, AnswerRelevancy, FactualAccuracy,
    AnswerCompleteness, AnswerCriteria,
    ContextualRelevancy, ContextualPrecision, ContextualRecall,
    ExactStringMatch, CitationPresence, Latency,
    HitRateAtK, MeanReciprocalRank,
    GoalCompletion, ConversationFlow,
)
```

</div>

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">B</span>
<p class="rule-card__title">BaseMetric</p>
<p class="rule-card__desc">Base class for all metrics. Provides LLM integration, field validation, structured I/O, and the <code>execute()</code> contract.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">R</span>
<p class="rule-card__title">MetricRegistry</p>
<p class="rule-card__desc">Global registry for storing, retrieving, and discovering metric classes by key, tag, or compatible fields.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">@</span>
<p class="rule-card__title">@metric</p>
<p class="rule-card__desc">Decorator that attaches config (name, fields, threshold, tags) and auto-registers the class.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">30+</span>
<p class="rule-card__title">Built-in Metrics</p>
<p class="rule-card__desc">Composite (LLM-judged), heuristic, retrieval, and conversational metrics ready to use out of the box.</p>
</div>
</div>

---

## BaseMetric

::: axion.metrics.base.BaseMetric
    options:
      show_root_heading: true
      members:
        - name
        - description
        - threshold
        - input_item
        - required_fields
        - optional_fields
        - metric_category
        - execute
        - get_field
        - get_mapped_fields
        - get_evaluation_fields
        - set_instruction
        - set_examples
        - add_examples
        - compute_cost_estimate
        - display_prompt
        - get_sub_metrics

---

## MetricRegistry

::: axion.metrics.MetricRegistry
    options:
      show_root_heading: true
      members:
        - register
        - get
        - find
        - get_compatible_metrics
        - get_metric_descriptions
        - display
        - display_table

---

## @metric decorator

::: axion.metrics.base.metric
    options:
      show_root_heading: true

---

## Composite Metrics

### Faithfulness

::: axion.metrics.Faithfulness
    options:
      show_root_heading: true

### AnswerRelevancy

::: axion.metrics.AnswerRelevancy
    options:
      show_root_heading: true

### FactualAccuracy

::: axion.metrics.FactualAccuracy
    options:
      show_root_heading: true

### AnswerCompleteness

::: axion.metrics.AnswerCompleteness
    options:
      show_root_heading: true

### AnswerCriteria

::: axion.metrics.AnswerCriteria
    options:
      show_root_heading: true

### ContextualRelevancy

::: axion.metrics.ContextualRelevancy
    options:
      show_root_heading: true

### ContextualPrecision

::: axion.metrics.ContextualPrecision
    options:
      show_root_heading: true

### ContextualRecall

::: axion.metrics.ContextualRecall
    options:
      show_root_heading: true

---

## Heuristic Metrics

### ExactStringMatch

::: axion.metrics.ExactStringMatch
    options:
      show_root_heading: true

### CitationPresence

::: axion.metrics.CitationPresence
    options:
      show_root_heading: true

### Latency

::: axion.metrics.Latency
    options:
      show_root_heading: true

---

## Retrieval Metrics

### HitRateAtK

::: axion.metrics.HitRateAtK
    options:
      show_root_heading: true

### MeanReciprocalRank

::: axion.metrics.MeanReciprocalRank
    options:
      show_root_heading: true

---

## Conversational Metrics

### GoalCompletion

::: axion.metrics.GoalCompletion
    options:
      show_root_heading: true

### ConversationFlow

::: axion.metrics.ConversationFlow
    options:
      show_root_heading: true

---

<div class="ref-nav" markdown>

[Metrics & Evaluation Guide :octicons-arrow-right-24:](../guides/metrics.md){ .md-button .md-button--primary }
[Creating Custom Metrics :octicons-arrow-right-24:](../deep-dives/metrics/creating-metrics.md){ .md-button }
[Metric Registry :octicons-arrow-right-24:](../metric-registry/composite/index.md){ .md-button }

</div>
