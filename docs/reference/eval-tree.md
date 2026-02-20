---
icon: custom/git-branch
---
# Eval Tree API Reference

Hierarchical evaluation orchestration for composing complex evaluation workflows.

<div class="ref-import" markdown>

```python
from axion.eval_tree import EvalTree
```

</div>

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">T</span>
<p class="rule-card__title">EvalTree</p>
<p class="rule-card__desc">Hierarchical scoring model that builds a tree from a configuration and executes metrics using an optimized two-phase batch process.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">C</span>
<p class="rule-card__title">Config-Driven</p>
<p class="rule-card__desc">Define scoring trees via YAML, dict, or Config objects. Supports weighted aggregation, strategy overrides, and nested component hierarchies.</p>
</div>
</div>

---

## EvalTree

::: axion.eval_tree.EvalTree
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - execute
        - batch_execute
        - get_node
        - get_metric_summary
        - elapsed_time
        - summary
        - display

---

<div class="ref-nav" markdown>

[Hierarchical Scoring Guide :octicons-arrow-right-24:](../guides/hierarchical-scoring.md){ .md-button .md-button--primary }
[Running Evaluations :octicons-arrow-right-24:](../guides/evaluation.md){ .md-button }

</div>
