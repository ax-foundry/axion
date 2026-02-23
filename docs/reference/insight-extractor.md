---
icon: custom/search
---
# Insight Extractor Reference

API reference for the cross-metric insight extraction module.

<div class="ref-import" markdown>

```python
from axion.reporting import (
    InsightExtractor,
    InsightPattern,
    InsightResult,
)
```

</div>

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">I</span>
<p class="rule-card__title">InsightExtractor</p>
<p class="rule-card__desc">Bridges IssueExtractor output with EvidencePipeline for cross-metric pattern discovery and learning distillation.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">P</span>
<p class="rule-card__title">InsightPattern</p>
<p class="rule-card__desc">A discovered cluster enriched with cross-metric metadata — which metrics are involved, test case coverage, and confidence.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">R</span>
<p class="rule-card__title">InsightResult</p>
<p class="rule-card__desc">Complete result containing patterns, learnings, and the full pipeline result for advanced access.</p>
</div>
</div>

---

## InsightExtractor

::: axion.reporting.insight_extractor.InsightExtractor
    options:
      show_source: false
      members:
        - __init__
        - analyze
        - analyze_sync

---

## Data Classes

### InsightPattern

::: axion.reporting.insight_extractor.InsightPattern
    options:
      show_source: false

### InsightResult

::: axion.reporting.insight_extractor.InsightResult
    options:
      show_source: false

---

## Adapter Function

### _issue_to_evidence

::: axion.reporting.insight_extractor._issue_to_evidence
    options:
      show_source: false

---

<div class="ref-nav" markdown>

[Insight Extraction Guide :octicons-arrow-right-24:](../guides/insight-extraction.md){ .md-button .md-button--primary }
[Issue Extractor Reference :octicons-arrow-right-24:](issue-extractor.md){ .md-button }
[Pattern Discovery :octicons-arrow-right-24:](../deep-dives/caliber/pattern-discovery.md){ .md-button }

</div>
