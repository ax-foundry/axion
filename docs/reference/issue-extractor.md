---
icon: custom/search
---
# Issue Extractor Reference

API reference for the issue extraction and signal analysis module.

<div class="ref-import" markdown>

```python
from axion.reporting import (
    IssueExtractor,
    ExtractedIssue,
    IssueExtractionResult,
    IssueGroup,
    LLMSummaryInput,
    MetricSignalAdapter,
    SignalAdapterRegistry,
)
```

</div>

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">E</span>
<p class="rule-card__title">IssueExtractor</p>
<p class="rule-card__desc">Core engine for extracting low-score signals from evaluation results into structured, actionable issues.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">R</span>
<p class="rule-card__title">SignalAdapterRegistry</p>
<p class="rule-card__desc">Registry for metric signal adapters. Maps metric keys to extraction rules for pass/fail signals.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">A</span>
<p class="rule-card__title">MetricSignalAdapter</p>
<p class="rule-card__desc">Adapter defining how to extract issues from a specific metric's signals — headline signals, failure values, and context.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">D</span>
<p class="rule-card__title">Data Classes</p>
<p class="rule-card__desc">Structured types for extracted issues, groups, summaries, and LLM input — the output of the extraction pipeline.</p>
</div>
</div>

---

## IssueExtractor

::: axion.reporting.issue_extractor.IssueExtractor
    options:
      show_source: false
      members:
        - __init__
        - extract_from_evaluation
        - extract_from_test_result
        - extract_from_metric_score
        - to_llm_input
        - to_prompt_text
        - to_grouped_prompt_text
        - to_grouped_prompt_text_async
        - summarize
        - summarize_sync

---

## SignalAdapterRegistry

::: axion.reporting.issue_extractor.SignalAdapterRegistry
    options:
      show_source: false
      members:
        - register
        - register_adapter
        - get
        - list_adapters

---

## Data Classes

### ExtractedIssue

::: axion.reporting.issue_extractor.ExtractedIssue
    options:
      show_source: false

### IssueExtractionResult

::: axion.reporting.issue_extractor.IssueExtractionResult
    options:
      show_source: false

### IssueGroup

::: axion.reporting.issue_extractor.IssueGroup
    options:
      show_source: false

### IssueSummary

::: axion.reporting.issue_extractor.IssueSummary
    options:
      show_source: false

### LLMSummaryInput

::: axion.reporting.issue_extractor.LLMSummaryInput
    options:
      show_source: false

### MetricSignalAdapter

::: axion.reporting.issue_extractor.MetricSignalAdapter
    options:
      show_source: false

---

## Built-in Adapters

The following adapters are pre-registered:

| Adapter Key | Headline Signals | Issue Values |
|-------------|------------------|--------------|
| `faithfulness` | `faithfulness_verdict` | `CONTRADICTORY`, `NO_EVIDENCE` |
| `answer_criteria` | `is_covered`, `concept_coverage` | `False` |
| `answer_relevancy` | `is_relevant`, `verdict` | `False`, `no` |
| `answer_completeness` | `is_covered`, `is_addressed` | `False` |
| `factual_accuracy` | `is_correct`, `accuracy_score` | `False`, `0` |
| `answer_conciseness` | `conciseness_score` | (score-based) |
| `contextual_relevancy` | `is_relevant` | `False` |
| `contextual_recall` | `is_attributable`, `is_supported` | `False` |
| `contextual_precision` | `is_useful`, `map_score` | `False` |
| `contextual_utilization` | `is_utilized` | `False` |
| `contextual_sufficiency` | `is_sufficient` | `False` |
| `contextual_ranking` | `is_correctly_ranked` | `False` |
| `citation_relevancy` | `relevance_verdict` | `False` |
| `pii_leakage` | `pii_verdict` | `yes` |
| `tone_style_consistency` | `is_consistent` | `False` |
| `persona_tone_adherence` | `persona_match` | `False` |
| `conversation_efficiency` | `efficiency_score` | (score-based) |
| `conversation_flow` | `final_score` | (score-based) |
| `goal_completion` | `is_completed`, `goal_achieved` | `False` |
| `citation_presence` | `presence_check_passed` | `False` |
| `latency` | `latency_score` | (threshold-based) |
| `tool_correctness` | `all_tools_correct` | `False` |

---

<div class="ref-nav" markdown>

[Issue Extraction Guide :octicons-arrow-right-24:](../guides/issue-extraction.md){ .md-button .md-button--primary }
[Insight Extractor Reference :octicons-arrow-right-24:](insight-extractor.md){ .md-button }
[Running Evaluations :octicons-arrow-right-24:](../guides/evaluation.md){ .md-button }

</div>
