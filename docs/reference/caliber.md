---
icon: custom/calibration
---
# Caliber API Reference

LLM-as-judge calibration and alignment tools for improving evaluation quality.

<div class="ref-import" markdown>

```python
from axion.caliber import (
    CaliberMetric,
    ExampleSelector, SelectionStrategy, SelectionResult,
    PatternDiscovery, DiscoveredPattern, PatternDiscoveryResult,
    EvidenceItem, EvidencePipeline, LearningArtifact, PipelineResult,
    Provenance, MetadataConfig, ClusteringMethod, AnnotatedItem,
    InMemorySink, JsonlSink, InMemoryDeduper, EmbeddingDeduper,
    MisalignmentAnalyzer, MisalignmentAnalysis, MisalignmentPattern,
    PromptOptimizer, OptimizedPrompt, PromptSuggestion,
    CaliberRenderer, NotebookCaliberRenderer,
    ConsoleCaliberRenderer, JsonCaliberRenderer,
)
```

</div>

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">C</span>
<p class="rule-card__title">CaliberMetric</p>
<p class="rule-card__desc">Core metric for measuring LLM judge alignment against human ground truth scores.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">P</span>
<p class="rule-card__title">Pattern Discovery</p>
<p class="rule-card__desc">Cluster any text evidence into themes and distill actionable learning artifacts via a full pipeline.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">M</span>
<p class="rule-card__title">Misalignment Analysis</p>
<p class="rule-card__desc">Identify systematic disagreements between human and LLM judges and surface root causes.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">O</span>
<p class="rule-card__title">Prompt Optimization</p>
<p class="rule-card__desc">Automatically generate improved evaluation prompts that better align with human judgment.</p>
</div>
</div>

---

## CaliberMetric

::: axion.caliber.CaliberMetric
    options:
      show_root_heading: true

---

## Example Selection

### ExampleSelector

::: axion.caliber.ExampleSelector
    options:
      show_root_heading: true

### SelectionStrategy

::: axion.caliber.SelectionStrategy
    options:
      show_root_heading: true

### SelectionResult

::: axion.caliber.SelectionResult
    options:
      show_root_heading: true

---

## Pattern Discovery

### PatternDiscovery

::: axion.caliber.PatternDiscovery
    options:
      show_root_heading: true

### EvidenceItem

::: axion.caliber.EvidenceItem
    options:
      show_root_heading: true

### DiscoveredPattern

::: axion.caliber.DiscoveredPattern
    options:
      show_root_heading: true

### PatternDiscoveryResult

::: axion.caliber.PatternDiscoveryResult
    options:
      show_root_heading: true

### ClusteringMethod

::: axion.caliber.ClusteringMethod
    options:
      show_root_heading: true

### EvidencePipeline

::: axion.caliber.EvidencePipeline
    options:
      show_root_heading: true

### LearningArtifact

::: axion.caliber.LearningArtifact
    options:
      show_root_heading: true

### PipelineResult

::: axion.caliber.PipelineResult
    options:
      show_root_heading: true

### Provenance

::: axion.caliber.Provenance
    options:
      show_root_heading: true

### MetadataConfig

::: axion.caliber.MetadataConfig
    options:
      show_root_heading: true

### Sinks & Dedupers

#### InMemorySink

::: axion.caliber.InMemorySink
    options:
      show_root_heading: true

#### JsonlSink

::: axion.caliber.JsonlSink
    options:
      show_root_heading: true

#### InMemoryDeduper

::: axion.caliber.InMemoryDeduper
    options:
      show_root_heading: true

#### EmbeddingDeduper

::: axion.caliber.EmbeddingDeduper
    options:
      show_root_heading: true

### AnnotatedItem (Legacy)

::: axion.caliber.AnnotatedItem
    options:
      show_root_heading: true

---

## Misalignment Analysis

### MisalignmentAnalyzer

::: axion.caliber.MisalignmentAnalyzer
    options:
      show_root_heading: true

### MisalignmentAnalysis

::: axion.caliber.MisalignmentAnalysis
    options:
      show_root_heading: true

### MisalignmentPattern

::: axion.caliber.MisalignmentPattern
    options:
      show_root_heading: true

---

## Prompt Optimization

### PromptOptimizer

::: axion.caliber.PromptOptimizer
    options:
      show_root_heading: true

### OptimizedPrompt

::: axion.caliber.OptimizedPrompt
    options:
      show_root_heading: true

### PromptSuggestion

::: axion.caliber.PromptSuggestion
    options:
      show_root_heading: true

---

## Renderers

### CaliberRenderer

::: axion.caliber.CaliberRenderer
    options:
      show_root_heading: true

### NotebookCaliberRenderer

::: axion.caliber.NotebookCaliberRenderer
    options:
      show_root_heading: true

### ConsoleCaliberRenderer

::: axion.caliber.ConsoleCaliberRenderer
    options:
      show_root_heading: true

### JsonCaliberRenderer

::: axion.caliber.JsonCaliberRenderer
    options:
      show_root_heading: true

---

<div class="ref-nav" markdown>

[CaliberHQ Guide :octicons-arrow-right-24:](../guides/caliberhq.md){ .md-button .md-button--primary }
[Pattern Discovery Deep Dive :octicons-arrow-right-24:](../deep-dives/caliber/pattern-discovery.md){ .md-button }
[Example Selector Deep Dive :octicons-arrow-right-24:](../deep-dives/caliber/example-selector.md){ .md-button }

</div>
