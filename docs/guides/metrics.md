# Metrics & Evaluation

Axion provides 30+ metrics for evaluating AI agents across multiple dimensions.

## Quick Start

```python
from axion import Dataset, metric_registry
from axion.metrics import Faithfulness, AnswerRelevancy

# Load your dataset
dataset = Dataset.from_csv("eval_data.csv")

# Select metrics
metrics = [Faithfulness(), AnswerRelevancy()]

# Run evaluation
from axion.runners import evaluation_runner
results = await evaluation_runner(dataset, metrics)
```

## Metric Output Types

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">S</span>
<p class="rule-card__title">Score</p>
<p class="rule-card__desc">Numeric value (0&ndash;1) with pass/fail threshold. Example: <code>Faithfulness</code> &rarr; <code>0.85</code></p>
</div>
<div class="rule-card">
<span class="rule-card__number">C</span>
<p class="rule-card__title">Classification</p>
<p class="rule-card__desc">Single label from a fixed set. Example: <code>SentimentClassification</code> &rarr; <code>"positive"</code></p>
</div>
<div class="rule-card">
<span class="rule-card__number">A</span>
<p class="rule-card__title">Analysis</p>
<p class="rule-card__desc">Structured insights without scoring. Example: <code>ReferralReasonAnalysis</code> &rarr; <code>{reasons[], citations[]}</code></p>
</div>
</div>

See [Creating Custom Metrics](../deep-dives/metrics/creating-metrics.md#metric-categories) for details on choosing the right output type.

## Metric Categories

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Composite (LLM-based)</p>
<p class="rule-card__desc">Nuanced evaluation requiring reasoning. Faithfulness, AnswerRelevancy, FactualAccuracy, AnswerCompleteness, AnswerCriteria.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">Heuristic (Non-LLM)</p>
<p class="rule-card__desc">Fast, deterministic checks. ExactStringMatch, CitationPresence, Latency, ContainsMatch.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">Retrieval</p>
<p class="rule-card__desc">RAG pipeline evaluation. HitRateAtK, MeanReciprocalRank, ContextualRelevancy, ContextualSufficiency.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">4</span>
<p class="rule-card__title">Conversational</p>
<p class="rule-card__desc">Multi-turn agent evaluation. GoalCompletion, ConversationEfficiency, ConversationFlow.</p>
</div>
</div>

### Composite Metrics (LLM-based)

| Metric | What it Measures |
|--------|-----------------|
| `Faithfulness` | Is the answer grounded in retrieved context? |
| `AnswerRelevancy` | Does the answer address the question? |
| `FactualAccuracy` | Is the answer factually correct vs. expected? |
| `AnswerCompleteness` | Are all parts of the question answered? |
| `AnswerCriteria` | Does the answer meet specific business rules? |

### Heuristic Metrics (Non-LLM)

| Metric | What it Measures |
|--------|-----------------|
| `ExactStringMatch` | Exact match between actual and expected |
| `CitationPresence` | Are citations/references present? |
| `Latency` | Response time (pass/fail threshold) |
| `ContainsMatch` | Does output contain required phrases? |

### Retrieval Metrics

| Metric | What it Measures |
|--------|-----------------|
| `HitRateAtK` | Is the right doc in top K results? |
| `MeanReciprocalRank` | Position of first relevant result |
| `ContextualRelevancy` | Are retrieved chunks relevant? |
| `ContextualSufficiency` | Do chunks contain the answer? |

### Conversational Metrics

| Metric | What it Measures |
|--------|-----------------|
| `GoalCompletion` | Did user achieve their goal? |
| `ConversationEfficiency` | Were there unnecessary loops? |
| `ConversationFlow` | Is the dialogue logical? |

## Using the Metric Registry

```python
from axion.metrics import metric_registry

# List all available metrics
print(metric_registry.list_metrics())

# Get metric by name
metric = metric_registry.get("Faithfulness")

# Filter by category
composite_metrics = metric_registry.filter(category="composite")
```

## Customizing Metrics

```python
from axion.metrics import Faithfulness

# Adjust threshold
metric = Faithfulness(threshold=0.8)

# Custom instructions
metric = AnswerCriteria(
    criteria_key="my_criteria",
    scoring_strategy="aspect"
)
```

---

[Running Evaluations :octicons-arrow-right-24:](evaluation.md){ .md-button .md-button--primary }
[Creating Custom Metrics :octicons-arrow-right-24:](../deep-dives/metrics/creating-metrics.md){ .md-button }
[Metrics Reference :octicons-arrow-right-24:](../reference/metrics.md){ .md-button }
