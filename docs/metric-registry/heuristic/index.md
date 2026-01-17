# Heuristic Metrics

<div style="border-left: 4px solid #f59e0b; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Fast, deterministic evaluation metrics using rule-based and statistical methods</strong><br>
<span class="badge" style="margin-top: 0.5rem; background: #f59e0b;">12 Metrics</span>
<span class="badge" style="background: #06b6d4;">No LLM Required</span>
</div>

Heuristic metrics are rule-based evaluation metrics that don't require LLM calls. They're ideal for production environments where speed, cost efficiency, and deterministic results are critical. These metrics use pattern matching, statistical analysis, and algorithmic comparisons.

---

## String Matching Metrics

Compare actual outputs against expected outputs using various matching strategies.

<div class="grid-container">

<div class="grid-item">
<strong><a href="exact_string_match/">Exact String Match</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Check for identical strings</p>
<code>actual_output</code> <code>expected_output</code>
</div>

<div class="grid-item">
<strong><a href="contains_match/">Contains Match</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Check if output contains expected text</p>
<code>actual_output</code> <code>expected_output</code>
</div>

<div class="grid-item">
<strong><a href="levenshtein_ratio/">Levenshtein Ratio</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Character-level string similarity</p>
<code>actual_output</code> <code>expected_output</code>
</div>

<div class="grid-item">
<strong><a href="sentence_bleu/">Sentence BLEU</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">N-gram precision similarity</p>
<code>actual_output</code> <code>expected_output</code>
</div>

</div>

---

## Safety & Compliance Metrics

Evaluate outputs for privacy, citations, and policy compliance.

<div class="grid-container">

<div class="grid-item">
<strong><a href="pii_leakage_heuristic/">PII Leakage (Heuristic)</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Detect PII using regex patterns</p>
<code>query</code> <code>actual_output</code>
</div>

<div class="grid-item">
<strong><a href="citation_presence/">Citation Presence</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Verify responses include citations</p>
<code>actual_output</code>
</div>

</div>

---

## Performance Metrics

Monitor execution time and operational performance.

<div class="grid-container">

<div class="grid-item">
<strong><a href="latency/">Latency</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Measure and evaluate execution time</p>
<code>latency</code>
</div>

</div>

---

## Retrieval Metrics (IR)

Standard information retrieval metrics for evaluating search and ranking quality.

<div class="grid-container">

<div class="grid-item">
<strong><a href="retrieval_metrics/">Hit Rate @ K</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Any relevant result in top K?</p>
<code>actual_ranking</code> <code>expected_reference</code>
</div>

<div class="grid-item">
<strong><a href="retrieval_metrics/">MRR</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Rank of first relevant result</p>
<code>actual_ranking</code> <code>expected_reference</code>
</div>

<div class="grid-item">
<strong><a href="retrieval_metrics/">NDCG @ K</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Graded relevance with discounting</p>
<code>actual_ranking</code> <code>expected_reference</code>
</div>

<div class="grid-item">
<strong><a href="retrieval_metrics/">Precision @ K</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Fraction of top K that's relevant</p>
<code>actual_ranking</code> <code>expected_reference</code>
</div>

<div class="grid-item">
<strong><a href="retrieval_metrics/">Recall @ K</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Coverage of relevant documents</p>
<code>actual_ranking</code> <code>expected_reference</code>
</div>

</div>

---

## Quick Reference

| Metric | Score Range | Threshold | Key Question |
|--------|-------------|-----------|--------------|
| **Exact String Match** | 0.0 or 1.0 | 0.5 | Are strings identical? |
| **Contains Match** | 0.0 or 1.0 | 0.5 | Is expected text in output? |
| **Levenshtein Ratio** | 0.0 – 1.0 | 0.2 | How similar are the strings? |
| **Sentence BLEU** | 0.0 – 1.0 | 0.5 | How much n-gram overlap? |
| **PII Leakage (Heuristic)** | 0.0 – 1.0 | 0.8 | Is output privacy-safe? (1.0 = safe) |
| **Citation Presence** | 0.0 or 1.0 | 0.5 | Are citations included? |
| **Latency** | 0.0 – ∞ | 5.0s | How fast was the response? |
| **Hit Rate @ K** | 0.0 or 1.0 | - | Any relevant in top K? |
| **MRR** | 0.0 – 1.0 | - | How early is first relevant? |
| **NDCG @ K** | 0.0 – 1.0 | - | Is ranking optimal? |
| **Precision @ K** | 0.0 – 1.0 | - | Are results mostly relevant? |
| **Recall @ K** | 0.0 – 1.0 | - | Did we find all relevant? |

---

## Usage Example

```python
from axion.metrics import (
    ExactStringMatch,
    LevenshteinRatio,
    PIILeakageHeuristic,
    HitRateAtK,
)
from axion.runners import MetricRunner
from axion.dataset import Dataset

# Initialize metrics
metrics = [
    ExactStringMatch(),
    LevenshteinRatio(case_sensitive=False),
    PIILeakageHeuristic(confidence_threshold=0.7),
    HitRateAtK(k=10),
]

# Run evaluation
runner = MetricRunner(metrics=metrics)
results = await runner.run(dataset)

# Analyze results
for item in results:
    print(f"Exact Match: {item.scores.get('exact_string_match', 'N/A')}")
    print(f"Similarity: {item.scores.get('levenshtein_ratio', 'N/A'):.2f}")
    print(f"Privacy Safe: {item.scores.get('pii_leakage_heuristic', 'N/A'):.2f}")
```

---

## Choosing the Right Metrics

!!! tip "Evaluation Strategy"

    **For Exact Outputs (Code, JSON, IDs):**

    - Use **Exact String Match** for strict equality
    - Add **Contains Match** for partial verification

    **For Natural Language:**

    - Use **Levenshtein Ratio** for typo/variation tolerance
    - Use **Sentence BLEU** for paraphrase comparison

    **For Privacy & Compliance:**

    - Use **PII Leakage (Heuristic)** for fast screening
    - Add **Citation Presence** for source attribution

    **For Search/Retrieval:**

    - Use **Hit Rate** for quick sanity checks
    - Use **NDCG** for comprehensive ranking evaluation
    - Use **Precision/Recall** for classic IR metrics

---

## Why Heuristic Metrics?

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">⚡</span>
<strong>Instant Results</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">No LLM calls needed—microsecond latency.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">💰</span>
<strong>Zero Cost</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">No API costs or token usage.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔄</span>
<strong>Deterministic</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Same input always produces same output.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📈</span>
<strong>Scalable</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Evaluate millions of items without limits.</p>
</div>

</div>
