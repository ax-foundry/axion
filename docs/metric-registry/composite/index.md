---
icon: custom/overview
---
# Composite Metrics

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">LLM-powered evaluation metrics for comprehensive AI response analysis</strong><br>
<span class="badge" style="margin-top: 0.5rem;">14 Metrics</span>
<span class="badge" style="background: #667eea;">LLM-Powered</span>
</div>

Composite metrics use language models to perform nuanced reasoning and analysis. These metrics evaluate complex aspects of AI responses including factual accuracy, relevance, grounding, and style—things that require understanding context, semantics, and intent.

---

## RAG & Retrieval Metrics

Evaluate the quality of retrieval-augmented generation systems.

<div class="grid-container">

<div class="grid-item">
<strong><a href="faithfulness/">Faithfulness</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Verify claims against retrieved context</p>
<code>query</code> <code>actual_output</code> <code>retrieved_content</code>
</div>

<div class="grid-item">
<strong><a href="contextual_relevancy/">Contextual Relevancy</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Check if retrieved chunks are relevant</p>
<code>query</code> <code>retrieved_content</code>
</div>

<div class="grid-item">
<strong><a href="contextual_recall/">Contextual Recall</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Check if context supports expected answer</p>
<code>expected_output</code> <code>retrieved_content</code>
</div>

<div class="grid-item">
<strong><a href="contextual_precision/">Contextual Precision</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Measure useful chunk ranking (MAP)</p>
<code>query</code> <code>expected_output</code> <code>retrieved_content</code>
</div>

<div class="grid-item">
<strong><a href="contextual_ranking/">Contextual Ranking</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Check if relevant chunks rank higher</p>
<code>query</code> <code>retrieved_content</code>
</div>

<div class="grid-item">
<strong><a href="contextual_sufficiency/">Contextual Sufficiency</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Binary check for enough context</p>
<code>query</code> <code>retrieved_content</code>
</div>

<div class="grid-item">
<strong><a href="contextual_utilization/">Contextual Utilization</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Measure context usage efficiency</p>
<code>query</code> <code>actual_output</code> <code>retrieved_content</code>
</div>

</div>

---

## Answer Quality Metrics

Evaluate the quality and correctness of AI-generated answers.

<div class="grid-container">

<div class="grid-item">
<strong><a href="answer_relevancy/">Answer Relevancy</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Check if response addresses the query</p>
<code>query</code> <code>actual_output</code>
</div>

<div class="grid-item">
<strong><a href="factual_accuracy/">Factual Accuracy</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Verify against ground truth</p>
<code>query</code> <code>actual_output</code> <code>expected_output</code>
</div>

<div class="grid-item">
<strong><a href="answer_completeness/">Answer Completeness</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Check coverage of expected content</p>
<code>query</code> <code>actual_output</code> <code>expected_output</code>
</div>

<div class="grid-item">
<strong><a href="answer_criteria/">Answer Criteria</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Evaluate against custom criteria</p>
<code>query</code> <code>actual_output</code> + <code>acceptance_criteria</code>
</div>

</div>

---

## Style & Safety Metrics

Evaluate tone, citations, and privacy compliance.

<div class="grid-container">

<div class="grid-item">
<strong><a href="tone_style_consistency/">Tone & Style Consistency</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Match expected voice and formatting</p>
<code>actual_output</code> <code>expected_output</code>
</div>

<div class="grid-item">
<strong><a href="citation_relevancy/">Citation Relevancy</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Validate citation quality</p>
<code>query</code> <code>actual_output</code>
</div>

<div class="grid-item">
<strong><a href="pii_leakage/">PII Leakage</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Detect privacy violations</p>
<code>query</code> <code>actual_output</code>
</div>

</div>

---

## Quick Reference

| Metric | Score Range | Threshold | Key Question |
|--------|-------------|-----------|--------------|
| **Faithfulness** | 0.0 – 1.0 | 0.5 | Are claims grounded in context? |
| **Answer Relevancy** | 0.0 – 1.0 | 0.5 | Does response address the query? |
| **Factual Accuracy** | 0.0 – 1.0 | 0.8 | Does it match ground truth? |
| **Answer Completeness** | 0.0 – 1.0 | 0.5 | Are all expected aspects covered? |
| **Answer Criteria** | 0.0 – 1.0 | 0.5 | Does it meet custom criteria? |
| **Tone & Style** | 0.0 – 1.0 | 0.8 | Does it match expected voice? |
| **Citation Relevancy** | 0.0 – 1.0 | 0.8 | Are citations relevant? |
| **PII Leakage** | 0.0 – 1.0 | 0.5 | Is output privacy-safe? (1.0 = safe) |
| **Contextual Relevancy** | 0.0 – 1.0 | 0.5 | Are chunks relevant to query? |
| **Contextual Recall** | 0.0 – 1.0 | 0.5 | Is expected answer in context? |
| **Contextual Precision** | 0.0 – 1.0 | 0.5 | Are useful chunks ranked first? |
| **Contextual Ranking** | 0.0 – 1.0 | 0.5 | Are relevant chunks ranked first? |
| **Contextual Sufficiency** | 0.0 or 1.0 | 0.5 | Is context sufficient? (binary) |
| **Contextual Utilization** | 0.0 – 1.0 | 0.5 | Was relevant context used? |

---

## Usage Example

```python
from axion.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextualPrecision,
)
from axion.runners import MetricRunner
from axion.dataset import Dataset

# Initialize metrics
metrics = [
    Faithfulness(strict_mode=True),
    AnswerRelevancy(),
    ContextualPrecision(),
]

# Run evaluation
runner = MetricRunner(metrics=metrics)
results = await runner.run(dataset)

# Analyze results
for item in results:
    print(f"Faithfulness: {item.scores['faithfulness']:.2f}")
    print(f"Relevancy: {item.scores['answer_relevancy']:.2f}")
    print(f"Precision: {item.scores['contextual_precision']:.2f}")
```

---

## Choosing the Right Metrics

!!! tip "Evaluation Strategy"

    **For RAG Systems:**

    - Start with **Faithfulness** (hallucination detection)
    - Add **Contextual Relevancy** (retrieval quality)
    - Use **Contextual Precision/Ranking** (ranking quality)

    **For Q&A Systems:**

    - Use **Answer Relevancy** (topical alignment)
    - Add **Factual Accuracy** if you have ground truth
    - Add **Answer Completeness** for comprehensive responses

    **For Customer Service:**

    - Use **Tone & Style Consistency** (brand voice)
    - Add **Answer Criteria** (policy compliance)
    - Include **PII Leakage** (privacy protection)

    **For Research Assistants:**

    - Use **Citation Relevancy** (source quality)
    - Add **Faithfulness** (grounding)
    - Include **Answer Completeness** (thoroughness)
