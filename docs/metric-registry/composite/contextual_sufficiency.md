# Contextual Sufficiency

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluate if retrieved context contains enough information to answer the query</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Knowledge</span>
<span class="badge" style="background: #0F2440;">Single Turn</span>
<span class="badge" style="background: #06b6d4;">Retrieval</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> or <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Binary sufficiency verdict</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.5</code><br>
<small style="color: var(--md-text-muted);">Pass/fail cutoff</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>query</code> <code>retrieved_content</code><br>
<small style="color: var(--md-text-muted);">No answer required</small>
</div>

</div>

!!! abstract "What It Measures"
    Contextual Sufficiency evaluates whether the **retrieved context contains enough information** to fully answer the user's query. Unlike other metrics that measure partial coverage, this is a binary judgment: either the context is sufficient or it isn't.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Context is sufficient to answer the query |
    | **0.0** | :material-close: Context is insufficient—information missing |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Diagnosing retrieval quality</li>
<li>Testing retrieval before generation</li>
<li>Identifying information gaps</li>
<li>Deciding when to expand search</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Need granular coverage scores</li>
<li>Evaluating answer quality</li>
<li>Comparing retrieval strategies</li>
<li>Need partial credit</li>
</ul>
</div>

</div>

!!! tip "RAG Evaluation Suite"
    **Contextual Sufficiency** asks: *"Is there enough context to answer this question?"*

    Related retrieval metrics:

    - **[Contextual Relevancy](./contextual_relevancy.md)**: Are chunks relevant?
    - **[Contextual Recall](./contextual_recall.md)**: Are expected facts present?
    - **[Contextual Utilization](./contextual_utilization.md)**: Was the context actually used?

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric uses an LLM to make a binary judgment about context sufficiency.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[Retrieved Context]
        end

        subgraph JUDGE["⚖️ Sufficiency Judgment"]
            C[RAGAnalyzer Engine]
            D["Can this context answer the query?"]
            E["Binary Verdict"]
        end

        subgraph OUTPUT["📊 Result"]
            F["1.0 = Sufficient"]
            G["0.0 = Insufficient"]
            H["Reasoning Provided"]
        end

        A & B --> C
        C --> D
        D --> E
        E --> F & G
        F & G --> H

        style INPUT stroke:#1E3A5F,stroke-width:2px
        style JUDGE stroke:#f59e0b,stroke-width:2px
        style OUTPUT stroke:#10b981,stroke-width:2px
        style E fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Verdict System"

    A single binary verdict for the entire context.

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ SUFFICIENT</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1.0</div>
    <br><small>Context contains <strong>all necessary information</strong> to answer the query completely.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ INSUFFICIENT</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0.0</div>
    <br><small>Context is <strong>missing critical information</strong> needed to answer the query.</small>
    </div>

    </div>

    !!! info "Diagnostic Purpose"
        This metric helps diagnose retrieval issues independent of generation. If sufficiency is low but faithfulness is high, your retriever needs improvement.

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level |

    !!! info "Binary by Design"
        Unlike other metrics that provide granular scores, Sufficiency is intentionally binary. For partial coverage scores, use Contextual Recall or Contextual Relevancy.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ContextualSufficiency
    from axion.dataset import DatasetItem

    metric = ContextualSufficiency()

    item = DatasetItem(
        query="What is the boiling point of water?",
        retrieved_content=[
            "Water boils at 100 degrees Celsius at sea level.",
            "This is equivalent to 212 degrees Fahrenheit.",
        ],
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 1.0 (context is sufficient)
    ```

=== ":material-close: Insufficient Example"

    ```python
    from axion.metrics import ContextualSufficiency

    metric = ContextualSufficiency()

    item = DatasetItem(
        query="What is the boiling point of water at high altitude?",
        retrieved_content=[
            "Water boils at 100 degrees Celsius at sea level.",
        ],
    )

    result = await metric.execute(item)
    # Score: 0.0 (missing altitude information)
    print(result.signals.reasoning)
    # "Context only mentions sea level; no information about altitude effects."
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ContextualSufficiency
    from axion.runners import MetricRunner

    metric = ContextualSufficiency()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    sufficient_count = sum(1 for r in results if r.score == 1.0)
    print(f"Sufficient: {sufficient_count}/{len(results)}")

    for item_result in results:
        if item_result.score == 0.0:
            print(f"⚠️ Insufficient for: {item_result.signals.query[:50]}...")
            print(f"   Reason: {item_result.signals.reasoning}")
    ```

---

## Metric Diagnostics

Every evaluation is **fully interpretable**. Access detailed diagnostic results via `result.signals` to understand exactly why a score was given—no black boxes.

```python
result = await metric.execute(item)
print(result.pretty())      # Human-readable summary
result.signals              # Full diagnostic breakdown
```

<details markdown="1">
<summary><strong>📊 ContextualSufficiencyResult Structure</strong></summary>

```python
ContextualSufficiencyResult(
{
    "sufficiency_score": 1.0,
    "is_sufficient": true,
    "reasoning": "The context fully addresses the query by providing the boiling point of water (100°C) and its Fahrenheit equivalent (212°F).",
    "query": "What is the boiling point of water?",
    "context": "Water boils at 100 degrees Celsius at sea level. This is equivalent to 212 degrees Fahrenheit."
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `sufficiency_score` | `float` | Binary score (1.0 or 0.0) |
| `is_sufficient` | `bool` | Whether context is sufficient |
| `reasoning` | `str` | Explanation for the verdict |
| `query` | `str` | The user query (preview) |
| `context` | `str` | The retrieved context (preview) |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Sufficient Context (Score: 1.0)</strong></summary>

!!! success "Complete Information"

    **Query:**
    > "Who invented the telephone and when?"

    **Retrieved Context:**

    > "Alexander Graham Bell invented the telephone in 1876. He was granted the patent on March 7th of that year."

    **Analysis:**

    - ✅ Inventor identified: Alexander Graham Bell
    - ✅ Year provided: 1876
    - ✅ Additional detail: Patent date

    **Verdict:** Sufficient

    **Reasoning:** "The context directly answers both parts of the query—who (Alexander Graham Bell) and when (1876)."

    **Final Score:** `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 2: Insufficient - Missing Key Info (Score: 0.0)</strong></summary>

!!! failure "Critical Information Missing"

    **Query:**
    > "What are the side effects of aspirin?"

    **Retrieved Context:**

    > "Aspirin is a common pain reliever. It belongs to a class of drugs called NSAIDs. It can be purchased over the counter."

    **Analysis:**

    - ✅ Drug identification: Correct
    - ✅ Drug class: NSAIDs
    - ❌ Side effects: Not mentioned

    **Verdict:** Insufficient

    **Reasoning:** "The context describes what aspirin is but does not mention any side effects, which is the core of the query."

    **Final Score:** `0.0` :material-close:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Insufficient - Partial Answer (Score: 0.0)</strong></summary>

!!! failure "Incomplete Coverage"

    **Query:**
    > "Compare the populations of Tokyo and New York City."

    **Retrieved Context:**

    > "Tokyo is the capital of Japan with a metropolitan population of over 37 million people, making it the world's most populous metropolitan area."

    **Analysis:**

    - ✅ Tokyo population: Provided
    - ❌ NYC population: Missing
    - ❌ Comparison: Cannot be made

    **Verdict:** Insufficient

    **Reasoning:** "Context only provides Tokyo's population. NYC population is missing, making a comparison impossible."

    **Final Score:** `0.0` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Retrieval Diagnosis</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Quickly identify if poor answers stem from insufficient retrieval, not generation quality.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔄</span>
<strong>Adaptive Search</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Use as a signal to expand search or trigger alternative retrieval strategies.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">⚡</span>
<strong>Pre-Generation Check</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Evaluate context before generating—don't waste tokens on insufficient information.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Contextual Sufficiency** = Is there enough context to fully answer the query?

    - **Use it when:** Diagnosing retrieval gaps or deciding to expand search
    - **Score interpretation:** 1.0 = sufficient, 0.0 = insufficient (binary)
    - **Key insight:** Identifies "missing information" problems in retrieval

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ContextualSufficiency`](../../reference/metrics.md#contextual-sufficiency)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Contextual Recall](./contextual_recall.md) · Contextual Relevancy · Contextual Utilization

</div>
