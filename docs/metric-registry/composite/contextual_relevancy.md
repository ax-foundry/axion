# Contextual Relevancy

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluate if retrieved context is relevant to the user's query</strong><br>
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
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Ratio of relevant chunks</small>
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
    Contextual Relevancy evaluates whether the **retrieved context chunks** are relevant to the user's query. It measures retrieval quality independent of generation—answering: "Did we retrieve the right documents?"

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All retrieved chunks are relevant |
    | **0.7+** | :material-check: Most chunks relevant, some noise |
    | **0.5** | :material-alert: Mixed relevance—half helpful |
    | **< 0.5** | :material-close: Mostly irrelevant retrieval |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Evaluating RAG retrieval quality</li>
<li>Tuning vector search parameters</li>
<li>Debugging poor answer quality</li>
<li>Comparing retrieval strategies</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>No retrieval component exists</li>
<li>Evaluating answer quality (use Faithfulness)</li>
<li>All chunks are from same document</li>
<li>Retrieval is keyword-based only</li>
</ul>
</div>

</div>

!!! tip "RAG Evaluation Suite"
    **Contextual Relevancy** asks: *"Are the retrieved chunks relevant to the query?"*

    Related retrieval metrics:

    - **[Contextual Precision](./contextual_precision.md)**: Are relevant chunks ranked higher?
    - **[Contextual Recall](./contextual_recall.md)**: Do chunks cover the expected answer?
    - **[Contextual Sufficiency](./contextual_sufficiency.md)**: Is there enough info to answer?

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric evaluates each retrieved chunk's relevance to the query.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[Retrieved Chunks]
        end

        subgraph EVALUATE["⚖️ Step 1: Relevancy Check"]
            C[RAGAnalyzer Engine]
            D1["Chunk 1: ✓/✗"]
            D2["Chunk 2: ✓/✗"]
            D3["Chunk 3: ✓/✗"]
            DN["..."]
        end

        subgraph SCORE["📊 Step 2: Scoring"]
            E["Count Relevant Chunks"]
            F["Calculate Ratio"]
            G["Final Score"]
        end

        A & B --> C
        C --> D1 & D2 & D3 & DN
        D1 & D2 & D3 & DN --> E
        E --> F
        F --> G

        style INPUT stroke:#1E3A5F,stroke-width:2px
        style EVALUATE stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style G fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Verdict System"

    Each chunk receives a binary relevance verdict.

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ RELEVANT</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1</div>
    <br><small>Chunk contains information useful for answering the query.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ IRRELEVANT</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0</div>
    <br><small>Chunk is off-topic or doesn't help answer the query.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = relevant_chunks / total_chunks
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level |

    !!! info "Shared Cache"
        Contextual Relevancy shares an internal cache with other contextual metrics. Running multiple retrieval metrics together is efficient.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ContextualRelevancy
    from axion.dataset import DatasetItem

    metric = ContextualRelevancy()

    item = DatasetItem(
        query="What is the capital of France?",
        retrieved_content=[
            "Paris is the capital and largest city of France.",
            "France is known for its wine and cuisine.",
            "The Eiffel Tower was built in 1889.",
        ],
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 0.67 (2 of 3 chunks relevant)
    ```

=== ":material-compare: Comparing Retrieval"

    ```python
    from axion.metrics import ContextualRelevancy
    from axion.runners import MetricRunner

    metric = ContextualRelevancy()
    runner = MetricRunner(metrics=[metric])

    # Compare two retrieval strategies
    results_v1 = await runner.run(dataset_with_bm25)
    results_v2 = await runner.run(dataset_with_embeddings)

    avg_v1 = sum(r.score for r in results_v1) / len(results_v1)
    avg_v2 = sum(r.score for r in results_v2) / len(results_v2)

    print(f"BM25 Relevancy: {avg_v1:.2f}")
    print(f"Embedding Relevancy: {avg_v2:.2f}")
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ContextualRelevancy
    from axion.runners import MetricRunner

    metric = ContextualRelevancy()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        print(f"Relevant: {item_result.signals.relevant_chunks}/{item_result.signals.total_chunks}")
        for i, chunk in enumerate(item_result.signals.chunk_breakdown):
            status = "✅" if chunk.is_relevant else "❌"
            print(f"  {status} Chunk {i+1}: {chunk.chunk_text[:50]}...")
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
<summary><strong>📊 ContextualRelevancyResult Structure</strong></summary>

```python
ContextualRelevancyResult(
{
    "relevancy_score": 0.67,
    "total_chunks": 3,
    "relevant_chunks": 2,
    "chunk_breakdown": [
        {
            "chunk_text": "Paris is the capital and largest city of France.",
            "is_relevant": true
        },
        {
            "chunk_text": "France is known for its wine and cuisine.",
            "is_relevant": false
        },
        {
            "chunk_text": "The Eiffel Tower was built in 1889.",
            "is_relevant": true
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `relevancy_score` | `float` | Ratio of relevant chunks (0.0-1.0) |
| `total_chunks` | `int` | Total chunks retrieved |
| `relevant_chunks` | `int` | Number of relevant chunks |
| `chunk_breakdown` | `List` | Per-chunk verdict details |

### Chunk Breakdown Fields

| Field | Type | Description |
|-------|------|-------------|
| `chunk_text` | `str` | The retrieved chunk content |
| `is_relevant` | `bool` | Whether chunk is relevant to query |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: High Relevancy (Score: 1.0)</strong></summary>

!!! success "All Chunks Relevant"

    **Query:**
    > "How does photosynthesis work?"

    **Retrieved Chunks:**

    1. "Photosynthesis converts light energy into chemical energy."
    2. "Plants use chlorophyll to absorb sunlight."
    3. "The process produces glucose and oxygen from CO2 and water."

    **Analysis:**

    | Chunk | Verdict |
    |-------|---------|
    | Light energy conversion | ✅ Core concept |
    | Chlorophyll absorption | ✅ Key mechanism |
    | Glucose/oxygen production | ✅ Process outputs |

    **Final Score:** `3 / 3 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Mixed Relevancy (Score: 0.5)</strong></summary>

!!! warning "Retrieval Noise"

    **Query:**
    > "What are the symptoms of diabetes?"

    **Retrieved Chunks:**

    1. "Diabetes symptoms include increased thirst and frequent urination."
    2. "Exercise is important for overall health."
    3. "Blurred vision and fatigue are common in diabetic patients."
    4. "Healthy eating includes fruits and vegetables."

    **Analysis:**

    | Chunk | Verdict |
    |-------|---------|
    | Thirst and urination | ✅ Direct symptoms |
    | Exercise importance | ❌ General health, not symptoms |
    | Blurred vision, fatigue | ✅ Diabetes symptoms |
    | Fruits and vegetables | ❌ Diet info, not symptoms |

    **Final Score:** `2 / 4 = 0.5` :material-alert:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Poor Relevancy (Score: 0.0)</strong></summary>

!!! failure "Retrieval Failure"

    **Query:**
    > "What is quantum computing?"

    **Retrieved Chunks:**

    1. "Classical computers use binary bits."
    2. "The internet was invented in the 1960s."
    3. "Programming languages include Python and Java."

    **Analysis:**

    | Chunk | Verdict |
    |-------|---------|
    | Binary bits | ❌ Classical computing, not quantum |
    | Internet history | ❌ Completely off-topic |
    | Programming languages | ❌ Unrelated to quantum concepts |

    **Final Score:** `0 / 3 = 0.0` :material-close:

    *Retrieval completely failed to find quantum computing content.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Retrieval Quality</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Identifies when your retrieval system returns irrelevant documents, causing poor answer quality.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>Debug Isolation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Separates retrieval problems from generation problems. Low relevancy = fix retrieval, not the LLM.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">⚡</span>
<strong>Efficiency</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Irrelevant chunks waste context window space and can confuse the generator.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Contextual Relevancy** = Are the retrieved chunks relevant to the query?

    - **Use it when:** Evaluating or tuning RAG retrieval
    - **Score interpretation:** Higher = more relevant retrieval
    - **Key insight:** Measures retrieval, not generation

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ContextualRelevancy`](../../reference/metrics.md#contextual-relevancy)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Contextual Precision](./contextual_precision.md) · Contextual Recall · Faithfulness

</div>
