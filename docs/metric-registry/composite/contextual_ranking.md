# Contextual Ranking

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluate if relevant context chunks are ranked higher</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Knowledge</span>
<span class="badge" style="background: #6B7A3A;">Single Turn</span>
<span class="badge" style="background: #06b6d4;">Retrieval</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Precision-weighted ranking score</small>
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
<small style="color: var(--md-text-muted);">No expected_output needed</small>
</div>

</div>

!!! abstract "What It Measures"
    Contextual Ranking evaluates whether **relevant chunks are positioned higher** in retrieval results. Unlike Contextual Precision (which checks usefulness for generating an answer), Ranking simply checks query relevance—making it usable without ground truth.

    | Score | Interpretation |
    |-------|----------------|
    | **≥ 0.9** | :material-check-all: Excellent ranking—relevant chunks at top |
    | **≥ 0.7** | :material-check: Good ranking quality |
    | **0.5** | :material-alert: Mediocre—relevant chunks scattered |
    | **< 0.5** | :material-close: Poor ranking—relevant chunks buried |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>No expected_output available</li>
<li>Evaluating retrieval ranking</li>
<li>Comparing re-ranking algorithms</li>
<li>Testing search relevance</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>You have expected_output (use Precision)</li>
<li>Chunk order doesn't matter</li>
<li>Single-chunk retrieval</li>
<li>Evaluating answer quality</li>
</ul>
</div>

</div>

!!! tip "Ranking vs Precision"
    **Contextual Ranking** checks: *"Are relevant chunks ranked higher?"* (based on query)
    **[Contextual Precision](./contextual_precision.md)** checks: *"Are useful chunks ranked higher?"* (based on expected answer)

    Use Ranking when you don't have ground truth; use Precision when you do.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric evaluates chunk relevance to the query, then calculates a precision-weighted ranking score.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[Retrieved Chunks in Order]
        end

        subgraph EVALUATE["⚖️ Step 1: Relevancy Check"]
            C[RAGAnalyzer Engine]
            D1["Chunk 1: R/✗"]
            D2["Chunk 2: R/✗"]
            D3["Chunk 3: R/✗"]
            DN["..."]
        end

        subgraph RANK["📊 Step 2: Calculate Ranking Score"]
            E["For each relevant chunk at position k"]
            F["Precision@k = relevant_seen / k"]
            G["Sum all Precision@k values"]
            H["Divide by total relevant chunks"]
            I["Final Score"]
        end

        A & B --> C
        C --> D1 & D2 & D3 & DN
        D1 & D2 & D3 & DN --> E
        E --> F
        F --> G
        G --> H
        H --> I

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style EVALUATE stroke:#f59e0b,stroke-width:2px
        style RANK stroke:#10b981,stroke-width:2px
        style I fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-calculator: Ranking Calculation"

    The score heavily penalizes relevant chunks ranked low.

    **Example with 5 chunks (R = relevant, X = not relevant):**

    ```
    Position:  1    2    3    4    5
    Chunks:   [R]  [X]  [R]  [X]  [R]

    Precision@1 = 1/1 = 1.0   (first relevant at position 1)
    Precision@3 = 2/3 = 0.67  (second relevant at position 3)
    Precision@5 = 3/5 = 0.6   (third relevant at position 5)

    Score = (1.0 + 0.67 + 0.6) / 3 = 0.76
    ```

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ RELEVANT</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">+P@k</div>
    <br><small>Chunk is relevant to the query. Contributes to ranking score.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ NOT RELEVANT</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0</div>
    <br><small>Chunk is off-topic. Dilutes precision at each position.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = sum(Precision@k for each relevant chunk) / total_relevant_chunks
        score = clamp(score, 0.0, 1.0)
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level |

    !!! info "Interpretation Guide"

        | Score Range | Quality | Recommendation |
        |-------------|---------|----------------|
        | ≥ 0.9 | Excellent | Ranking is optimal |
        | ≥ 0.7 | Good | Acceptable for most use cases |
        | < 0.7 | Poor | Consider improving re-ranking |
        | 0.0 | None | No relevant chunks found |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ContextualRanking
    from axion.dataset import DatasetItem

    metric = ContextualRanking()

    item = DatasetItem(
        query="What is machine learning?",
        retrieved_content=[
            "Machine learning is a subset of AI.",       # Relevant
            "Python is a programming language.",         # Not relevant
            "ML models learn from data.",                # Relevant
            "The weather is nice today.",                # Not relevant
        ],
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: (1/1 + 2/3) / 2 = 0.83
    ```

=== ":material-sort: Compare Rankings"

    ```python
    from axion.metrics import ContextualRanking

    metric = ContextualRanking()

    # Good ranking: relevant first
    good_order = DatasetItem(
        query="Benefits of exercise",
        retrieved_content=[
            "Exercise improves cardiovascular health.",   # Relevant
            "Regular workouts boost energy levels.",      # Relevant
            "Cooking is a useful skill.",                 # Not relevant
        ],
    )
    # Score: (1/1 + 2/2) / 2 = 1.0

    # Bad ranking: relevant last
    bad_order = DatasetItem(
        query="Benefits of exercise",
        retrieved_content=[
            "Cooking is a useful skill.",                 # Not relevant
            "Exercise improves cardiovascular health.",   # Relevant
            "Regular workouts boost energy levels.",      # Relevant
        ],
    )
    # Score: (1/2 + 2/3) / 2 = 0.58
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ContextualRanking
    from axion.runners import MetricRunner

    metric = ContextualRanking()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Ranking Score: {item_result.score}")
        print(f"Relevant: {item_result.signals.relevant_chunks}/{item_result.signals.total_chunks}")
        for i, chunk in enumerate(item_result.signals.chunk_breakdown):
            status = "✅" if chunk.is_relevant else "❌"
            print(f"  {i+1}. {status} {chunk.chunk_text[:40]}...")
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
<summary><strong>📊 ContextualRankingResult Structure</strong></summary>

```python
ContextualRankingResult(
{
    "final_score": 0.83,
    "relevant_chunks": 2,
    "total_chunks": 4,
    "chunk_breakdown": [
        {
            "chunk_text": "Machine learning is a subset of AI.",
            "is_relevant": true
        },
        {
            "chunk_text": "Python is a programming language.",
            "is_relevant": false
        },
        {
            "chunk_text": "ML models learn from data.",
            "is_relevant": true
        },
        {
            "chunk_text": "The weather is nice today.",
            "is_relevant": false
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `final_score` | `float` | Ranking score (0.0-1.0) |
| `relevant_chunks` | `int` | Number of relevant chunks |
| `total_chunks` | `int` | Total chunks retrieved |
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
<summary><strong>✅ Scenario 1: Excellent Ranking (Score: 1.0)</strong></summary>

!!! success "All Relevant Chunks First"

    **Query:**
    > "How does photosynthesis work?"

    **Retrieved Context (in order):**

    1. "Photosynthesis converts light energy into chemical energy." ✅
    2. "Plants use chlorophyll to absorb sunlight." ✅
    3. "Photosynthesis produces glucose and oxygen." ✅
    4. "The ocean covers 71% of Earth." ❌
    5. "Volcanic eruptions release gases." ❌

    **Ranking Calculation:**

    ```
    Relevant at positions: 1, 2, 3
    P@1 = 1/1 = 1.0
    P@2 = 2/2 = 1.0
    P@3 = 3/3 = 1.0
    Score = (1.0 + 1.0 + 1.0) / 3 = 1.0
    ```

    **Final Score:** `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Mediocre Ranking (Score: 0.5)</strong></summary>

!!! warning "Relevant Chunks Scattered"

    **Query:**
    > "What are the benefits of meditation?"

    **Retrieved Context (in order):**

    1. "Yoga is an ancient practice." ❌
    2. "Meditation reduces stress and anxiety." ✅
    3. "Cooking can be therapeutic." ❌
    4. "Mindfulness improves focus." ✅

    **Ranking Calculation:**

    ```
    Relevant at positions: 2, 4
    P@2 = 1/2 = 0.5
    P@4 = 2/4 = 0.5
    Score = (0.5 + 0.5) / 2 = 0.5
    ```

    **Final Score:** `0.5` :material-alert:

    *Relevant content not prioritized at top positions.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Poor Ranking (Score: 0.33)</strong></summary>

!!! failure "Relevant Chunks at Bottom"

    **Query:**
    > "What is the capital of Japan?"

    **Retrieved Context (in order):**

    1. "Japan has a population of 125 million." ❌
    2. "Japanese cuisine includes sushi." ❌
    3. "Tokyo is the capital of Japan." ✅

    **Ranking Calculation:**

    ```
    Relevant at positions: 3
    P@3 = 1/3 = 0.33
    Score = 0.33 / 1 = 0.33
    ```

    **Final Score:** `0.33` :material-close:

    *The only relevant chunk is last—poor ranking quality.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>No Ground Truth Needed</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Evaluate ranking quality without expected answers—ideal for production monitoring.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📊</span>
<strong>Re-ranker Evaluation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Directly measures whether your re-ranking model improves result ordering.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">⚡</span>
<strong>Context Window Efficiency</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">When using top-k results, good ranking ensures the best content is included.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Contextual Ranking** = Are relevant chunks positioned at the top of results?

    - **Use it when:** Evaluating retrieval ranking without ground truth
    - **Score interpretation:** Higher = relevant chunks appear earlier
    - **Key difference:** Uses query relevance, not answer usefulness

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ContextualRanking`](../../reference/metrics.md#contextual-ranking)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Contextual Precision](./contextual_precision.md) · Contextual Relevancy · Faithfulness

</div>
