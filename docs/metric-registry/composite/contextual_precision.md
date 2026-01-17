# Contextual Precision

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluate if useful context chunks are ranked higher</strong><br>
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
<small style="color: var(--md-text-muted);">Mean Average Precision</small>
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
<code>query</code> <code>expected_output</code> <code>retrieved_content</code><br>
<small style="color: var(--md-text-muted);">Ground truth + context</small>
</div>

</div>

!!! abstract "What It Measures"
    Contextual Precision evaluates whether **useful chunks are ranked higher** in the retrieval results. Using Mean Average Precision (MAP), it rewards retrieval systems that place the most helpful documents at the top. A useful chunk is one that contributes to generating the expected answer.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All useful chunks at the top |
    | **0.7+** | :material-check: Good ranking, useful chunks near top |
    | **0.5** | :material-alert: Mixed ranking quality |
    | **< 0.5** | :material-close: Useful chunks buried low in results |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Evaluating retrieval ranking quality</li>
<li>Tuning re-ranking algorithms</li>
<li>Testing with limited context windows</li>
<li>Optimizing for top-k retrieval</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>No expected_output available</li>
<li>Chunk order doesn't matter</li>
<li>Using all retrieved chunks equally</li>
<li>Single-chunk retrieval</li>
</ul>
</div>

</div>

!!! tip "RAG Evaluation Suite"
    **Contextual Precision** asks: *"Are the most useful chunks ranked first?"*

    Related retrieval metrics:

    - **[Contextual Relevancy](./contextual_relevancy.md)**: Are chunks relevant to the query?
    - **[Contextual Recall](./contextual_recall.md)**: Do chunks cover the expected answer?
    - **[Contextual Ranking](./contextual_ranking.md)**: Are relevant chunks ranked higher?

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric evaluates chunk usefulness for generating the expected answer, then calculates MAP.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[Expected Output]
            C[Retrieved Chunks in Order]
        end

        subgraph EVALUATE["⚖️ Step 1: Usefulness Check"]
            D[RAGAnalyzer Engine]
            E["Useful / Not Useful per Chunk"]
        end

        subgraph MAP["📊 Step 2: Calculate MAP"]
            F["For each useful chunk at position k"]
            G["Precision@k = useful_seen / k"]
            H["Average all Precision@k values"]
            I["Final MAP Score"]
        end

        A & B & C --> D
        D --> E
        E --> F
        F --> G
        G --> H
        H --> I

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style EVALUATE stroke:#f59e0b,stroke-width:2px
        style MAP stroke:#10b981,stroke-width:2px
        style I fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-calculator: MAP Calculation"

    Mean Average Precision rewards useful chunks appearing early in the ranking.

    **Example with 5 chunks (U = useful, X = not useful):**

    ```
    Position:  1    2    3    4    5
    Chunks:   [U]  [X]  [U]  [X]  [U]

    Precision@1 = 1/1 = 1.0   (first useful at position 1)
    Precision@3 = 2/3 = 0.67  (second useful at position 3)
    Precision@5 = 3/5 = 0.6   (third useful at position 5)

    MAP = (1.0 + 0.67 + 0.6) / 3 = 0.76
    ```

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ USEFUL</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">+P@k</div>
    <br><small>Chunk helps generate the expected answer. Contributes to MAP score.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ NOT USEFUL</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0</div>
    <br><small>Chunk doesn't contribute to the answer. Dilutes precision.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        MAP = sum(Precision@k for each useful chunk) / total_useful_chunks
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level |

    !!! info "Usefulness vs Relevancy"
        - **Relevancy** (Contextual Relevancy): "Is this chunk about the topic?"
        - **Usefulness** (Contextual Precision): "Does this chunk help generate the correct answer?"

        A chunk can be relevant but not useful (e.g., background info that isn't needed).

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ContextualPrecision
    from axion.dataset import DatasetItem

    metric = ContextualPrecision()

    item = DatasetItem(
        query="Who invented the telephone?",
        expected_output="Alexander Graham Bell invented the telephone in 1876.",
        retrieved_content=[
            "Alexander Graham Bell invented the telephone.",  # Useful
            "The telephone revolutionized communication.",    # Not useful
            "Bell patented it in 1876.",                      # Useful
        ],
    )

    result = await metric.execute(item)
    print(result.pretty())
    # MAP: (1/1 + 2/3) / 2 = 0.83
    ```

=== ":material-sort: Perfect vs Poor Ranking"

    ```python
    from axion.metrics import ContextualPrecision

    metric = ContextualPrecision()

    # Perfect ranking: useful chunks first
    perfect_order = DatasetItem(
        query="What is Python?",
        expected_output="Python is a programming language created by Guido van Rossum.",
        retrieved_content=[
            "Python is a high-level programming language.",   # Useful (pos 1)
            "Guido van Rossum created Python.",               # Useful (pos 2)
            "Programming is fun.",                            # Not useful
        ],
    )
    # MAP = (1/1 + 2/2) / 2 = 1.0

    # Poor ranking: useful chunks last
    poor_order = DatasetItem(
        query="What is Python?",
        expected_output="Python is a programming language created by Guido van Rossum.",
        retrieved_content=[
            "Programming is fun.",                            # Not useful
            "Python is a high-level programming language.",   # Useful (pos 2)
            "Guido van Rossum created Python.",               # Useful (pos 3)
        ],
    )
    # MAP = (1/2 + 2/3) / 2 = 0.58
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ContextualPrecision
    from axion.runners import MetricRunner

    metric = ContextualPrecision()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"MAP Score: {item_result.score}")
        print(f"Useful chunks: {item_result.signals.useful_chunks}/{item_result.signals.total_chunks}")
        print(f"First useful at position: {item_result.signals.first_useful_position}")
        for i, chunk in enumerate(item_result.signals.chunk_breakdown):
            status = "✅" if chunk.is_useful else "❌"
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
<summary><strong>📊 ContextualPrecisionResult Structure</strong></summary>

```python
ContextualPrecisionResult(
{
    "map_score": 0.83,
    "total_chunks": 3,
    "useful_chunks": 2,
    "first_useful_position": 1,
    "chunk_breakdown": [
        {
            "chunk_text": "Alexander Graham Bell invented the telephone.",
            "is_useful": true,
            "position": 1
        },
        {
            "chunk_text": "The telephone revolutionized communication.",
            "is_useful": false,
            "position": 2
        },
        {
            "chunk_text": "Bell patented it in 1876.",
            "is_useful": true,
            "position": 3
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `map_score` | `float` | Mean Average Precision (0.0-1.0) |
| `total_chunks` | `int` | Total chunks retrieved |
| `useful_chunks` | `int` | Chunks useful for generating answer |
| `first_useful_position` | `int` | Rank of first useful chunk |
| `chunk_breakdown` | `List` | Per-chunk verdict details |

### Chunk Breakdown Fields

| Field | Type | Description |
|-------|------|-------------|
| `chunk_text` | `str` | The retrieved chunk content |
| `is_useful` | `bool` | Whether chunk helps generate expected answer |
| `position` | `int` | Rank position in retrieval results |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Perfect Precision (Score: 1.0)</strong></summary>

!!! success "Useful Chunks Ranked First"

    **Query:**
    > "What are the three states of matter?"

    **Expected Output:**
    > "The three states of matter are solid, liquid, and gas."

    **Retrieved Context (in order):**

    1. "Matter exists in three states: solid, liquid, and gas." ✅
    2. "Solids have fixed shape, liquids take container shape." ✅
    3. "Gases expand to fill available space." ✅
    4. "Matter is anything that has mass." ❌
    5. "Chemistry is the study of matter." ❌

    **MAP Calculation:**

    ```
    Useful at positions: 1, 2, 3
    P@1 = 1/1 = 1.0
    P@2 = 2/2 = 1.0
    P@3 = 3/3 = 1.0
    MAP = (1.0 + 1.0 + 1.0) / 3 = 1.0
    ```

    **Final Score:** `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Mixed Precision (Score: 0.58)</strong></summary>

!!! warning "Useful Chunks Buried"

    **Query:**
    > "Who wrote Romeo and Juliet?"

    **Expected Output:**
    > "William Shakespeare wrote Romeo and Juliet."

    **Retrieved Context (in order):**

    1. "Shakespeare was born in Stratford-upon-Avon." ❌
    2. "Romeo and Juliet is a famous tragedy." ❌
    3. "William Shakespeare wrote Romeo and Juliet." ✅
    4. "The play was written in the 1590s." ✅

    **MAP Calculation:**

    ```
    Useful at positions: 3, 4
    P@3 = 1/3 = 0.33
    P@4 = 2/4 = 0.50
    MAP = (0.33 + 0.50) / 2 = 0.42
    ```

    **Final Score:** `0.42` :material-alert:

    *Key information buried at positions 3-4 instead of 1-2.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Poor Precision (Score: 0.2)</strong></summary>

!!! failure "Useful Chunk at Bottom"

    **Query:**
    > "What is the speed of light?"

    **Expected Output:**
    > "The speed of light is approximately 299,792 km/s."

    **Retrieved Context (in order):**

    1. "Light is a form of electromagnetic radiation." ❌
    2. "Light travels in waves." ❌
    3. "Light can be refracted through prisms." ❌
    4. "Light behaves as both particles and waves." ❌
    5. "The speed of light is about 300,000 km/s in vacuum." ✅

    **MAP Calculation:**

    ```
    Useful at positions: 5
    P@5 = 1/5 = 0.2
    MAP = 0.2 / 1 = 0.2
    ```

    **Final Score:** `0.2` :material-close:

    *The only useful chunk is at the very bottom.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">📊</span>
<strong>Ranking Quality</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Measures not just what you retrieve, but how well you rank it. Critical for top-k systems.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">⚡</span>
<strong>Efficiency</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">When context windows are limited, having useful chunks first means better answers faster.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔧</span>
<strong>Re-ranking Tuning</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Directly measures re-ranking model performance. Low MAP = improve your re-ranker.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Contextual Precision** = Are the most useful chunks ranked at the top?

    - **Use it when:** Evaluating retrieval ranking, especially with limited context windows
    - **Score interpretation:** Higher MAP = useful chunks appear earlier
    - **Key formula:** Mean Average Precision over useful chunk positions

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ContextualPrecision`](../../reference/metrics.md#contextual-precision)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Contextual Ranking](./contextual_ranking.md) · Contextual Recall · Contextual Relevancy

</div>
