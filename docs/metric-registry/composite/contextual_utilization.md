# Contextual Utilization

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Measure the efficiency of context usage in generation</strong><br>
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
<small style="color: var(--md-text-muted);">Ratio of utilized chunks</small>
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
<code>query</code> <code>actual_output</code> <code>retrieved_content</code><br>
<small style="color: var(--md-text-muted);">Answer + context required</small>
</div>

</div>

!!! abstract "What It Measures"
    Contextual Utilization measures the **efficiency of context usage**—what proportion of relevant retrieved chunks were actually used in the generated answer. Low utilization means relevant information was retrieved but ignored.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All relevant chunks were utilized |
    | **0.7+** | :material-check: Good utilization, minor waste |
    | **0.5** | :material-alert: Half of relevant chunks unused |
    | **< 0.5** | :material-close: Significant waste—relevant info ignored |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Optimizing context window usage</li>
<li>Debugging incomplete answers</li>
<li>Identifying generation issues</li>
<li>Measuring retrieval efficiency</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>No actual_output available</li>
<li>Evaluating retrieval only</li>
<li>Testing factual correctness</li>
<li>All chunks are equally important</li>
</ul>
</div>

</div>

!!! tip "Utilization vs Faithfulness"
    **Contextual Utilization** asks: *"Was the relevant context actually used?"*
    **[Faithfulness](./faithfulness.md)** asks: *"Is the answer grounded in context?"*

    High Faithfulness + Low Utilization = Answer is correct but incomplete (missed relevant info).

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric evaluates which relevant chunks were actually utilized in the answer.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[Retrieved Chunks]
            C[Generated Answer]
        end

        subgraph FILTER["🔍 Step 1: Identify Relevant"]
            D[Check Chunk Relevancy]
            E["Relevant Chunks Only"]
        end

        subgraph CHECK["⚖️ Step 2: Check Utilization"]
            F[Compare to Answer]
            G1["Chunk 1: ✓/✗"]
            G2["Chunk 2: ✓/✗"]
            G3["Chunk 3: ✓/✗"]
            GN["..."]
        end

        subgraph SCORE["📊 Step 3: Scoring"]
            H["Count Utilized"]
            I["Calculate Ratio"]
            J["Final Score"]
        end

        A & B --> D
        D --> E
        E --> F
        C --> F
        F --> G1 & G2 & G3 & GN
        G1 & G2 & G3 & GN --> H
        H --> I
        I --> J

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style FILTER stroke:#3b82f6,stroke-width:2px
        style CHECK stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style J fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Verdict System"

    Each **relevant** chunk is checked for utilization.

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ UTILIZED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1</div>
    <br><small>Information from this chunk <strong>appears</strong> in the generated answer.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ NOT UTILIZED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0</div>
    <br><small>Relevant chunk was <strong>ignored</strong>—information not used in answer.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = utilized_chunks / total_relevant_chunks
        ```
        Only relevant chunks are counted—irrelevant chunks don't affect the score.

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level |

    !!! info "Relevance Filtering"
        The metric first filters to only relevant chunks (using the same logic as Contextual Relevancy), then checks which of those were utilized. This means irrelevant chunks don't penalize or inflate the score.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ContextualUtilization
    from axion.dataset import DatasetItem

    metric = ContextualUtilization()

    item = DatasetItem(
        query="What are the health benefits of green tea?",
        actual_output="Green tea contains antioxidants that reduce inflammation.",
        retrieved_content=[
            "Green tea is rich in antioxidants.",                    # Relevant, utilized
            "Antioxidants help reduce inflammation.",                # Relevant, utilized
            "Green tea can boost metabolism.",                       # Relevant, NOT utilized
            "Tea originated in China thousands of years ago.",       # Not relevant
        ],
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 2/3 = 0.67 (2 of 3 relevant chunks utilized)
    ```

=== ":material-check-all: Full Utilization"

    ```python
    from axion.metrics import ContextualUtilization

    metric = ContextualUtilization()

    item = DatasetItem(
        query="What is Python?",
        actual_output="Python is a high-level programming language created by Guido van Rossum, known for its readability.",
        retrieved_content=[
            "Python is a high-level programming language.",          # Utilized
            "Guido van Rossum created Python.",                      # Utilized
            "Python emphasizes code readability.",                   # Utilized
        ],
    )

    result = await metric.execute(item)
    # Score: 1.0 (all relevant chunks utilized)
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ContextualUtilization
    from axion.runners import MetricRunner

    metric = ContextualUtilization()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Utilization: {item_result.score:.0%}")
        print(f"Used: {item_result.signals.utilized_chunks}/{item_result.signals.total_relevant_chunks}")
        for chunk in item_result.signals.chunk_breakdown:
            status = "✅" if chunk.is_utilized else "❌"
            print(f"  {status} {chunk.chunk_text[:40]}...")
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
<summary><strong>📊 ContextualUtilizationResult Structure</strong></summary>

```python
ContextualUtilizationResult(
{
    "utilization_score": 0.67,
    "total_relevant_chunks": 3,
    "utilized_chunks": 2,
    "utilization_rate": "66.7%",
    "chunk_breakdown": [
        {
            "chunk_text": "Green tea is rich in antioxidants.",
            "is_utilized": true
        },
        {
            "chunk_text": "Antioxidants help reduce inflammation.",
            "is_utilized": true
        },
        {
            "chunk_text": "Green tea can boost metabolism.",
            "is_utilized": false
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `utilization_score` | `float` | Ratio of utilized chunks (0.0-1.0) |
| `total_relevant_chunks` | `int` | Relevant chunks in context |
| `utilized_chunks` | `int` | Chunks actually used in answer |
| `utilization_rate` | `str` | Human-readable percentage |
| `chunk_breakdown` | `List` | Per-chunk (relevant only) details |

### Chunk Breakdown Fields

| Field | Type | Description |
|-------|------|-------------|
| `chunk_text` | `str` | The relevant chunk content |
| `is_utilized` | `bool` | Whether chunk was used in answer |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Full Utilization (Score: 1.0)</strong></summary>

!!! success "All Relevant Info Used"

    **Query:**
    > "What is the boiling point of water?"

    **Retrieved Context:**

    1. "Water boils at 100°C at sea level." ✅ Relevant
    2. "This is equivalent to 212°F." ✅ Relevant
    3. "Ice cream is a popular dessert." ❌ Not relevant

    **Generated Answer:**
    > "Water boils at 100°C (212°F) at sea level."

    **Analysis:**

    | Relevant Chunk | Utilized |
    |----------------|----------|
    | Boils at 100°C | ✅ Used |
    | Equivalent to 212°F | ✅ Used |

    **Final Score:** `2 / 2 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Partial Utilization (Score: 0.5)</strong></summary>

!!! warning "Relevant Info Ignored"

    **Query:**
    > "What are the benefits of exercise?"

    **Retrieved Context:**

    1. "Exercise improves cardiovascular health." ✅ Relevant
    2. "Regular exercise boosts mood and energy." ✅ Relevant
    3. "Exercise helps with weight management." ✅ Relevant
    4. "Gyms offer various equipment." ❌ Not relevant

    **Generated Answer:**
    > "Exercise improves cardiovascular health and boosts mood."

    **Analysis:**

    | Relevant Chunk | Utilized |
    |----------------|----------|
    | Cardiovascular health | ✅ Used |
    | Mood and energy | ✅ Used (partial) |
    | Weight management | ❌ Not used |

    **Final Score:** `2 / 3 = 0.67` :material-alert:

    *Weight management benefit was retrieved but not mentioned.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Poor Utilization (Score: 0.25)</strong></summary>

!!! failure "Most Relevant Info Wasted"

    **Query:**
    > "Explain the causes of World War I."

    **Retrieved Context:**

    1. "Assassination of Archduke Franz Ferdinand triggered WWI." ✅ Relevant
    2. "Alliance systems escalated regional conflicts." ✅ Relevant
    3. "Nationalism and imperialism created tensions." ✅ Relevant
    4. "The war lasted from 1914 to 1918." ✅ Relevant

    **Generated Answer:**
    > "World War I began after the assassination of Archduke Franz Ferdinand."

    **Analysis:**

    | Relevant Chunk | Utilized |
    |----------------|----------|
    | Assassination | ✅ Used |
    | Alliance systems | ❌ Not used |
    | Nationalism/imperialism | ❌ Not used |
    | Duration | ❌ Not used |

    **Final Score:** `1 / 4 = 0.25` :material-close:

    *Only one cause mentioned despite retrieving multiple.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>Answer Completeness</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Low utilization often indicates incomplete answers that miss relevant information.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">💰</span>
<strong>Efficiency</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Retrieved content costs tokens. Low utilization = wasted context window space.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔧</span>
<strong>Debug Generation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">If retrieval is good but utilization is low, the problem is in generation, not retrieval.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Contextual Utilization** = Was the relevant retrieved context actually used in the answer?

    - **Use it when:** Debugging incomplete answers or optimizing context usage
    - **Score interpretation:** Higher = more efficient use of retrieved information
    - **Key insight:** Measures generation efficiency, not retrieval quality

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ContextualUtilization`](../../reference/metrics.md#contextual-utilization)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Faithfulness](./faithfulness.md) · Contextual Relevancy · Answer Completeness

</div>
