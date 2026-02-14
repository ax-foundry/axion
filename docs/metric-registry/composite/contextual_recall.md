# Contextual Recall

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Measure if retrieved context supports the expected answer</strong><br>
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
<small style="color: var(--md-text-muted);">Ratio of supported statements</small>
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
<code>expected_output</code> <code>retrieved_content</code><br>
<small style="color: var(--md-text-muted);">Ground truth required</small>
</div>

</div>

!!! abstract "What It Measures"
    Contextual Recall evaluates whether the **retrieved context contains sufficient information** to support the expected answer. It extracts statements from the ground truth and checks if each is supported by the retrieved chunks. High recall means the retrieval didn't miss important information.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All expected facts are in retrieved context |
    | **0.7+** | :material-check: Most expected facts supported, minor gaps |
    | **0.5** | :material-alert: Half the expected facts missing from context |
    | **< 0.5** | :material-close: Significant information not retrieved |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>You have ground truth answers</li>
<li>Evaluating retrieval completeness</li>
<li>Testing if critical info is retrieved</li>
<li>Debugging "information not found" errors</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>No expected_output available</li>
<li>Multiple valid answers exist</li>
<li>Testing retrieval ranking (use Precision)</li>
<li>Evaluating generation quality</li>
</ul>
</div>

</div>

!!! tip "RAG Evaluation Suite"
    **Contextual Recall** asks: *"Does the context contain everything needed to answer correctly?"*

    Related retrieval metrics:

    - **[Contextual Relevancy](./contextual_relevancy.md)**: Are chunks relevant to the query?
    - **[Contextual Precision](./contextual_precision.md)**: Are useful chunks ranked higher?
    - **[Contextual Sufficiency](./contextual_sufficiency.md)**: Is there enough info overall?

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric extracts factual statements from the expected answer and checks context support.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Expected Output]
            B[Retrieved Context]
        end

        subgraph EXTRACT["🔍 Step 1: Statement Extraction"]
            C[Extract Factual Statements]
            D["Ground Truth Statements"]
        end

        subgraph CHECK["⚖️ Step 2: Support Check"]
            E[Check Against Context]
            F1["Stmt 1: ✓/✗"]
            F2["Stmt 2: ✓/✗"]
            F3["Stmt 3: ✓/✗"]
            FN["..."]
        end

        subgraph SCORE["📊 Step 3: Scoring"]
            G["Count Supported"]
            H["Calculate Ratio"]
            I["Final Score"]
        end

        A --> C
        C --> D
        D --> E
        B --> E
        E --> F1 & F2 & F3 & FN
        F1 & F2 & F3 & FN --> G
        G --> H
        H --> I

        style INPUT stroke:#1E3A5F,stroke-width:2px
        style EXTRACT stroke:#3b82f6,stroke-width:2px
        style CHECK stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style I fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Verdict System"

    Each ground truth statement receives a support verdict.

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ SUPPORTED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1</div>
    <br><small>Statement from expected answer is <strong>found</strong> in retrieved context.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ NOT SUPPORTED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0</div>
    <br><small>Statement from expected answer is <strong>missing</strong> from retrieved context.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = supported_statements / total_statements
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level |

    !!! info "Ground Truth Focus"
        Unlike Contextual Relevancy (which asks "is this chunk relevant?"), Recall asks "is this expected fact present?" It measures retrieval from the answer's perspective.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ContextualRecall
    from axion.dataset import DatasetItem

    metric = ContextualRecall()

    item = DatasetItem(
        expected_output="Paris is the capital of France. It has a population of about 2 million.",
        retrieved_content=[
            "Paris is the capital and largest city of France.",
            "The Eiffel Tower is located in Paris.",
        ],
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 0.5 (capital fact supported, population fact missing)
    ```

=== ":material-target: Complete Recall"

    ```python
    from axion.metrics import ContextualRecall

    metric = ContextualRecall()

    item = DatasetItem(
        expected_output="Python was created by Guido van Rossum in 1991.",
        retrieved_content=[
            "Python is a programming language created by Guido van Rossum.",
            "Python was first released in 1991.",
            "Python emphasizes code readability.",
        ],
    )

    result = await metric.execute(item)
    # Score: 1.0 (both creator and year facts are in context)
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ContextualRecall
    from axion.runners import MetricRunner

    metric = ContextualRecall()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        print(f"Supported: {item_result.signals.supported_gt_statements}/{item_result.signals.total_gt_statements}")
        for stmt in item_result.signals.statement_breakdown:
            status = "✅" if stmt.is_supported else "❌"
            print(f"  {status} {stmt.statement_text}")
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
<summary><strong>📊 ContextualRecallResult Structure</strong></summary>

```python
ContextualRecallResult(
{
    "recall_score": 0.5,
    "total_gt_statements": 2,
    "supported_gt_statements": 1,
    "statement_breakdown": [
        {
            "statement_text": "Paris is the capital of France",
            "is_supported": true
        },
        {
            "statement_text": "Paris has a population of about 2 million",
            "is_supported": false
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `recall_score` | `float` | Ratio of supported statements (0.0-1.0) |
| `total_gt_statements` | `int` | Factual statements from expected output |
| `supported_gt_statements` | `int` | Statements found in context |
| `statement_breakdown` | `List` | Per-statement verdict details |

### Statement Breakdown Fields

| Field | Type | Description |
|-------|------|-------------|
| `statement_text` | `str` | The ground truth statement |
| `is_supported` | `bool` | Whether statement is in context |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Perfect Recall (Score: 1.0)</strong></summary>

!!! success "All Facts Retrieved"

    **Expected Output:**
    > "The Great Wall of China is over 13,000 miles long. It was built over many centuries, starting in the 7th century BC."

    **Retrieved Context:**

    1. "The Great Wall of China stretches over 13,000 miles."
    2. "Construction began in the 7th century BC."
    3. "Multiple dynasties contributed to building the wall over centuries."

    **Analysis:**

    | Statement | Verdict |
    |-----------|---------|
    | Over 13,000 miles long | ✅ Supported |
    | Built over many centuries | ✅ Supported |
    | Starting in 7th century BC | ✅ Supported |

    **Final Score:** `3 / 3 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Partial Recall (Score: 0.5)</strong></summary>

!!! warning "Missing Information"

    **Expected Output:**
    > "Water boils at 100°C at sea level. At higher altitudes, it boils at lower temperatures due to reduced pressure."

    **Retrieved Context:**

    1. "Water boils at 100 degrees Celsius under standard conditions."
    2. "Water is essential for life on Earth."

    **Analysis:**

    | Statement | Verdict |
    |-----------|---------|
    | Boils at 100°C at sea level | ✅ Supported |
    | Higher altitudes = lower boiling point | ❌ Not found |
    | Due to reduced pressure | ❌ Not found |

    **Final Score:** `1 / 3 = 0.33` :material-alert:

    *The altitude/pressure relationship wasn't retrieved.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Poor Recall (Score: 0.0)</strong></summary>

!!! failure "Critical Information Missing"

    **Expected Output:**
    > "Einstein developed the theory of relativity and won the Nobel Prize for the photoelectric effect."

    **Retrieved Context:**

    1. "Albert Einstein was a famous physicist."
    2. "Einstein was born in Germany in 1879."

    **Analysis:**

    | Statement | Verdict |
    |-----------|---------|
    | Developed theory of relativity | ❌ Not found |
    | Won Nobel Prize | ❌ Not found |
    | For photoelectric effect | ❌ Not found |

    **Final Score:** `0 / 3 = 0.0` :material-close:

    *None of the key facts from the expected answer were retrieved.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Completeness Check</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensures retrieval captures all necessary information, not just some of it.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>Answer-Focused</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Evaluates retrieval from the answer's perspective—did we get what's needed to answer correctly?</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🐛</span>
<strong>Debug Missing Info</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Identifies exactly which expected facts weren't retrieved, guiding retrieval improvements.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Contextual Recall** = Does the retrieved context contain all facts from the expected answer?

    - **Use it when:** You have ground truth and want to measure retrieval completeness
    - **Score interpretation:** Higher = more expected facts found in context
    - **Key insight:** Low recall means the retriever missed important information

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ContextualRecall`](../../reference/metrics.md#contextual-recall)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Contextual Precision](./contextual_precision.md) · Contextual Relevancy · Answer Completeness

</div>
