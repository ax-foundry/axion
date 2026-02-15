# Factual Accuracy

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Verify AI responses against ground truth statements</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Knowledge</span>
<span class="badge" style="background: #0F2440;">Single Turn</span>
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
<code style="font-size: 1.5rem; color: var(--md-primary);">0.8</code><br>
<small style="color: var(--md-text-muted);">Higher bar for accuracy</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>query</code> <code>actual_output</code> <code>expected_output</code><br>
<small style="color: var(--md-text-muted);">Ground truth required</small>
</div>

</div>

!!! abstract "What It Measures"
    Factual Accuracy calculates the percentage of statements in the AI's response that are **factually supported by the ground truth** (expected_output). Unlike Faithfulness (which checks against retrieved context), this metric verifies against a known-correct answer.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Every statement matches ground truth |
    | **0.8+** | :material-check: Most statements accurate, minor gaps |
    | **0.5** | :material-alert: Half the statements are unsupported |
    | **< 0.5** | :material-close: Significant factual errors |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>You have ground truth answers</li>
<li>Testing against known-correct responses</li>
<li>Evaluating factual Q&A systems</li>
<li>Regression testing AI outputs</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>No expected_output available</li>
<li>Multiple valid answers exist</li>
<li>Testing creative/generative tasks</li>
<li>Ground truth may be incomplete</li>
</ul>
</div>

</div>

!!! tip "See Also: Faithfulness"
    **Factual Accuracy** verifies against *ground truth* (expected_output).
    **[Faithfulness](./faithfulness.md)** verifies against *retrieved context*.

    Use Factual Accuracy when you have known-correct answers; use Faithfulness for RAG systems.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric extracts statements from the AI response and checks each against the ground truth.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[AI Response]
            C[Expected Output]
        end

        subgraph EXTRACT["🔍 Step 1: Statement Extraction"]
            D[Extract Statements from Response]
            E["Atomic Statements"]
        end

        subgraph VERIFY["⚖️ Step 2: Ground Truth Check"]
            F[Compare to Expected Output]
            G["Supported / Not Supported"]
        end

        subgraph SCORE["📊 Step 3: Scoring"]
            H["Count Supported"]
            I["Calculate Ratio"]
            J["Final Score"]
        end

        A & B & C --> D
        D --> E
        E --> F
        C --> F
        F --> G
        G --> H
        H --> I
        I --> J

        style INPUT stroke:#1E3A5F,stroke-width:2px
        style EXTRACT stroke:#3b82f6,stroke-width:2px
        style VERIFY stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style J fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Verdict System"

    Each statement receives a **binary verdict**—either supported or not supported by the ground truth.

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ SUPPORTED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1</div>
    <br><small>Statement is <strong>factually consistent</strong> with the ground truth.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ NOT SUPPORTED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0</div>
    <br><small>Statement is <strong>not found</strong> or <strong>contradicts</strong> the ground truth.</small>
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

    !!! info "Simple Configuration"
        Factual Accuracy has minimal configuration—it focuses on binary correctness against ground truth.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import FactualAccuracy
    from axion.dataset import DatasetItem

    metric = FactualAccuracy()

    item = DatasetItem(
        query="What is the capital of France?",
        actual_output="Paris is the capital of France. It has a population of about 2 million.",
        expected_output="Paris is the capital of France. The city has approximately 2.1 million inhabitants.",
    )

    result = await metric.execute(item)
    print(result.pretty())
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import FactualAccuracy
    from axion.runners import MetricRunner

    metric = FactualAccuracy()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        for verdict in item_result.signals.verdicts:
            status = "✅" if verdict.is_supported else "❌"
            print(f"  {status} {verdict.statement}")
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
<summary><strong>📊 FactualityReport Structure</strong></summary>

```python
FactualityReport(
{
    "verdicts": [
        {
            "statement": "Paris is the capital of France.",
            "is_supported": 1,
            "reason": "The ground truth confirms Paris is the capital of France."
        },
        {
            "statement": "It has a population of about 2 million.",
            "is_supported": 1,
            "reason": "The ground truth states approximately 2.1 million, which aligns with 'about 2 million'."
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `verdicts` | `List[StatementVerdict]` | Per-statement verdicts |

### Statement Verdict Fields

| Field | Type | Description |
|-------|------|-------------|
| `statement` | `str` | The extracted statement |
| `is_supported` | `int` | 1 = supported, 0 = not supported |
| `reason` | `str` | Explanation for the verdict |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Perfect Accuracy (Score: 1.0)</strong></summary>

!!! success "All Statements Supported"

    **Query:**
    > "What year did World War II end?"

    **Expected Output:**
    > "World War II ended in 1945. Germany surrendered in May, and Japan in September."

    **AI Response:**
    > "World War II ended in 1945. Germany surrendered in May, Japan in September."

    **Analysis:**

    | Statement | Verdict | Score |
    |-----------|---------|-------|
    | World War II ended in 1945 | SUPPORTED | 1 |
    | Germany surrendered in May | SUPPORTED | 1 |
    | Japan surrendered in September | SUPPORTED | 1 |

    **Final Score:** `3 / 3 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Partial Accuracy (Score: 0.67)</strong></summary>

!!! warning "Mixed Verdicts"

    **Query:**
    > "What is the speed of light?"

    **Expected Output:**
    > "The speed of light is approximately 299,792 km/s in a vacuum."

    **AI Response:**
    > "The speed of light is about 300,000 km/s. It travels slower through water. Light is the fastest thing in the universe."

    **Analysis:**

    | Statement | Verdict | Score |
    |-----------|---------|-------|
    | Speed of light is about 300,000 km/s | SUPPORTED | 1 |
    | It travels slower through water | NOT SUPPORTED | 0 |
    | Light is the fastest thing in the universe | NOT SUPPORTED | 0 |

    **Final Score:** `1 / 3 = 0.33` :material-alert:

    *The ground truth only mentions vacuum speed—other claims are unsupported.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Poor Accuracy (Score: 0.0)</strong></summary>

!!! failure "No Statements Supported"

    **Query:**
    > "Who wrote Romeo and Juliet?"

    **Expected Output:**
    > "Romeo and Juliet was written by William Shakespeare in the 1590s."

    **AI Response:**
    > "Romeo and Juliet was written by Christopher Marlowe in 1610."

    **Analysis:**

    | Statement | Verdict | Score |
    |-----------|---------|-------|
    | Written by Christopher Marlowe | NOT SUPPORTED | 0 |
    | Written in 1610 | NOT SUPPORTED | 0 |

    **Final Score:** `0 / 2 = 0.0` :material-close:

    *Both claims contradict the ground truth.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>Ground Truth Validation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">When you have known-correct answers, this metric tells you exactly how well your AI matches reality.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🧪</span>
<strong>Regression Testing</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Track factual accuracy over time as you update models or prompts. Catch regressions before deployment.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📊</span>
<strong>Benchmark Evaluation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Compare different models or configurations using the same ground truth dataset.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Factual Accuracy** = Does the AI's response match the known-correct answer?

    - **Use it when:** You have ground truth (expected_output) to compare against
    - **Score interpretation:** Higher = more statements verified against ground truth
    - **Key difference:** Compares to expected_output, not retrieved_content

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.FactualAccuracy`](../../reference/metrics.md#factual-accuracy)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Faithfulness](./faithfulness.md) · Answer Completeness · Answer Relevancy

</div>
