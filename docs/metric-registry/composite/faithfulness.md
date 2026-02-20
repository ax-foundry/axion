---
icon: custom/sliders
---
# Faithfulness

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Measure factual consistency between AI responses and source documents</strong><br>
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
<small style="color: var(--md-text-muted);">Clamped from weighted average</small>
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
<small style="color: var(--md-text-muted);">Three fields needed</small>
</div>

</div>

!!! abstract "What It Measures"
    Faithfulness evaluates whether **every claim** in the AI's response can be directly inferred from the provided source material. It acts as your primary defense against hallucinations—ensuring the AI summarizes existing knowledge rather than inventing facts.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Every claim is fully supported by context |
    | **0.7+** | :material-check: Most claims supported, minor gaps |
    | **0.5** | :material-alert: Threshold—mixture of supported and unsupported |
    | **< 0.5** | :material-close: Significant hallucinations or contradictions |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>RAG systems & document Q&A</li>
<li>Knowledge base assistants</li>
<li>Summarization tasks</li>
<li>Any system with retrieved context</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Creative writing / brainstorming</li>
<li>Opinion or preference questions</li>
<li>No retrieved context available</li>
<li>Open-ended generation tasks</li>
</ul>
</div>

</div>

!!! tip "See Also: Answer Relevancy"
    **Faithfulness** checks if claims are *grounded in the source context* (factual accuracy).
    **[Answer Relevancy](./answer_relevancy.md)** checks if statements *address the user's query* (topical alignment).

    Use both together for comprehensive RAG evaluation.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric uses an **Evaluator LLM** to decompose the response into atomic claims, then verify each against the retrieved context.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[Retrieved Context]
            C[AI Response]
        end

        subgraph EXTRACT["🔍 Step 1: Claim Extraction"]
            D[StatementExtractor LLM]
            E["Atomic Claims<br/><small>Self-contained, verifiable</small>"]
        end

        subgraph VERIFY["⚖️ Step 2: Verification"]
            F[FaithfulnessJudge LLM]
            G["Verdict per Claim"]
        end

        subgraph SCORE["📊 Step 3: Scoring"]
            H["Sum Weighted Verdicts"]
            I["Clamp to [0, 1]"]
            J["Final Score"]
        end

        A & B & C --> D
        D --> E
        E --> F
        B --> F
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

    Each extracted claim receives a **verdict** with a corresponding weight. The final score is the weighted average, clamped to `[0, 1]`.

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ FULLY_SUPPORTED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">+1.0</div>
    <br><small>The claim is <strong>explicitly stated</strong> in the context. Direct evidence exists.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">⚠️ PARTIALLY_SUPPORTED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #f59e0b;">+0.5</div>
    <br><small>Core subject is correct but claim <strong>exaggerates certainty</strong> or has minor inaccuracies.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #6b7280; padding-left: 1rem;">
    <strong style="color: #6b7280;">❓ NO_EVIDENCE</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #6b7280;">0.0</div>
    <br><small>Context doesn't contain information to verify the claim. <strong>Hallucination.</strong></small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ CONTRADICTORY</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">-1.0</div>
    <br><small>Evidence <strong>directly contradicts</strong> the claim. Critical factual error.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = max(0.0, min(1.0, sum(verdict_weights) / total_claims))
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `strict_mode` | `bool` | `False` | When `True`, `NO_EVIDENCE` verdicts receive **-1.0** (same as contradictions), heavily penalizing hallucinations |
    | `verdict_scores` | `Dict[str, float]` | `None` | Custom override for verdict weights. Takes precedence over `strict_mode` |
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level (`GRANULAR` or `HOLISTIC`) |

    !!! warning "Strict Mode"
        Enable `strict_mode=True` for **high-stakes domains** (legal, medical, financial) where any uncited claim is unacceptable—even if not directly contradicted.

=== ":material-tune-variant: Custom Weights"

    Override default verdict weights for domain-specific calibration:

    ```python
    from axion.metrics import Faithfulness

    # Extra penalty for contradictions, higher partial credit
    metric = Faithfulness(
        verdict_scores={
            'FULLY_SUPPORTED': 1.0,
            'PARTIALLY_SUPPORTED': 0.75,  # More generous
            'NO_EVIDENCE': -0.5,          # Moderate penalty
            'CONTRADICTORY': -2.0,        # Severe penalty
        }
    )
    ```

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import Faithfulness
    from axion.dataset import DatasetItem

    # Initialize with defaults
    metric = Faithfulness()

    item = DatasetItem(
        query="What is the infield fly rule in baseball?",
        actual_output="The infield fly rule prevents the defense from intentionally dropping a fly ball to turn a double play.",
        retrieved_content=[
            "The infield fly rule prevents unfair advantage.",
            "Applies with runners on first and second.",
        ],
    )

    result = await metric.execute(item)
    print(result.pretty())
    ```

=== ":material-shield-alert: Strict Mode"

    ```python hl_lines="4"
    from axion.metrics import Faithfulness

    # Zero tolerance for hallucinations
    metric = Faithfulness(strict_mode=True)

    # Any NO_EVIDENCE claim now scores -1.0 instead of 0.0
    # This dramatically lowers scores for responses with uncited claims
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import Faithfulness
    from axion.runners import MetricRunner

    # Initialize with strict mode
    faithfulness = Faithfulness(strict_mode=True)

    runner = MetricRunner(metrics=[faithfulness])
    results = await runner.run(dataset)

    # Access detailed breakdown
    for item_result in results:
        print(f"Score: {item_result.score}")
        print(f"Claims analyzed: {item_result.data.total_claims}")
        for claim in item_result.data.judged_claims:
            print(f"  - {claim.verdict}: {claim.text}")
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
<summary><strong>📊 FaithfulnessResult Structure</strong></summary>

```python
FaithfulnessResult(
{
    "overall_score": 0.5,
    "total_claims": 2,
    "verdict_counts": {
        "fully_supported": 1,
        "partially_supported": 0,
        "no_evidence": 1,
        "contradictory": 0
    },
    "judged_claims": [
        {
            "claim_text": "The infield fly rule prevents the defense from intentionally dropping a fly ball.",
            "faithfulness_verdict": "Fully Supported",
            "reason": "The evidence states that the infield fly rule prevents the defense from intentionally dropping a catchable fly ball."
        },
        {
            "claim_text": "The infield fly rule is designed to prevent an easy double play when runners are on base.",
            "faithfulness_verdict": "No Evidence",
            "reason": "The evidence does not mention anything about preventing an easy double play when runners are on base."
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | `float` | The 0-1 faithfulness score |
| `total_claims` | `int` | Total claims extracted from the response |
| `verdict_counts` | `Dict` | Breakdown by verdict type (`fully_supported`, `partially_supported`, `no_evidence`, `contradictory`) |
| `judged_claims` | `List` | Per-claim verdict details |

### Judged Claim Fields

| Field | Type | Description |
|-------|------|-------------|
| `claim_text` | `str` | The extracted claim text |
| `faithfulness_verdict` | `str` | Verdict: `Fully Supported`, `Partially Supported`, `No Evidence`, or `Contradictory` |
| `reason` | `str` | Human-readable explanation for the verdict |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Perfect Faithfulness (Score: 1.0)</strong></summary>

!!! success "FULLY_SUPPORTED"

    **Context:**
    > "The Apollo 11 mission launched on July 16, 1969. Neil Armstrong was the mission commander. The lunar module was named Eagle."

    **AI Response:**
    > "Apollo 11 launched in July 1969 with Neil Armstrong as commander. The lunar module was called Eagle."

    **Analysis:**

    | Claim | Verdict | Weight |
    |-------|---------|--------|
    | Apollo 11 launched in July 1969 | FULLY_SUPPORTED | +1.0 |
    | Neil Armstrong was commander | FULLY_SUPPORTED | +1.0 |
    | Lunar module was called Eagle | FULLY_SUPPORTED | +1.0 |

    **Final Score:** `3.0 / 3 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Partial Support (Score: 0.5)</strong></summary>

!!! warning "Mixed Verdicts"

    **Context:**
    > "Our refund policy allows returns within 30 days. Items must be unused and in original packaging."

    **AI Response:**
    > "You can return items within 30 days if unused. Refunds are processed within 24 hours."

    **Analysis:**

    | Claim | Verdict | Weight |
    |-------|---------|--------|
    | Returns within 30 days if unused | FULLY_SUPPORTED | +1.0 |
    | Refunds processed within 24 hours | NO_EVIDENCE | 0.0 |

    **Final Score:** `1.0 / 2 = 0.5` :material-alert:

    *In strict mode: `(1.0 + -1.0) / 2 = 0.0`*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Contradiction (Score: 0.0)</strong></summary>

!!! failure "CONTRADICTORY"

    **Context:**
    > "The maximum dosage is 500mg per day. Do not exceed this limit."

    **AI Response:**
    > "You can safely take up to 1000mg daily."

    **Analysis:**

    | Claim | Verdict | Weight |
    |-------|---------|--------|
    | Safe to take up to 1000mg daily | CONTRADICTORY | -1.0 |

    **Final Score:** `max(0, -1.0 / 1) = 0.0` :material-close:

    *Critical: This response could cause patient harm.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🛡️</span>
<strong>Risk Mitigation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Primary guardrail against hallucinations. Protects your brand from legal and reputational liability caused by invented facts.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">✅</span>
<strong>User Trust</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Essential for high-stakes domains (legal, financial, medical) where users must trust the AI is summarizing, not creating.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Debug Isolation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Distinguishes <strong>retrieval errors</strong> (wrong docs found) from <strong>generation errors</strong> (right docs ignored).</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Faithfulness** = Does the AI's response stick to the facts in the retrieved documents?

    - **Use it when:** You need to ensure AI responses don't contain hallucinations
    - **Score interpretation:** Higher = more grounded in source material
    - **Key config:** Enable `strict_mode` for zero-tolerance on uncited claims

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.Faithfulness`](../../reference/metrics.md#faithfulness)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Answer Relevancy](../../guides/metrics.md) · Context Precision · Factual Accuracy

</div>
