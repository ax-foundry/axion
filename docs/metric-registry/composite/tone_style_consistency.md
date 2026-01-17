# Tone & Style Consistency

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluate if responses match the expected tone, persona, and formatting style</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Knowledge</span>
<span class="badge" style="background: #6B7A3A;">Single Turn</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Style alignment score</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.8</code><br>
<small style="color: var(--md-text-muted);">Higher bar for consistency</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code> <code>expected_output</code><br>
<small style="color: var(--md-text-muted);">Optional: <code>persona_description</code></small>
</div>

</div>

!!! abstract "What It Measures"
    Tone & Style Consistency evaluates whether a response matches the **emotional tone** and **writing style** of an expected answer. For customer service agents, "Voice" is as important as "Fact"—this metric ensures your AI maintains the right persona.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Perfect match—exact emotion, enthusiasm, formatting |
    | **0.8** | :material-check: Minor drift—generally correct but slightly off |
    | **0.5** | :material-alert: Significant mismatch—wrong tone or style |
    | **0.0** | :material-close: Complete failure—robotic, rude, or ignores persona |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Building customer service agents</li>
<li>Persona consistency matters</li>
<li>Brand voice guidelines exist</li>
<li>Comparing against reference responses</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>No expected_output available</li>
<li>Tone flexibility is acceptable</li>
<li>Only factual accuracy matters</li>
<li>Creative writing tasks</li>
</ul>
</div>

</div>

!!! tip "See Also: Answer Completeness"
    **Tone & Style Consistency** evaluates *how* something is said (voice, formatting).
    **[Answer Completeness](./answer_completeness.md)** evaluates *what* is said (content coverage).

    Use both together for comprehensive response quality evaluation.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric uses an LLM-based judge to evaluate both emotional tone and writing style.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Actual Output]
            B[Expected Output]
            C[Persona Description]
        end

        subgraph ANALYZE["🔍 Step 1: Style Analysis"]
            D[ToneJudge LLM]
            E["Tone & Style Comparison"]
        end

        subgraph EVALUATE["⚖️ Step 2: Dimension Scoring"]
            F[Evaluate Tone Match]
            G[Evaluate Style Match]
            H["Identify Differences"]
        end

        subgraph SCORE["📊 Step 3: Final Score"]
            I["Combine Dimensions"]
            J["Final Score"]
        end

        A & B & C --> D
        D --> E
        E --> F & G
        F & G --> H
        H --> I
        I --> J

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style ANALYZE stroke:#3b82f6,stroke-width:2px
        style EVALUATE stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style J fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Scoring Rubric"

    The metric evaluates responses on a detailed rubric with clear benchmarks.

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ PERFECT MATCH</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1.0</div>
    <br><small>Exact emotion, enthusiasm level, and formatting style.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">📊 MINOR DRIFT</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #3b82f6;">0.8</div>
    <br><small>Generally correct but slightly less enthusiastic or formal.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">⚠️ SIGNIFICANT MISMATCH</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #f59e0b;">0.5</div>
    <br><small>Neutral when should be excited, or style completely different.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ COMPLETE FAILURE</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0.0</div>
    <br><small>Robotic, rude, or completely ignores persona.</small>
    </div>

    </div>

    !!! info "Two Dimensions"
        - **Tone Match**: Emotional alignment (enthusiasm, empathy, formality)
        - **Style Match**: Formatting, length, vocabulary, structure

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `persona_description` | `str` | `None` | Optional persona to enforce (e.g., "Helpful, excited, professional") |
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level |

    !!! info "Persona Description"
        When provided, the persona description guides the judge on expected tone characteristics, making evaluation more precise for specific brand voices.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ToneStyleConsistency
    from axion.dataset import DatasetItem

    metric = ToneStyleConsistency()

    item = DatasetItem(
        actual_output="Your order has been shipped. It will arrive in 3-5 business days.",
        expected_output="Great news! 🎉 Your order is on its way! You can expect it within 3-5 business days. We're so excited for you!",
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 0.5 (tone mismatch - neutral vs enthusiastic)
    ```

=== ":material-account: With Persona"

    ```python
    from axion.metrics import ToneStyleConsistency

    # Define expected persona
    metric = ToneStyleConsistency()

    item = DatasetItem(
        actual_output="I apologize for the inconvenience. Let me help resolve this.",
        expected_output="I'm truly sorry this happened. I completely understand your frustration, and I'm here to make things right!",
        persona_description="Empathetic, warm, solution-oriented customer service agent",
    )

    result = await metric.execute(item)
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ToneStyleConsistency
    from axion.runners import MetricRunner

    metric = ToneStyleConsistency()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        print(f"Tone Match: {item_result.signals.tone_match}")
        print(f"Style Match: {item_result.signals.style_match}")
        for diff in item_result.signals.differences:
            print(f"  - {diff.dimension}: {diff.description}")
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
<summary><strong>📊 ToneStyleResult Structure</strong></summary>

```python
ToneStyleResult(
{
    "final_score": 0.5,
    "tone_match": false,
    "style_match": true,
    "differences": [
        {
            "dimension": "Enthusiasm Level",
            "expected": "Excited, celebratory with emoji",
            "actual": "Neutral, matter-of-fact",
            "impact": "Major - missed opportunity to delight customer"
        },
        {
            "dimension": "Emotional Warmth",
            "expected": "Personal, caring language",
            "actual": "Formal, impersonal",
            "impact": "Moderate - feels robotic"
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `final_score` | `float` | Overall tone & style alignment score |
| `tone_match` | `bool` | Whether emotional tone matches expected |
| `style_match` | `bool` | Whether formatting/writing style matches |
| `differences` | `List` | Specific differences identified |

### Difference Fields

| Field | Type | Description |
|-------|------|-------------|
| `dimension` | `str` | Aspect that differs (e.g., "Enthusiasm Level") |
| `expected` | `str` | What was expected |
| `actual` | `str` | What was observed |
| `impact` | `str` | Severity of the mismatch |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Perfect Match (Score: 1.0)</strong></summary>

!!! success "Tone & Style Aligned"

    **Expected Output:**
    > "Hi there! 👋 Thanks for reaching out! I'd be happy to help you with your question about returns. Our policy allows full refunds within 30 days!"

    **AI Response:**
    > "Hello! 😊 Thanks so much for contacting us! I'm thrilled to assist with your returns question. You can get a full refund within 30 days—no problem at all!"

    **Analysis:**

    | Dimension | Match |
    |-----------|-------|
    | Enthusiasm | ✅ Both excited and welcoming |
    | Emoji usage | ✅ Appropriate friendly emoji |
    | Formality | ✅ Casual, approachable |
    | Helpfulness | ✅ Eager to assist |

    **Final Score:** `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Style Drift (Score: 0.5)</strong></summary>

!!! warning "Tone Mismatch"

    **Expected Output:**
    > "Great news! 🎉 Your order is on its way! You can expect delivery within 3-5 business days. We're so excited for you!"

    **AI Response:**
    > "Your order has been shipped. Estimated delivery: 3-5 business days."

    **Analysis:**

    | Dimension | Match |
    |-----------|-------|
    | Information | ✅ Same facts conveyed |
    | Enthusiasm | ❌ Neutral vs celebratory |
    | Emoji usage | ❌ None vs appropriate celebration |
    | Warmth | ❌ Impersonal vs personal |

    **Final Score:** `0.5` :material-alert:

    *Content is correct but voice is completely wrong.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Complete Mismatch (Score: 0.0)</strong></summary>

!!! failure "Persona Ignored"

    **Expected Output:**
    > "I'm so sorry to hear about this issue! 😔 That's definitely not the experience we want for you. Let me personally look into this right away and make it right!"

    **AI Response:**
    > "Your complaint has been logged. Reference number: #12345. Allow 5-7 business days for review."

    **Analysis:**

    | Dimension | Match |
    |-----------|-------|
    | Empathy | ❌ None vs deeply apologetic |
    | Tone | ❌ Cold/bureaucratic vs warm |
    | Personal touch | ❌ Ticket number vs personal commitment |
    | Resolution focus | ❌ Process vs solution |

    **Final Score:** `0.0` :material-close:

    *Response is robotic when empathy was expected.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🎭</span>
<strong>Brand Voice</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensures AI maintains your brand's personality across all interactions. Inconsistent tone damages brand perception.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">💬</span>
<strong>Customer Experience</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Customers expect warmth and empathy, not robotic responses. Tone directly impacts satisfaction and loyalty.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔄</span>
<strong>Consistency</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Maintain uniform voice across all AI-generated responses, regardless of the underlying model or prompt.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Tone & Style Consistency** = Does the AI response sound like it should?

    - **Use it when:** Brand voice and persona consistency matter
    - **Score interpretation:** Higher = better alignment with expected tone
    - **Key difference:** Measures *how* something is said, not *what* is said

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ToneStyleConsistency`](../../reference/metrics.md#tone-style-consistency)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Answer Completeness](./answer_completeness.md) · Answer Relevancy · Answer Criteria

</div>
