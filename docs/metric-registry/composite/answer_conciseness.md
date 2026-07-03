---
icon: custom/sliders
---
# Answer Conciseness

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluates if the response is concise or verbose compared to the expected answer, identifying specific redundant phrases</strong><br>
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
<small style="color: var(--md-text-muted);">1.0 = maximally concise</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.8</code><br>
<small style="color: var(--md-text-muted);">Pass/fail cutoff</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code> <code>expected_output</code><br>
<small style="color: var(--md-text-muted);">Optional: <code>query</code></small>
</div>

</div>

!!! abstract "What It Measures"
    Answer Conciseness evaluates whether the AI's response contains **unnecessary verbosity** relative to the expected answer. An LLM identifies redundant phrases, filler clauses, and padding not present in the reference output and penalizes each occurrence. A score of 1.0 means the response matches the expected answer's economy of language; lower scores indicate detectable bloat.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Response is as concise as the expected answer |
    | **0.8+** | :material-check: Minor verbosity; within acceptable threshold |
    | **0.5** | :material-alert: Noticeable redundancy — several filler phrases detected |
    | **< 0.5** | :material-close: Significant verbosity; response is much longer than needed |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Response conciseness matters and a reference answer is available</li>
<li>QA or summarization tasks where brevity is important</li>
<li>Evaluating chatbots or agents that should give direct, focused answers</li>
<li>Detecting prompt-injected padding or unnecessary hedging</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Creative writing or exploratory tasks where verbose output is appropriate</li>
<li>No <code>expected_output</code> is available (it is required)</li>
<li>Evaluating factual accuracy — use <a href="./faithfulness.md">Faithfulness</a> instead</li>
<li>Tasks where longer explanations are inherently desirable</li>
</ul>
</div>

</div>

!!! tip "See Also: Answer Completeness"
    **Answer Conciseness** penalizes *excess* content not in the reference answer.
    **[Answer Completeness](./answer_completeness.md)** penalizes *missing* content that should have been included.

    Use both together to catch responses that are simultaneously incomplete on one topic and verbose on another.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric uses an **Evaluator LLM** to compare the actual output against the expected output, extracting redundant phrases and clauses absent from the reference. Each redundant phrase reduces the score.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[actual_output]
            B[expected_output]
            C["query (optional)"]
        end

        subgraph EXTRACT["🔍 Step 1: Redundancy Detection"]
            D[ConcissnessJudge LLM]
            E["Redundant Phrases<br/><small>Clauses absent from expected_output</small>"]
        end

        subgraph SCORE["📊 Step 2: Scoring"]
            F["Penalize per redundant phrase"]
            G["Clamp to [0, 1]"]
            H["Final Score"]
        end

        A & B & C --> D
        D --> E
        E --> F
        F --> G
        G --> H

        style INPUT stroke:#1E3A5F,stroke-width:2px
        style EXTRACT stroke:#3b82f6,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style H fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Scoring Logic"

    The LLM enumerates each redundant phrase detected in the actual output that is not justified by the expected output. The score is computed as:

    !!! tip "Score Formula"
        ```
        score = max(0.0, min(1.0, 1.0 - (redundant_phrase_count / normalizer)))
        ```

        The normalizer is calibrated to the length and complexity of the expected output so that longer reference answers do not unfairly penalize responses for reasonable elaboration.

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level (`GRANULAR` or `HOLISTIC`) |

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import AnswerConciseness
    from axion.dataset import DatasetItem

    metric = AnswerConciseness()

    item = DatasetItem(
        query="What is photosynthesis?",
        actual_output=(
            "Photosynthesis is, as you might already know, the process by which plants, "
            "and indeed many other organisms, use sunlight — solar energy — to synthesize "
            "nutrients from carbon dioxide and water, producing oxygen as a byproduct."
        ),
        expected_output=(
            "Photosynthesis is the process plants use to convert sunlight, CO2, and water "
            "into nutrients, releasing oxygen."
        ),
    )

    result = await metric.execute(item)
    print(result.pretty())
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import AnswerConciseness
    from axion.runners import MetricRunner

    metric = AnswerConciseness()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        for phrase in item_result.data.redundant_phrases:
            print(f"  - Redundant: {phrase}")
    ```

---

## Metric Diagnostics

Every evaluation is **fully interpretable**. Access detailed diagnostic results via `result.signals`.

```python
result = await metric.execute(item)
print(result.pretty())   # Human-readable summary
result.signals           # Full diagnostic breakdown
```

<details markdown="1">
<summary><strong>📊 AnswerConcisenessResult Structure</strong></summary>

```python
AnswerConcisenessResult(
{
    "overall_score": 0.6,
    "redundant_phrases": [
        "as you might already know",
        "and indeed many other organisms",
        "solar energy"
    ],
    "reason": "The response contains several filler phrases and restatements not present in the expected answer, increasing length without adding information."
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | `float` | The 0–1 conciseness score |
| `redundant_phrases` | `List[str]` | Phrases in the actual output not justified by the expected output |
| `reason` | `str` | Human-readable explanation for the score |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Concise Match (Score: 1.0)</strong></summary>

!!! success "No Redundancy"

    **Expected Output:**
    > "The capital of France is Paris."

    **Actual Output:**
    > "The capital of France is Paris."

    **Analysis:** No redundant phrases detected. Score: `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Moderate Verbosity (Score: 0.7)</strong></summary>

!!! warning "Minor Padding"

    **Expected Output:**
    > "Paris is the capital of France."

    **Actual Output:**
    > "Great question! Paris is, of course, the capital of France and has been for centuries."

    **Redundant phrases detected:** `"Great question!"`, `"of course"`, `"and has been for centuries"`

    **Final Score:** `0.7` :material-alert:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Excessive Verbosity (Score: 0.2)</strong></summary>

!!! failure "High Verbosity"

    **Expected Output:**
    > "Water boils at 100°C at sea level."

    **Actual Output:**
    > "That's a wonderful question about the physical properties of water. As you may or may not be aware, water — the molecule H₂O that we all depend on for life — boils at a temperature of 100 degrees Celsius, which is equivalent to 212 degrees Fahrenheit, when measured at standard sea-level atmospheric pressure, though of course this can vary at altitude."

    **Redundant phrases detected:** 7 distinct phrases flagged.

    **Final Score:** `0.2` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">✂️</span>
<strong>Response Quality</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Verbose responses erode user trust and reduce satisfaction. Concise answers signal confidence and mastery of the subject.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">💰</span>
<strong>Token Efficiency</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Unnecessary verbosity increases inference cost. Monitoring conciseness helps identify prompts that consistently produce bloated output.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Prompt Debugging</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Low conciseness scores often trace to system prompt artifacts (e.g., "be thorough and detailed") that conflict with user expectations for brevity.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Answer Conciseness** = Does the AI's response say what the expected answer says, without unnecessary extra words?

    - **Use it when:** You have a reference answer and want to penalize verbose output
    - **Score interpretation:** Higher = more economical; 0.8+ passes by default
    - **Key requirement:** `expected_output` is mandatory — this metric is reference-based

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.AnswerConciseness`](../../reference/metrics.md#answer-conciseness)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Answer Completeness](./answer_completeness.md) · [Answer Relevancy](./answer_relevancy.md) · [Faithfulness](./faithfulness.md)

</div>
