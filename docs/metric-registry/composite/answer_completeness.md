# Answer Completeness

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Measure how completely the response covers expected content</strong><br>
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
<small style="color: var(--md-text-muted);">Coverage ratio</small>
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
<code>query</code> <code>actual_output</code> <code>expected_output</code><br>
<small style="color: var(--md-text-muted);">Reference answer required</small>
</div>

</div>

!!! abstract "What It Measures"
    Answer Completeness evaluates whether the response covers all the **key aspects** from the expected output. It answers: "Did the AI mention everything important from the reference answer?"

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All aspects from expected output covered |
    | **0.7+** | :material-check: Most aspects covered, minor omissions |
    | **0.5** | :material-alert: Half the expected content covered |
    | **< 0.5** | :material-close: Significant content missing |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>You have reference answers</li>
<li>Completeness matters more than brevity</li>
<li>Testing comprehensive responses</li>
<li>Evaluating educational content</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Brevity is preferred</li>
<li>Multiple valid answer formats</li>
<li>No expected_output available</li>
<li>Creative/generative tasks</li>
</ul>
</div>

</div>

!!! tip "See Also: Answer Criteria"
    **Answer Completeness** checks coverage of *expected output* aspects.
    **[Answer Criteria](./answer_criteria.md)** checks coverage of *custom acceptance criteria*.

    Use Completeness when you have a reference answer; use Criteria for custom requirements.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric extracts key aspects from the expected output and checks if each is covered in the actual response.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[AI Response]
            C[Expected Output]
        end

        subgraph EXTRACT["🔍 Step 1: Aspect Extraction"]
            D[Extract Aspects from Expected]
            E["Key Aspects List"]
        end

        subgraph CHECK["⚖️ Step 2: Coverage Check"]
            F[Check Each Aspect in Response]
            G["Covered / Not Covered"]
        end

        subgraph SCORE["📊 Step 3: Scoring"]
            H["Count Covered Aspects"]
            I["Calculate Ratio"]
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
        style CHECK stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style J fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Scoring System"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ COVERED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1</div>
    <br><small>Aspect from expected output is <strong>present</strong> in the response.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ NOT COVERED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0</div>
    <br><small>Aspect from expected output is <strong>missing</strong> from the response.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = covered_aspects / total_aspects
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `use_expected_output` | `bool` | `True` | Use expected_output for aspect extraction |
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level |

    !!! info "Alternative Mode"
        When `use_expected_output=False`, the metric uses sub-question decomposition instead of aspect extraction.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import AnswerCompleteness
    from axion.dataset import DatasetItem

    metric = AnswerCompleteness()

    item = DatasetItem(
        query="What are the benefits of exercise?",
        actual_output="Exercise improves cardiovascular health and boosts mood.",
        expected_output="Exercise improves cardiovascular health, strengthens muscles, boosts mood, and helps with weight management.",
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 0.5 (2 of 4 aspects covered)
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import AnswerCompleteness
    from axion.runners import MetricRunner

    metric = AnswerCompleteness()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        print(f"Covered: {item_result.signals.covered_aspects_count}/{item_result.signals.total_aspects_count}")
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
<summary><strong>📊 AnswerCompletenessResult Structure</strong></summary>

```python
AnswerCompletenessResult(
{
    "score": 0.5,
    "covered_aspects_count": 2,
    "total_aspects_count": 4,
    "concept_coverage_score": 0.5,
    "aspect_breakdown": [
        {
            "aspect": "cardiovascular health improvement",
            "covered": true,
            "concepts_covered": ["cardiovascular health"],
            "reason": "Mentioned in response"
        },
        {
            "aspect": "muscle strengthening",
            "covered": false,
            "concepts_missing": ["muscles", "strength"],
            "reason": "Not mentioned in response"
        },
        {
            "aspect": "mood improvement",
            "covered": true,
            "concepts_covered": ["mood", "boosts"],
            "reason": "Mentioned in response"
        },
        {
            "aspect": "weight management",
            "covered": false,
            "concepts_missing": ["weight"],
            "reason": "Not mentioned in response"
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Overall completeness score |
| `covered_aspects_count` | `int` | Aspects found in response |
| `total_aspects_count` | `int` | Total aspects from expected output |
| `aspect_breakdown` | `List` | Per-aspect coverage details |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Complete Coverage (Score: 1.0)</strong></summary>

!!! success "All Aspects Covered"

    **Expected Output:**
    > "Python is a high-level programming language known for readability, extensive libraries, and cross-platform support."

    **AI Response:**
    > "Python is a high-level language with clean, readable syntax. It has a vast ecosystem of libraries and runs on Windows, Mac, and Linux."

    **Analysis:**

    | Aspect | Covered |
    |--------|---------|
    | High-level language | ✅ |
    | Readability | ✅ |
    | Extensive libraries | ✅ |
    | Cross-platform | ✅ |

    **Final Score:** `4 / 4 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Partial Coverage (Score: 0.6)</strong></summary>

!!! warning "Some Aspects Missing"

    **Expected Output:**
    > "Our product offers: free shipping, 30-day returns, 24/7 support, price matching, and warranty."

    **AI Response:**
    > "We provide free shipping on all orders and a 30-day return policy. Our support team is available around the clock."

    **Analysis:**

    | Aspect | Covered |
    |--------|---------|
    | Free shipping | ✅ |
    | 30-day returns | ✅ |
    | 24/7 support | ✅ |
    | Price matching | ❌ |
    | Warranty | ❌ |

    **Final Score:** `3 / 5 = 0.6` :material-alert:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Poor Coverage (Score: 0.25)</strong></summary>

!!! failure "Most Aspects Missing"

    **Expected Output:**
    > "The recipe requires flour, sugar, eggs, and butter. Preheat oven to 350°F. Mix ingredients, pour into pan, bake 25 minutes."

    **AI Response:**
    > "You'll need flour and sugar."

    **Analysis:**

    | Aspect | Covered |
    |--------|---------|
    | Flour | ✅ |
    | Sugar | ✅ |
    | Eggs | ❌ |
    | Butter | ❌ |
    | Oven temperature | ❌ |
    | Mixing instructions | ❌ |
    | Baking time | ❌ |

    **Final Score:** `2 / 7 = 0.29` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">📝</span>
<strong>Content Coverage</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensures AI responses include all important information, not just some of it.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🎓</span>
<strong>Educational Quality</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Critical for tutoring systems where incomplete answers leave knowledge gaps.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📋</span>
<strong>Requirements Coverage</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Verify that responses address all parts of complex queries.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Answer Completeness** = Does the response cover all aspects from the expected answer?

    - **Use it when:** You have reference answers and need comprehensive coverage
    - **Score interpretation:** Higher = more aspects from expected output covered
    - **Key difference:** Measures coverage, not accuracy

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.AnswerCompleteness`](../../reference/metrics.md#answer-completeness)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Answer Criteria](./answer_criteria.md) · Factual Accuracy · Answer Relevancy

</div>
