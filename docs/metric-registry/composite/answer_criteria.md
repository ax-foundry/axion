# Answer Criteria

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluate responses against user-defined acceptance criteria</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Knowledge</span>
<span class="badge" style="background: #6B7A3A;">Single Turn</span>
<span class="badge" style="background: #8b5cf6;">Multi-Turn</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Criteria coverage ratio</small>
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
<code>query</code> <code>actual_output</code><br>
<small style="color: var(--md-text-muted);">Optional: <code>acceptance_criteria</code></small>
</div>

</div>

!!! abstract "What It Measures"
    Answer Criteria evaluates whether a response meets **user-defined acceptance criteria**. It decomposes criteria into aspects and concepts, then checks coverage. This is ideal for custom evaluation requirements that don't fit standard metrics.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All criteria aspects fully covered |
    | **0.7+** | :material-check: Most criteria met, minor gaps |
    | **0.5** | :material-alert: Half the criteria covered |
    | **< 0.5** | :material-close: Significant criteria not met |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Custom acceptance criteria exist</li>
<li>Domain-specific requirements</li>
<li>Multi-aspect evaluation needed</li>
<li>Testing conversational agents</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Standard metrics suffice</li>
<li>No clear acceptance criteria</li>
<li>Purely factual evaluation</li>
<li>Simple pass/fail needed</li>
</ul>
</div>

</div>

!!! tip "See Also: Answer Completeness"
    **Answer Criteria** evaluates against *custom acceptance criteria*.
    **[Answer Completeness](./answer_completeness.md)** evaluates against *expected output* aspects.

    Use Criteria for custom requirements; use Completeness when you have a reference answer.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric decomposes acceptance criteria into aspects, identifies key concepts per aspect, then checks if the response covers them.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[AI Response]
            C[Acceptance Criteria]
        end

        subgraph DECOMPOSE["🔍 Step 1: Criteria Decomposition"]
            D[Extract Aspects]
            E["Aspects with Key Concepts"]
        end

        subgraph EVALUATE["⚖️ Step 2: Coverage Check"]
            F[Check Each Aspect]
            G["Covered / Missing Concepts"]
        end

        subgraph SCORE["📊 Step 3: Scoring"]
            H["Apply Scoring Strategy"]
            I["Final Score"]
        end

        A & B & C --> D
        D --> E
        E --> F
        B --> F
        F --> G
        G --> H
        H --> I

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style DECOMPOSE stroke:#3b82f6,stroke-width:2px
        style EVALUATE stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style I fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Scoring Strategies"

    Choose how to calculate the final score based on aspect and concept coverage.

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">📊 CONCEPT</strong>
    <br><small>Score = total_concepts_covered / total_concepts<br><strong>Default.</strong> Granular concept-level coverage.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">📋 ASPECT</strong>
    <br><small>Score = covered_aspects / total_aspects<br>Binary per-aspect (all-or-nothing).</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #8b5cf6; padding-left: 1rem;">
    <strong style="color: #8b5cf6;">⚖️ WEIGHTED</strong>
    <br><small>Score = 0.7 × concept_score + 0.3 × aspect_score<br>Blend of both approaches.</small>
    </div>

    </div>

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `criteria_key` | `str` | `'Complete'` | Key to look up criteria |
    | `scoring_strategy` | `'concept'` \| `'aspect'` \| `'weighted'` | `'concept'` | How to calculate score |
    | `check_for_contradictions` | `bool` | `False` | Check if response contradicts criteria |
    | `weighted_concept_score_weight` | `float` | `0.7` | Weight for concept score in weighted strategy |
    | `multi_turn_strategy` | `'last_turn'` \| `'all_turns'` | `'last_turn'` | How to evaluate conversations |
    | `multi_turn_aggregation` | `'cumulative'` \| `'average'` | `'cumulative'` | How to aggregate multi-turn scores |

=== ":material-tune-variant: Scoring Strategies"

    ```python
    from axion.metrics import AnswerCriteria

    # Concept-level (default, most granular)
    metric = AnswerCriteria(scoring_strategy='concept')

    # Aspect-level (binary per aspect)
    metric = AnswerCriteria(scoring_strategy='aspect')

    # Weighted blend
    metric = AnswerCriteria(
        scoring_strategy='weighted',
        weighted_concept_score_weight=0.7  # 70% concept, 30% aspect
    )
    ```

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import AnswerCriteria
    from axion.dataset import DatasetItem

    metric = AnswerCriteria()

    item = DatasetItem(
        query="Explain how to make a good cup of coffee",
        actual_output="Use fresh beans, grind just before brewing, use water at 200°F, and brew for 4 minutes.",
        acceptance_criteria="Must mention: bean freshness, grind timing, water temperature, brew time",
    )

    result = await metric.execute(item)
    print(result.pretty())
    ```

=== ":material-format-list-checks: Custom Criteria"

    ```python
    from axion.metrics import AnswerCriteria

    # Strict aspect-level scoring
    metric = AnswerCriteria(
        scoring_strategy='aspect',
        check_for_contradictions=True
    )

    item = DatasetItem(
        query="What's your return policy?",
        actual_output="You can return items within 30 days with receipt.",
        acceptance_criteria="""
        Must cover:
        1. Return window (30 days)
        2. Receipt requirement
        3. Condition of items
        4. Refund method
        """,
    )
    ```

=== ":material-chat-processing: Multi-Turn"

    ```python
    from axion.metrics import AnswerCriteria

    metric = AnswerCriteria(
        multi_turn_strategy='all_turns',
        multi_turn_aggregation='cumulative'  # Criteria can be met across turns
    )
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
<summary><strong>📊 AnswerCriteriaResult Structure</strong></summary>

```python
AnswerCriteriaResult(
{
    "scoring_strategy": "concept",
    "covered_aspects_count": 3,
    "total_aspects_count": 4,
    "total_concepts_covered": 5,
    "total_concepts": 7,
    "concept_coverage_score": 0.71,
    "aspect_breakdown": [
        {
            "aspect": "Bean freshness",
            "covered": true,
            "concepts_covered": ["fresh beans", "quality"],
            "concepts_missing": [],
            "reason": "Response mentions using fresh beans"
        },
        {
            "aspect": "Water temperature",
            "covered": true,
            "concepts_covered": ["200°F"],
            "concepts_missing": ["optimal range"],
            "reason": "Specific temperature provided"
        }
    ],
    "evaluated_turns_count": 1
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `scoring_strategy` | `str` | Strategy used (concept/aspect/weighted) |
| `covered_aspects_count` | `int` | Aspects fully covered |
| `total_aspects_count` | `int` | Total aspects in criteria |
| `total_concepts_covered` | `int` | Concepts found in response |
| `total_concepts` | `int` | Total concepts across all aspects |
| `concept_coverage_score` | `float` | Concept-level coverage ratio |
| `aspect_breakdown` | `List` | Per-aspect coverage details |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Full Coverage (Score: 1.0)</strong></summary>

!!! success "All Criteria Met"

    **Criteria:**
    > "Must mention: greeting, issue acknowledgment, solution, follow-up offer"

    **AI Response:**
    > "Hello! I understand you're having trouble with your order. I've issued a full refund which will appear in 3-5 days. Is there anything else I can help with?"

    **Analysis:**

    | Aspect | Covered | Concepts |
    |--------|---------|----------|
    | Greeting | ✅ | "Hello" |
    | Issue acknowledgment | ✅ | "trouble with your order" |
    | Solution | ✅ | "full refund", "3-5 days" |
    | Follow-up offer | ✅ | "anything else I can help" |

    **Final Score:** `4 / 4 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Partial Coverage (Score: 0.5)</strong></summary>

!!! warning "Some Criteria Missing"

    **Criteria:**
    > "Must include: product name, price, availability, shipping info"

    **AI Response:**
    > "The Widget Pro costs $49.99 and is currently in stock."

    **Analysis:**

    | Aspect | Covered | Concepts |
    |--------|---------|----------|
    | Product name | ✅ | "Widget Pro" |
    | Price | ✅ | "$49.99" |
    | Availability | ✅ | "in stock" |
    | Shipping info | ❌ | *missing* |

    **Final Score (aspect):** `3 / 4 = 0.75`

    *No shipping information provided.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Poor Coverage (Score: 0.25)</strong></summary>

!!! failure "Most Criteria Not Met"

    **Criteria:**
    > "Must cover: apology, explanation, compensation, prevention steps"

    **AI Response:**
    > "We apologize for the inconvenience."

    **Analysis:**

    | Aspect | Covered | Concepts |
    |--------|---------|----------|
    | Apology | ✅ | "apologize" |
    | Explanation | ❌ | *missing* |
    | Compensation | ❌ | *missing* |
    | Prevention steps | ❌ | *missing* |

    **Final Score:** `1 / 4 = 0.25` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>Custom Requirements</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Define exactly what a good response looks like for your specific use case.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📋</span>
<strong>Policy Compliance</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensure AI responses follow company guidelines, scripts, or regulatory requirements.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">💬</span>
<strong>Agent Quality</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Evaluate customer service agents against expected response patterns.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Answer Criteria** = Does the response meet your custom acceptance criteria?

    - **Use it when:** You have specific requirements beyond standard metrics
    - **Score interpretation:** Higher = more criteria aspects covered
    - **Key config:** Choose `scoring_strategy` based on granularity needs

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.AnswerCriteria`](../../reference/metrics.md#answer-criteria)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Answer Completeness](./answer_completeness.md) · Answer Relevancy · Factual Accuracy

</div>
