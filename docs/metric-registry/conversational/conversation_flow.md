---
icon: custom/sliders
---
# Conversation Flow

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Modular conversation flow analysis with coherence, efficiency, and issue detection</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #8b5cf6;">Multi-Turn</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Composite weighted average</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.7</code><br>
<small style="color: var(--md-text-muted);">Pass/fail cutoff</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>conversation</code><br>
<small style="color: var(--md-text-muted);">List of <code>{role, content}</code> turns</small>
</div>

</div>

!!! abstract "What It Measures"
    Conversation Flow evaluates the **quality of multi-turn dialogue** across three modular dimensions: coherence, efficiency, and issue detection. Rather than measuring whether a specific goal was achieved, it assesses the structural and logical quality of the conversation itself — how well each turn connects to the last, how smoothly the dialogue progresses, and whether common failure modes (drift, repetition, dead ends) are present.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Excellent flow — coherent, efficient, no issues detected |
    | **0.7+** | :material-check: Good flow; minor inefficiencies or isolated issues |
    | **0.5** | :material-alert: Threshold — noticeable flow problems impacting experience |
    | **< 0.5** | :material-close: Poor flow — repeated breakdowns, drift, or dead ends |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Evaluating multi-turn agent conversations holistically</li>
<li>Detecting where conversations structurally break down</li>
<li>Agent dialogue quality assessment and benchmarking</li>
<li>Comparing prompt versions for conversational smoothness</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Single-turn Q&A — use <a href="../composite/answer_relevancy.md">Answer Relevancy</a> or <a href="../composite/faithfulness.md">Faithfulness</a></li>
<li>Evaluating factual accuracy of responses</li>
<li>Evaluating goal achievement specifically — use <a href="./goal_completion.md">Goal Completion</a></li>
<li>Conversations with fewer than 3 turns</li>
</ul>
</div>

</div>

!!! tip "See Also: Goal Completion & Conversation Efficiency"
    **Conversation Flow** evaluates *structural and dialogic quality* across coherence, efficiency, and issues.
    **[Goal Completion](./goal_completion.md)** evaluates *whether the user's objective was achieved*.
    **[Conversation Efficiency](./conversation_efficiency.md)** focuses *specifically on turn economy*.

    Use all three together for comprehensive multi-turn evaluation.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric applies a **modular LLM evaluation** across three independent dimensions. Each dimension is scored separately, then combined via a weighted average to produce the final composite score.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Input"]
            A["conversation<br/><small>List of {role, content} turns</small>"]
        end

        subgraph DIM["🔍 Dimension Scoring"]
            B["Coherence Judge<br/><small>Logical progression, context maintenance</small>"]
            C["Efficiency Judge<br/><small>Turn count vs. goal complexity</small>"]
            D["Issue Detector<br/><small>Drift, repetition, dead ends, clarification failures</small>"]
        end

        subgraph SCORE["📊 Composite Score"]
            E["Weighted Average"]
            F["Clamp to [0, 1]"]
            G["Final Score"]
        end

        A --> B & C & D
        B & C & D --> E
        E --> F
        F --> G

        style INPUT stroke:#1E3A5F,stroke-width:2px
        style DIM stroke:#3b82f6,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style G fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-layers: Dimensions"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">🧩 Coherence</strong>
    <br><small>Evaluates logical progression between turns and whether the agent maintains context across the conversation. Penalizes non-sequiturs, ignored user statements, and context amnesia.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">⚡ Efficiency</strong>
    <br><small>Compares the number of turns used against the complexity of the conversational goal. Penalizes unnecessary back-and-forth that a competent agent should have resolved in fewer exchanges.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">🚨 Issue Detection</strong>
    <br><small>Flags specific failure patterns: topic drift, repetition of prior content, dead-end responses that stall progress, and clarification failures where the agent misunderstands the user repeatedly.</small>
    </div>

    </div>

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `coherence_weight` | `float` | `0.4` | Weight applied to the coherence dimension score |
    | `efficiency_weight` | `float` | `0.3` | Weight applied to the efficiency dimension score |
    | `issue_weight` | `float` | `0.3` | Weight applied to the issue detection dimension score |
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level (`GRANULAR` or `HOLISTIC`) |

    !!! info "Weight Customization"
        Weights must sum to `1.0`. Adjust them to emphasize the dimension most important to your use case — e.g., increase `coherence_weight` for narrative agents or `issue_weight` for customer service bots where repetition is particularly damaging.

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ConversationFlow
    from axion.dataset import DatasetItem

    metric = ConversationFlow()

    item = DatasetItem(
        conversation=[
            {"role": "user", "content": "I need to cancel my subscription."},
            {"role": "assistant", "content": "I can help with that. Can you confirm your account email?"},
            {"role": "user", "content": "it's jane@example.com"},
            {"role": "assistant", "content": "Thank you, Jane. Your subscription has been cancelled. You'll receive a confirmation email shortly."},
        ]
    )

    result = await metric.execute(item)
    print(result.pretty())
    ```

=== ":material-cog-outline: With Custom Weights"

    ```python
    from axion.metrics import ConversationFlow

    # Emphasize coherence for a narrative agent
    metric = ConversationFlow(
        coherence_weight=0.6,
        efficiency_weight=0.2,
        issue_weight=0.2,
    )
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ConversationFlow
    from axion.runners import MetricRunner

    metric = ConversationFlow()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        print(f"Coherence: {item_result.data.coherence_score}")
        print(f"Efficiency: {item_result.data.efficiency_score}")
        print(f"Issues: {item_result.data.issues_detected}")
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
<summary><strong>📊 ConversationFlowResult Structure</strong></summary>

```python
ConversationFlowResult(
{
    "overall_score": 0.78,
    "coherence_score": 0.9,
    "efficiency_score": 0.75,
    "issue_score": 0.65,
    "issues_detected": [
        {
            "type": "repetition",
            "turn": 4,
            "description": "Agent restated the cancellation policy already covered in turn 2."
        }
    ],
    "reason": "Conversation was largely coherent and resolved efficiently. One instance of repetition noted in turn 4."
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | `float` | Weighted composite score (0–1) |
| `coherence_score` | `float` | Score for logical progression and context maintenance |
| `efficiency_score` | `float` | Score for turn economy relative to goal complexity |
| `issue_score` | `float` | Score reflecting absence of flow issues (1.0 = no issues) |
| `issues_detected` | `List` | Individual issue records with type, turn, and description |
| `reason` | `str` | Human-readable summary of the evaluation |

### Issue Types

| Type | Description |
|------|-------------|
| `topic_drift` | Conversation shifted away from the user's stated topic |
| `repetition` | Agent repeated content already communicated earlier |
| `dead_end` | Agent response stalled progress without actionable next step |
| `clarification_failure` | Agent failed to resolve an ambiguity after multiple attempts |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Excellent Flow (Score: 0.95)</strong></summary>

!!! success "Coherent, Efficient, No Issues"

    **Conversation:**

    | Turn | Role | Content |
    |------|------|---------|
    | 1 | User | "What's my current plan?" |
    | 2 | Agent | "You're on the Pro plan, billed monthly at $29." |
    | 3 | User | "Can I switch to annual?" |
    | 4 | Agent | "Yes — switching to annual saves you $58/year. Want me to switch you now?" |
    | 5 | User | "Yes please." |
    | 6 | Agent | "Done! You're now on the Annual Pro plan." |

    **Analysis:** Perfect context carry-through, goal resolved in minimal turns, no issues detected.

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Moderate Issues (Score: 0.55)</strong></summary>

!!! warning "Repetition + Topic Drift"

    **Issues detected:**
    - Turn 3: Agent re-explained the refund policy already stated in turn 1 (`repetition`)
    - Turn 5: Agent pivoted to discuss a new feature not relevant to the user's question (`topic_drift`)

    **Final Score:** `0.55` :material-alert:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🔗</span>
<strong>End-to-End Quality</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Individual turn quality metrics miss how a conversation hangs together. Flow evaluates the full arc, catching degradation patterns that only emerge across turns.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🐛</span>
<strong>Root-Cause Diagnosis</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Dimension-level scores pinpoint exactly where flow breaks: context amnesia (coherence), unnecessary turns (efficiency), or specific anti-patterns (issues).</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📈</span>
<strong>Regression Detection</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Track flow scores across prompt versions to catch regressions before deployment — a new system prompt may improve task completion but introduce repetition.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Conversation Flow** = Does the multi-turn conversation progress logically, efficiently, and without common failure patterns?

    - **Use it when:** You need holistic quality assessment of multi-turn agent dialogue
    - **Score interpretation:** Higher = better structural flow across all three dimensions
    - **Key config:** Adjust dimension weights to match your evaluation priorities

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ConversationFlow`](../../reference/metrics.md#conversation-flow)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Goal Completion](./goal_completion.md) · [Conversation Efficiency](./conversation_efficiency.md) · [Persona & Tone Adherence](./persona_tone_adherence.md)

</div>
