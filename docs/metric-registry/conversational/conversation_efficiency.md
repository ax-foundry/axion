---
icon: custom/sliders
---
# Conversation Efficiency

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluates whether the conversation achieved its goal through the most efficient path</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #8b5cf6;">Multi-Turn</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">1.0 = optimally efficient path</small>
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
<code>conversation</code><br>
<small style="color: var(--md-text-muted);">List of <code>{role, content}</code> turns</small>
</div>

</div>

!!! abstract "What It Measures"
    Conversation Efficiency evaluates whether the agent reached the conversational outcome via the **most direct path** available. It penalizes unnecessary clarification loops (asking for information already provided), repeated questions, excessive hedging, and digressive turns that do not advance the conversation toward its goal. A score of 1.0 means every turn was purposeful and load-bearing.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Every turn was necessary — optimal path taken |
    | **0.7+** | :material-check: Minor inefficiencies; one or two unnecessary exchanges |
    | **0.5** | :material-alert: Threshold — noticeable turn waste impacting experience |
    | **< 0.5** | :material-close: Significant looping, repetition, or digression detected |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Optimizing agent dialogue for high-volume customer interactions where turn count matters</li>
<li>Identifying verbose or looping agents that frustrate users with unnecessary back-and-forth</li>
<li>Benchmarking prompt versions on turn economy</li>
<li>Measuring efficiency regressions after model or prompt changes</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Creative or exploratory conversations where depth and digression are desirable</li>
<li>Single-turn Q&A — this metric requires a multi-turn <code>conversation</code></li>
<li>Evaluating whether the correct answer was given — use <a href="../composite/faithfulness.md">Faithfulness</a></li>
<li>Evaluating whether the user's goal was achieved — use <a href="./goal_completion.md">Goal Completion</a></li>
</ul>
</div>

</div>

!!! tip "See Also: Conversation Flow"
    **Conversation Efficiency** focuses narrowly on *turn economy* — was the goal reached with minimal back-and-forth?
    **[Conversation Flow](./conversation_flow.md)** is broader — it also evaluates coherence and detects issue patterns like topic drift and dead ends.

    Use Efficiency when you want a sharp signal on turn waste; use Flow when you want holistic dialogue quality.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    An **Evaluator LLM** reads the full conversation and estimates the minimum number of turns a competent agent would have needed to achieve the same outcome. It then compares this to the actual turn count and identifies specific wasteful exchanges.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Input"]
            A["conversation<br/><small>List of {role, content} turns</small>"]
        end

        subgraph ANALYZE["🔍 Analysis"]
            B["Goal Identification"]
            C["Minimum Path Estimation<br/><small>What would a competent agent need?</small>"]
            D["Wasteful Turn Detection<br/><small>Loops, repetition, hedging, digression</small>"]
        end

        subgraph SCORE["📊 Scoring"]
            E["Efficiency Ratio"]
            F["Clamp to [0, 1]"]
            G["Final Score"]
        end

        A --> B --> C & D
        C & D --> E
        E --> F
        F --> G

        style INPUT stroke:#1E3A5F,stroke-width:2px
        style ANALYZE stroke:#3b82f6,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style G fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-list-box: Inefficiency Patterns"

    The LLM flags these specific turn-waste patterns:

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">🔁 Unnecessary Clarification Loops</strong>
    <br><small>Asking for information the user already provided, or asking multiple clarifying questions sequentially when they could be batched.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">🔄 Repeated Questions</strong>
    <br><small>Re-asking questions already answered by the user, indicating context-tracking failure.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #6b7280; padding-left: 1rem;">
    <strong style="color: #6b7280;">🛡️ Excessive Hedging</strong>
    <br><small>Multi-turn disclaimer cycles or repeated caveats that delay delivering the actual answer.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #8b5cf6; padding-left: 1rem;">
    <strong style="color: #8b5cf6;">↗️ Digressive Turns</strong>
    <br><small>Agent turns that do not advance the conversation toward its goal — tangents, unsolicited information, off-topic elaboration.</small>
    </div>

    </div>

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level (`GRANULAR` or `HOLISTIC`) |

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ConversationEfficiency
    from axion.dataset import DatasetItem

    metric = ConversationEfficiency()

    item = DatasetItem(
        conversation=[
            {"role": "user", "content": "I want to reset my password. My email is jane@example.com."},
            {"role": "assistant", "content": "Sure! What's your email address?"},
            {"role": "user", "content": "It's jane@example.com, as I said."},
            {"role": "assistant", "content": "Got it. I've sent a reset link to jane@example.com."},
        ]
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score will be penalized: agent asked for email already provided in turn 1
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ConversationEfficiency
    from axion.runners import MetricRunner

    metric = ConversationEfficiency()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        for waste in item_result.data.wasteful_turns:
            print(f"  - Turn {waste.turn}: {waste.pattern} — {waste.description}")
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
<summary><strong>📊 ConversationEfficiencyResult Structure</strong></summary>

```python
ConversationEfficiencyResult(
{
    "overall_score": 0.5,
    "actual_turns": 4,
    "estimated_minimum_turns": 2,
    "wasteful_turns": [
        {
            "turn": 2,
            "pattern": "unnecessary_clarification",
            "description": "Agent asked for the user's email address, which was already provided in turn 1."
        }
    ],
    "reason": "The agent failed to use information provided upfront, resulting in a redundant clarification round-trip."
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | `float` | The 0–1 efficiency score |
| `actual_turns` | `int` | Total number of turns in the conversation |
| `estimated_minimum_turns` | `int` | LLM estimate of minimum turns needed |
| `wasteful_turns` | `List` | Individual wasteful turn records |
| `reason` | `str` | Human-readable summary |

### Wasteful Turn Fields

| Field | Type | Description |
|-------|------|-------------|
| `turn` | `int` | Turn index (1-based) where waste occurred |
| `pattern` | `str` | Pattern type: `unnecessary_clarification`, `repeated_question`, `excessive_hedging`, `digression` |
| `description` | `str` | Explanation of the specific inefficiency |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Optimal Efficiency (Score: 1.0)</strong></summary>

!!! success "Every Turn Purposeful"

    **Conversation:**

    | Turn | Role | Content |
    |------|------|---------|
    | 1 | User | "Book me a flight to NYC next Friday." |
    | 2 | Agent | "Found 3 options for next Friday. Do you prefer morning, afternoon, or evening?" |
    | 3 | User | "Morning." |
    | 4 | Agent | "Booked: UA 472, departs 8:15 AM. Confirmation sent to your email." |

    **Analysis:** Each turn was necessary. The agent batched all needed info gathering into one question. Score: `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Clarification Loop (Score: 0.4)</strong></summary>

!!! warning "Unnecessary Clarifications"

    **Wasteful turns detected:**
    - Turn 2: Asked for account ID already given in turn 1 (`unnecessary_clarification`)
    - Turn 4: Asked for the same issue description user provided in turn 1 (`repeated_question`)
    - Turn 6: Added three-turn disclaimer cycle before answering (`excessive_hedging`)

    **Final Score:** `0.4` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">⏱️</span>
<strong>User Experience</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Users abandon conversations that feel circular. Efficient agents resolve intent faster, producing higher satisfaction scores and lower drop-off rates.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">💰</span>
<strong>Cost Control</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Unnecessary turns inflate token usage and per-conversation cost. Tracking efficiency identifies prompt patterns that reliably produce lean, purposeful dialogue.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Context Tracking Signal</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Low efficiency often signals a context-tracking failure — the agent isn't retaining user-provided information across turns. This metric surfaces those failures precisely.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Conversation Efficiency** = Did the agent reach the outcome with the minimum necessary turns, without loops, repetition, or digression?

    - **Use it when:** Turn count and directness matter — especially high-volume customer-facing agents
    - **Score interpretation:** Higher = more purposeful turns; lower = detectable waste
    - **Key requirement:** Multi-turn `conversation` — not applicable to single-turn Q&A

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ConversationEfficiency`](../../reference/metrics.md#conversation-efficiency)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Conversation Flow](./conversation_flow.md) · [Goal Completion](./goal_completion.md) · [Persona & Tone Adherence](./persona_tone_adherence.md)

</div>
