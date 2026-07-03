---
icon: custom/sliders
---
# Goal Completion

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Analyzes if the user's goal was achieved, tracking sub-goals and goal evolution</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #8b5cf6;">Multi-Turn</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Proportion of goals achieved</small>
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
    Goal Completion evaluates whether the **user's intent was ultimately fulfilled** across the conversation. It identifies the primary goal, any sub-goals, and tracks whether user intent evolved mid-conversation (goal evolution). Each identified goal is assessed for completion status, and the final score reflects the proportion of goals successfully addressed. This is the outcome-focused complement to structural metrics like Conversation Flow.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All goals — primary, sub-goals, evolved — fully completed |
    | **0.7+** | :material-check: Primary goal met; some sub-goals incomplete |
    | **0.5** | :material-alert: Threshold — primary goal partially completed or ambiguous |
    | **< 0.5** | :material-close: Primary goal not achieved; user intent unfulfilled |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Evaluating task-oriented agents where a clear outcome is expected</li>
<li>Measuring outcome success across multi-turn conversations at scale</li>
<li>Identifying conversations where user intent was not fulfilled by the agent</li>
<li>Benchmarking agents that handle complex, multi-step user requests</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Open-ended conversations with no clear goal (e.g., casual chat, brainstorming)</li>
<li>Evaluating response quality vs. outcome — use <a href="./conversation_flow.md">Conversation Flow</a> or <a href="../composite/faithfulness.md">Faithfulness</a></li>
<li>Single-turn Q&A — use <a href="../composite/answer_relevancy.md">Answer Relevancy</a> instead</li>
<li>Conversations where success is entirely subjective with no verifiable outcome</li>
</ul>
</div>

</div>

!!! tip "See Also: Conversation Flow"
    **Goal Completion** answers: *Was the user's objective achieved?*
    **[Conversation Flow](./conversation_flow.md)** answers: *Did the conversation progress well structurally?*

    A conversation can have excellent flow but fail to complete the goal (agent was polite but unhelpful), or complete the goal despite poor flow (correct answer buried in a disorganized exchange). Use both for complete multi-turn evaluation.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    An **Evaluator LLM** reads the full conversation, extracts the goal hierarchy, tracks any goal evolution, and assesses the completion status of each goal.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Input"]
            A["conversation<br/><small>List of {role, content} turns</small>"]
        end

        subgraph EXTRACT["🔍 Goal Extraction"]
            B["Primary Goal<br/><small>User's stated main intent</small>"]
            C["Sub-Goals<br/><small>Implicit or explicit secondary needs</small>"]
            D["Goal Evolution Tracking<br/><small>Did intent shift mid-conversation?</small>"]
        end

        subgraph ASSESS["⚖️ Completion Assessment"]
            E["Status per Goal<br/><small>COMPLETED / PARTIAL / NOT_COMPLETED</small>"]
        end

        subgraph SCORE["📊 Scoring"]
            F["Weighted Completion Ratio"]
            G["Clamp to [0, 1]"]
            H["Final Score"]
        end

        A --> B & C & D
        B & C & D --> E
        E --> F
        F --> G
        G --> H

        style INPUT stroke:#1E3A5F,stroke-width:2px
        style EXTRACT stroke:#3b82f6,stroke-width:2px
        style ASSESS stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style H fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-layers: Goal Types"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #1E3A5F; padding-left: 1rem;">
    <strong>🎯 Primary Goal</strong>
    <br><small>The main intent expressed or implied in the user's opening message. Weighted most heavily in the final score.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong>📌 Sub-Goals</strong>
    <br><small>Secondary needs identified during the conversation — explicitly stated ("also…") or implicit (e.g., needing confirmation after a booking).</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong>🔄 Evolved Goals</strong>
    <br><small>Goals that replaced or refined the original intent mid-conversation. The metric tracks the <em>final</em> stated intent as the target, not the original if the user corrected it.</small>
    </div>

    </div>

=== ":material-scale-balance: Completion Verdicts"

    Each identified goal receives a completion verdict:

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ COMPLETED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1.0</div>
    <br><small>Goal was fully addressed and resolved by the agent.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">⚠️ PARTIAL</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #f59e0b;">0.5</div>
    <br><small>Goal was partially addressed — relevant progress made but not fully resolved.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ NOT_COMPLETED</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0.0</div>
    <br><small>Goal was not addressed or the agent explicitly failed to resolve it.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = sum(goal_weight * verdict_value for each goal) / sum(goal_weights)
        ```
        Primary goal carries higher weight than sub-goals. Evolved goals replace their predecessor in the weight calculation.

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `primary_goal_weight` | `float` | `0.6` | Weight of the primary goal in the final score |
    | `sub_goal_weight` | `float` | `0.4` | Total weight distributed across sub-goals |
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level (`GRANULAR` or `HOLISTIC`) |

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import GoalCompletion
    from axion.dataset import DatasetItem

    metric = GoalCompletion()

    item = DatasetItem(
        conversation=[
            {"role": "user", "content": "I need to change my shipping address for order #4821."},
            {"role": "assistant", "content": "I can help! What's the new address?"},
            {"role": "user", "content": "123 Maple St, Portland, OR 97201."},
            {"role": "assistant", "content": "Updated! Your order #4821 will now ship to 123 Maple St, Portland, OR 97201."},
            {"role": "user", "content": "Also, can I get a tracking number?"},
            {"role": "assistant", "content": "Your tracking number is 1Z999AA10123456784."},
        ]
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Primary goal: address change → COMPLETED
    # Sub-goal: tracking number → COMPLETED
    # Score: 1.0
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import GoalCompletion
    from axion.runners import MetricRunner

    metric = GoalCompletion()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        print(f"Primary goal: {item_result.data.primary_goal.description}")
        print(f"  Status: {item_result.data.primary_goal.completion_status}")
        for sub in item_result.data.sub_goals:
            print(f"  Sub-goal: {sub.description} → {sub.completion_status}")
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
<summary><strong>📊 GoalCompletionResult Structure</strong></summary>

```python
GoalCompletionResult(
{
    "overall_score": 0.75,
    "primary_goal": {
        "description": "Change shipping address for order #4821",
        "completion_status": "COMPLETED",
        "verdict_value": 1.0
    },
    "sub_goals": [
        {
            "description": "Receive tracking number for the order",
            "completion_status": "NOT_COMPLETED",
            "verdict_value": 0.0
        }
    ],
    "goal_evolution": null,
    "reason": "The primary goal of updating the shipping address was completed successfully. The sub-goal of obtaining a tracking number was not addressed by the agent."
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | `float` | Weighted completion ratio (0–1) |
| `primary_goal` | `GoalRecord` | Primary goal with description, status, and verdict value |
| `sub_goals` | `List[GoalRecord]` | Sub-goals with descriptions, statuses, and verdict values |
| `goal_evolution` | `GoalEvolution \| null` | Records original goal and evolved goal if intent shifted |
| `reason` | `str` | Human-readable summary of what was and was not completed |

### GoalRecord Fields

| Field | Type | Description |
|-------|------|-------------|
| `description` | `str` | Natural-language description of the goal |
| `completion_status` | `str` | `COMPLETED`, `PARTIAL`, or `NOT_COMPLETED` |
| `verdict_value` | `float` | Numeric value: 1.0, 0.5, or 0.0 |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: All Goals Completed (Score: 1.0)</strong></summary>

!!! success "Full Completion"

    **Primary goal:** Book a table for two at 7 PM → `COMPLETED`
    **Sub-goal:** Receive confirmation email → `COMPLETED`

    **Final Score:** `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Goal Evolution (Score: 0.8)</strong></summary>

!!! warning "Evolved Goal Tracked"

    **Original goal:** "Refund order #881"
    **Evolved goal:** User clarified mid-conversation: "Actually, just exchange it for size L"

    **Primary goal (evolved):** Exchange for size L → `COMPLETED`
    **Sub-goal:** Receive exchange confirmation → `PARTIAL`

    The metric correctly scores against the *evolved* goal (exchange), not the original (refund).

    **Final Score:** `0.8` :material-check:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Primary Goal Not Met (Score: 0.2)</strong></summary>

!!! failure "Goal Not Completed"

    **Primary goal:** Cancel subscription → `NOT_COMPLETED` (agent redirected to retention flow without resolving)
    **Sub-goal:** Receive cancellation confirmation → `NOT_COMPLETED`

    **Final Score:** `0.2` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">✅</span>
<strong>Outcome Measurement</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Structural metrics tell you how a conversation felt; Goal Completion tells you whether it actually delivered value. Both are necessary for a complete picture.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔄</span>
<strong>Goal Evolution Awareness</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Real users change their minds. Evaluating against a fixed initial goal penalizes agents that correctly adapt. Goal Completion tracks evolved intent and scores fairly against the final objective.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📊</span>
<strong>Sub-Goal Coverage</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Users often have implicit needs alongside their stated request. Tracking sub-goals surfaces partial failures where the main ask was handled but adjacent needs were missed.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Goal Completion** = Did the agent actually accomplish what the user came to do — including sub-goals and mid-conversation intent shifts?

    - **Use it when:** Evaluating task-oriented agents where user success is the primary metric
    - **Score interpretation:** Higher = more goals fully resolved; 0.5 passes by default
    - **Key strength:** Handles goal evolution — scores against final intent, not initial phrasing

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.GoalCompletion`](../../reference/metrics.md#goal-completion)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Conversation Flow](./conversation_flow.md) · [Conversation Efficiency](./conversation_efficiency.md) · [Persona & Tone Adherence](./persona_tone_adherence.md)

</div>
