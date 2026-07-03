---
icon: custom/sliders
---
# Persona & Tone Adherence

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluates whether the agent maintains its intended persona and tone consistently</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #8b5cf6;">Multi-Turn</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">1.0 = fully consistent persona</small>
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
<small style="color: var(--md-text-muted);">Optional: <code>additional_input</code> (persona spec)</small>
</div>

</div>

!!! abstract "What It Measures"
    Persona & Tone Adherence evaluates whether the agent **maintains its intended persona and tone consistently** throughout a conversation. It tracks tone consistency, formality level, language style alignment, and persona drift — the gradual (or sudden) departure from a specified character. When a persona specification is provided via `additional_input`, the metric evaluates against that explicit contract; without it, the metric infers the intended persona from early turns and assesses whether the agent held it.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Persona maintained perfectly — tone consistent across all turns |
    | **0.7+** | :material-check: Minor drift; overall character well-preserved |
    | **0.5** | :material-alert: Threshold — noticeable inconsistencies in tone or persona |
    | **< 0.5** | :material-close: Significant drift — agent's voice is inconsistent or misaligned with spec |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Branded chatbots or agents with a defined persona (name, voice, character)</li>
<li>Checking for tone drift across long conversations</li>
<li>Customer service agents where consistent tone is a brand requirement</li>
<li>Validating that a system prompt persona specification is being honored</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Internal tools with no defined persona where tone is irrelevant</li>
<li>Factual accuracy assessment — use <a href="../composite/faithfulness.md">Faithfulness</a></li>
<li>Goal outcome evaluation — use <a href="./goal_completion.md">Goal Completion</a></li>
<li>Single-turn Q&A where there are no prior turns to drift from</li>
</ul>
</div>

</div>

!!! tip "Best Results: Provide a Persona Spec"
    Pass the agent's persona description via `additional_input` for the most precise evaluation. Without it, the metric infers persona from early turns — which works, but is less precise than an explicit spec.

    ```python
    item = DatasetItem(
        conversation=[...],
        additional_input="You are Aria, a warm and professional customer success agent for Acme Corp. Maintain a friendly but formal tone. Never use slang. Always address users by first name."
    )
    ```

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    An **Evaluator LLM** reads the conversation — and the persona spec if provided — then assesses each agent turn for adherence across four evaluation dimensions. Drift incidents are catalogued with turn-level precision.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A["conversation<br/><small>List of {role, content} turns</small>"]
            B["additional_input<br/><small>Persona spec (optional but recommended)</small>"]
        end

        subgraph ESTABLISH["🔍 Persona Baseline"]
            C["Extract or infer persona<br/><small>From spec or early turns</small>"]
            D["Identify: tone, formality, style, character traits"]
        end

        subgraph ASSESS["⚖️ Per-Turn Assessment"]
            E["Evaluate each agent turn<br/>against baseline dimensions"]
            F["Flag drift incidents<br/><small>Turn, dimension, severity</small>"]
        end

        subgraph SCORE["📊 Scoring"]
            G["Aggregate drift severity"]
            H["Clamp to [0, 1]"]
            I["Final Score"]
        end

        A & B --> C
        C --> D
        D --> E
        E --> F
        F --> G
        G --> H
        H --> I

        style INPUT stroke:#1E3A5F,stroke-width:2px
        style ESTABLISH stroke:#3b82f6,stroke-width:2px
        style ASSESS stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style I fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-layers: Evaluation Dimensions"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">🎭 Tone Consistency</strong>
    <br><small>Does the agent maintain the same emotional register across turns? Tracks shifts from warm to curt, professional to casual, confident to uncertain.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">📐 Formality Level</strong>
    <br><small>Does the agent hold a consistent formality level? Flags inappropriate contractions, slang, or sudden formality spikes in casual-persona agents.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✍️ Language Style Alignment</strong>
    <br><small>Does the agent's vocabulary, sentence structure, and phrasing match the specified or established style? Tracks divergence from the persona's linguistic fingerprint.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">🌀 Persona Drift</strong>
    <br><small>Has the agent's overall character shifted? Catches compound drift where multiple small deviations across dimensions produce a meaningfully different persona by the end of the conversation.</small>
    </div>

    </div>

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level (`GRANULAR` or `HOLISTIC`) |

=== ":material-play: Basic Usage (No Spec)"

    ```python
    from axion.metrics import PersonaToneAdherence
    from axion.dataset import DatasetItem

    metric = PersonaToneAdherence()

    item = DatasetItem(
        conversation=[
            {"role": "user", "content": "Hello, I have a billing question."},
            {"role": "assistant", "content": "Good afternoon! I'd be delighted to help. Could you share the details?"},
            {"role": "user", "content": "My invoice seems wrong."},
            {"role": "assistant", "content": "yeah sure what's the issue lol"},  # Drift detected
        ]
    )

    result = await metric.execute(item)
    print(result.pretty())
    ```

=== ":material-shield-check: With Persona Spec"

    ```python
    from axion.metrics import PersonaToneAdherence
    from axion.dataset import DatasetItem

    metric = PersonaToneAdherence()

    persona_spec = (
        "You are Aria, a warm and professional customer success agent for Acme Corp. "
        "Maintain a friendly but formal tone at all times. Never use contractions or slang. "
        "Always address users by their first name when known."
    )

    item = DatasetItem(
        conversation=[
            {"role": "user", "content": "Hi, I'm Sarah. I need help with my order."},
            {"role": "assistant", "content": "Hello, Sarah! I am happy to assist. Could you share your order number?"},
            {"role": "user", "content": "It's 4821."},
            {"role": "assistant", "content": "Thanks Sarah, let me check that for you."},  # "Thanks" may drift
        ],
        additional_input=persona_spec,
    )

    result = await metric.execute(item)
    print(result.pretty())
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import PersonaToneAdherence
    from axion.runners import MetricRunner

    metric = PersonaToneAdherence()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        for drift in item_result.data.drift_incidents:
            print(f"  - Turn {drift.turn} [{drift.dimension}]: {drift.description}")
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
<summary><strong>📊 PersonaToneAdherenceResult Structure</strong></summary>

```python
PersonaToneAdherenceResult(
{
    "overall_score": 0.55,
    "persona_baseline": "Warm, formal, professional customer success agent. No contractions or slang.",
    "drift_incidents": [
        {
            "turn": 4,
            "dimension": "formality",
            "severity": "high",
            "description": "'yeah sure what's the issue lol' is highly informal and inconsistent with the established warm-professional persona."
        }
    ],
    "dimension_scores": {
        "tone_consistency": 0.6,
        "formality_level": 0.2,
        "language_style": 0.6,
        "persona_drift": 0.8
    },
    "reason": "A severe formality breakdown in turn 4 significantly degraded the overall score. All other turns maintained appropriate tone and style."
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | `float` | The 0–1 adherence score |
| `persona_baseline` | `str` | Description of the persona used for evaluation (inferred or from spec) |
| `drift_incidents` | `List` | Per-incident records with turn, dimension, severity, and description |
| `dimension_scores` | `Dict` | Individual scores per evaluation dimension |
| `reason` | `str` | Human-readable explanation of the overall score |

### Drift Incident Fields

| Field | Type | Description |
|-------|------|-------------|
| `turn` | `int` | Turn index (1-based) where drift occurred |
| `dimension` | `str` | Dimension: `tone_consistency`, `formality_level`, `language_style`, or `persona_drift` |
| `severity` | `str` | `low`, `medium`, or `high` |
| `description` | `str` | Specific explanation of the drift |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Perfect Adherence (Score: 1.0)</strong></summary>

!!! success "Persona Maintained"

    **Persona spec:** "Friendly, concise, slightly informal tech support agent. Uses contractions. First-name basis."

    **Conversation:** All 6 turns use consistent informal-friendly register, first names, and contractions throughout.

    **Drift incidents:** 0

    **Final Score:** `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Gradual Drift (Score: 0.6)</strong></summary>

!!! warning "Compound Drift"

    **Persona spec:** "Professional, formal financial advisor persona. No slang. Complete sentences."

    **Drift incidents:**
    - Turn 3: Used "no worries" (`formality_level`, medium)
    - Turn 5: Incomplete sentence fragment (`language_style`, low)
    - Turn 7: Switched to first-person casual ("I think maybe…") (`tone_consistency`, medium)

    By turn 7, the agent reads significantly more casual than the spec requires.

    **Final Score:** `0.6` :material-alert:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Severe Persona Break (Score: 0.2)</strong></summary>

!!! failure "Persona Collapsed"

    **Persona spec:** "Calm, empathetic mental health support companion."

    A frustrated user made three repeated complaints. The agent's tone shifted to dismissive and curt by turn 5, directly contradicting the empathy requirement.

    **Drift incidents:** 3 high-severity incidents across tone and persona dimensions.

    **Final Score:** `0.2` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🎭</span>
<strong>Brand Consistency</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">An agent that breaks persona mid-conversation erodes user trust and brand integrity. Persona & Tone Adherence provides a measurable signal for brand consistency at scale.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">😤</span>
<strong>Stress-Test Discovery</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Persona drift most commonly occurs under user pressure — repeated complaints, frustration, or adversarial inputs. This metric surfaces those vulnerabilities before deployment.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>System Prompt Validation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Pass the system prompt persona spec as <code>additional_input</code> to validate that the model actually honors your persona instructions — a critical check for production-ready branded agents.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Persona & Tone Adherence** = Does the agent maintain a consistent voice, tone, and character throughout the conversation — especially under user pressure?

    - **Use it when:** Your agent has a defined persona or brand voice that must be held consistently
    - **Score interpretation:** Higher = more consistent persona; 0.5 passes by default
    - **Key tip:** Pass the persona spec via `additional_input` for the most precise evaluation

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.PersonaToneAdherence`](../../reference/metrics.md#persona-tone-adherence)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Conversation Flow](./conversation_flow.md) · [Goal Completion](./goal_completion.md) · [Conversation Efficiency](./conversation_efficiency.md)

</div>
