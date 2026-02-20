---
icon: custom/tool
---
# Tool Correctness

<div style="border-left: 4px solid #8b5cf6; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluate whether AI agents call the correct tools with proper parameters</strong><br>
<span class="badge" style="margin-top: 0.5rem; background: #8b5cf6;">Tool</span>
<span class="badge" style="background: #667eea;">Agent</span>
<span class="badge" style="background: #0F2440;">Single Turn</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Recall of expected tools</small>
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
<code>tools_called</code> <code>expected_tools</code><br>
<small style="color: var(--md-text-muted);">Tool call lists</small>
</div>

</div>

!!! abstract "What It Measures"
    Tool Correctness evaluates whether an AI agent **called the correct tools** by comparing actual tool calls against expected ones. It supports name-only matching, parameter validation, and strict ordering requirements.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All expected tools called correctly |
    | **0.5-0.9** | :material-alert: Partial match—some tools missing |
    | **0.0** | :material-close: No expected tools called correctly |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Evaluating AI agents</li>
<li>Testing function calling</li>
<li>Validating tool selection</li>
<li>Checking parameter passing</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Order doesn't matter (consider disabling strict_order)</li>
<li>Tool output quality matters more</li>
<li>Parameters have valid variations</li>
<li>No expected tools defined</li>
</ul>
</div>

</div>

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric compares called tools against expected tools with configurable matching strategies.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Tools Called]
            B[Expected Tools]
        end

        subgraph CONFIG["⚙️ Configuration"]
            C{Strict Order?}
            D{Check Parameters?}
        end

        subgraph MATCH["🔍 Matching"]
            E[Compare names]
            F[Validate parameters]
            G[Check sequence]
        end

        subgraph SCORE["📊 Score"]
            H["matched / expected"]
        end

        A & B --> C
        C -->|No| E
        C -->|Yes| G
        E --> D
        G --> D
        D -->|Yes| F
        D -->|No| H
        F --> H

        style INPUT stroke:#8b5cf6,stroke-width:2px
        style CONFIG stroke:#3b82f6,stroke-width:2px
        style MATCH stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
    ```

=== ":material-tune-variant: Matching Strategies"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">Name Only</strong> (default)
    <br><small>Just check if the tool name matches. Parameters ignored.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">Exact Parameters</strong>
    <br><small>Parameters must match exactly.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">Subset Parameters</strong>
    <br><small>Called args must contain all expected args (extras OK).</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #8b5cf6; padding-left: 1rem;">
    <strong style="color: #8b5cf6;">Fuzzy Parameters</strong>
    <br><small>Similarity-based matching with threshold.</small>
    </div>

    </div>

=== ":material-calculator: Score Formula"

    ```
    score = matched_tools / total_expected_tools
    ```

    **Example:**
    - Expected: `[search, calculate, format]`
    - Called: `[search, format]`
    - Score: `2/3 = 0.67`

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `check_parameters` | `bool` | `False` | Also validate tool parameters |
    | `strict_order` | `bool` | `False` | Tools must be called in exact order |
    | `parameter_matching_strategy` | `str` | `exact` | `exact`, `subset`, or `fuzzy` |
    | `fuzzy_threshold` | `float` | `0.8` | Similarity threshold for fuzzy matching |

    !!! info "Parameter Matching Strategies"
        - **exact**: Parameters must match exactly (default)
        - **subset**: Called args must contain all expected args (extras allowed)
        - **fuzzy**: Similarity-based matching using SequenceMatcher

---

## Code Examples

=== ":material-play: Basic Usage (Name Only)"

    ```python
    from axion.metrics import ToolCorrectness
    from axion.dataset import DatasetItem
    from axion._core.schema import ToolCall

    metric = ToolCorrectness()

    item = DatasetItem(
        tools_called=[
            ToolCall(name="search", args={"query": "weather"}),
            ToolCall(name="format", args={"style": "brief"}),
        ],
        expected_tools=[
            ToolCall(name="search", args={}),
            ToolCall(name="format", args={}),
        ],
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - both tools called (params not checked)
    ```

=== ":material-check-all: With Parameter Checking"

    ```python
    from axion.metrics import ToolCorrectness

    metric = ToolCorrectness(
        check_parameters=True,
        parameter_matching_strategy='exact'
    )

    item = DatasetItem(
        tools_called=[
            ToolCall(name="calculate", args={"a": 5, "b": 3}),
        ],
        expected_tools=[
            ToolCall(name="calculate", args={"a": 5, "b": 3}),
        ],
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - params match exactly
    ```

=== ":material-sort: Strict Order"

    ```python
    from axion.metrics import ToolCorrectness

    metric = ToolCorrectness(strict_order=True)

    # Correct order
    item_correct = DatasetItem(
        tools_called=[
            ToolCall(name="fetch", args={}),
            ToolCall(name="process", args={}),
            ToolCall(name="store", args={}),
        ],
        expected_tools=[
            ToolCall(name="fetch", args={}),
            ToolCall(name="process", args={}),
            ToolCall(name="store", args={}),
        ],
    )
    # Score: 1.0

    # Wrong order
    item_wrong = DatasetItem(
        tools_called=[
            ToolCall(name="process", args={}),  # Should be second
            ToolCall(name="fetch", args={}),    # Should be first
            ToolCall(name="store", args={}),
        ],
        expected_tools=[
            ToolCall(name="fetch", args={}),
            ToolCall(name="process", args={}),
            ToolCall(name="store", args={}),
        ],
    )
    # Score: 0.0 - order mismatch at position 0
    ```

=== ":material-tune-vertical: Fuzzy Parameter Matching"

    ```python
    from axion.metrics import ToolCorrectness

    metric = ToolCorrectness(
        check_parameters=True,
        parameter_matching_strategy='fuzzy',
        fuzzy_threshold=0.8
    )

    item = DatasetItem(
        tools_called=[
            ToolCall(name="search", args={"query": "what is machine learning"}),
        ],
        expected_tools=[
            ToolCall(name="search", args={"query": "what is ML"}),
        ],
    )

    result = await metric.execute(item)
    # Score depends on string similarity of query values
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ToolCorrectness
    from axion.runners import MetricRunner

    metric = ToolCorrectness(check_parameters=True)
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score:.2f}")
        print(f"Explanation: {item_result.explanation}")
    ```

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Perfect Match (Score: 1.0)</strong></summary>

!!! success "All Tools Correct"

    **Expected Tools:**

    1. `search(query="weather")`
    2. `parse(format="json")`

    **Called Tools:**

    1. `search(query="weather")`
    2. `parse(format="json")`

    **Result:** `1.0` :material-check-all:

    *All expected tools called with correct parameters.*

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Partial Match (Score: 0.67)</strong></summary>

!!! warning "Missing Tool"

    **Expected Tools:**

    1. `fetch`
    2. `transform`
    3. `store`

    **Called Tools:**

    1. `fetch`
    2. `transform`
    *(store not called)*

    **Result:** `2/3 = 0.67` :material-alert:

    **Explanation:** "Correctly called: ['fetch', 'transform']; Missing tools: ['store']"

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Wrong Tool (Score: 0.0)</strong></summary>

!!! failure "Incorrect Tool Called"

    **Expected Tools:**

    1. `calculate`

    **Called Tools:**

    1. `search`

    **Result:** `0.0` :material-close:

    **Explanation:** "Missing tools: ['calculate']; Unexpected tools: ['search']"

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 4: Parameter Mismatch</strong></summary>

!!! warning "Wrong Parameters"

    **Config:** `check_parameters=True, strategy='exact'`

    **Expected:**
    > `search(query="Python tutorials")`

    **Called:**
    > `search(query="python tutorial")`  *(different text)*

    **Result:** `0.0` :material-close:

    *Exact matching fails on parameter difference.*

    **Fix:** Use `strategy='fuzzy'` or `strategy='subset'` for flexibility.

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🤖</span>
<strong>Agent Evaluation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Verify AI agents select and call the right tools for tasks.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔧</span>
<strong>Function Calling</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Test LLM function calling capabilities and parameter handling.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📊</span>
<strong>Workflow Validation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensure multi-step agent workflows execute correctly.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Tool Correctness** = Did the agent call the expected tools correctly?

    - **Use it when:** Evaluating AI agents or function calling
    - **Score interpretation:** Fraction of expected tools called correctly
    - **Key configs:** `check_parameters`, `strict_order`, `parameter_matching_strategy`

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ToolCorrectness`](../../reference/metrics.md#tool-correctness)

- :material-link-variant: **Related Concepts**

    [:octicons-arrow-right-24: Agent Evaluation · Function Calling · Tool Use

</div>
