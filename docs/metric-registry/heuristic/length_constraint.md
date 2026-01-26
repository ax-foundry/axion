# Length Constraint

<div style="border-left: 4px solid #f59e0b; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Verify response length meets character and sentence constraints</strong><br>
<span class="badge" style="margin-top: 0.5rem; background: #f59e0b;">Heuristic</span>
<span class="badge" style="background: #6B7A3A;">Single Turn</span>
<span class="badge" style="background: #ec4899;">Binary</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📏</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> or <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Binary pass/fail</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">1.0</code><br>
<small style="color: var(--md-text-muted);">Pass/fail cutoff</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code><br>
<small style="color: var(--md-text-muted);">Text to evaluate</small>
</div>

</div>

!!! abstract "What It Measures"
    Length Constraint verifies that the **response meets character and/or sentence count constraints**. This is useful for enforcing output limits in chatbots, summaries, or any application with length requirements.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Passed—within all constraints |
    | **0.0** | :material-close: Failed—exceeded character limit or outside sentence range |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Enforcing token/character limits for LLM outputs</li>
<li>Validating summary length requirements</li>
<li>Ensuring concise responses in chatbots</li>
<li>Meeting UI display constraints</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Quality matters more than length</li>
<li>Variable-length outputs are acceptable</li>
<li>Sentence structure is irregular (code, lists)</li>
</ul>
</div>

</div>

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    Checks character count against `max_chars` and sentence count against `sentence_range`.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["Input"]
            A[Actual Output]
        end

        subgraph PROCESS["Processing"]
            B[Count characters]
            C[Split into sentences]
            D["Check: chars <= max_chars?"]
            E["Check: sentence_count in range?"]
        end

        subgraph OUTPUT["Result"]
            F["1.0 = All constraints met"]
            G["0.0 = Constraint violated"]
        end

        A --> B
        A --> C
        B --> D
        C --> E
        D & E -->|All pass| F
        D & E -->|Any fail| G

        style INPUT stroke:#f59e0b,stroke-width:2px
        style PROCESS stroke:#3b82f6,stroke-width:2px
        style OUTPUT stroke:#10b981,stroke-width:2px
    ```

=== ":material-code-tags: Logic"

    ```python
    # Character check
    char_passed = len(text) <= max_chars

    # Sentence check (splits on .!? followed by space or end)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences)

    min_s, max_s = sentence_range
    sentence_passed = (min_s is None or sentence_count >= min_s) and \
                      (max_s is None or sentence_count <= max_s)

    score = 1.0 if (char_passed and sentence_passed) else 0.0
    ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `max_chars` | `int` | `2800` | Maximum allowed characters. Set to `None` to disable. |
    | `sentence_range` | `tuple` | `None` | `(min, max)` sentence bounds. Use `None` for open ends. |
    | `field_mapping` | `dict` | `None` | Remap `actual_output` to another field path. |

    !!! tip "Flexible Sentence Ranges"
        - `(3, 5)` — Between 3 and 5 sentences (inclusive)
        - `(None, 5)` — At most 5 sentences
        - `(3, None)` — At least 3 sentences

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import LengthConstraint
    from axion.dataset import DatasetItem

    # Default: max 2800 characters, no sentence constraint
    metric = LengthConstraint()

    item = DatasetItem(
        actual_output="This is a short response.",
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - within limits
    print(result.signals.char_count)  # 26
    ```

=== ":material-counter: With Sentence Range"

    ```python
    from axion.metrics import LengthConstraint

    # Require 2-4 sentences, max 500 characters
    metric = LengthConstraint(
        max_chars=500,
        sentence_range=(2, 4),
    )

    item = DatasetItem(
        actual_output="First sentence. Second sentence. Third sentence.",
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - 3 sentences, 48 chars
    print(result.signals.sentence_count)  # 3
    ```

=== ":material-close: Constraint Violation"

    ```python
    from axion.metrics import LengthConstraint

    metric = LengthConstraint(
        max_chars=50,
        sentence_range=(None, 2),  # max 2 sentences
    )

    item = DatasetItem(
        actual_output="First. Second. Third. Fourth. Fifth sentence here.",
    )

    result = await metric.execute(item)
    print(result.score)  # 0.0 - too many sentences
    print(result.explanation)
    # "FAILED. Sentence count 5 outside (Range: 0-2)."
    ```

=== ":material-map-marker-path: Field Mapping"

    ```python
    from axion.metrics import LengthConstraint

    # Evaluate a nested field instead of actual_output
    metric = LengthConstraint(
        max_chars=1000,
        field_mapping={'actual_output': 'additional_output.summary'},
    )

    item = DatasetItem(
        additional_output={'summary': 'A brief summary of the document.'},
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import LengthConstraint
    from axion.runners import MetricRunner

    metric = LengthConstraint(max_chars=1000, sentence_range=(1, 5))
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    passed = sum(1 for r in results if r.score == 1.0)
    print(f"Passed length constraints: {passed}/{len(results)}")
    ```

---

## Example Scenarios

<details markdown="1">
<summary><strong>Scenario 1: Within All Constraints (Score: 1.0)</strong></summary>

!!! success "Passed"

    **Config:** `max_chars=100`, `sentence_range=(1, 3)`

    **Actual Output:** `"Hello there. How can I help you today?"`

    **Result:** `1.0` :material-check-all:

    - Characters: 39 (within 100)
    - Sentences: 2 (within 1-3 range)

</details>

<details markdown="1">
<summary><strong>Scenario 2: Character Limit Exceeded (Score: 0.0)</strong></summary>

!!! failure "Failed - Too Long"

    **Config:** `max_chars=50`

    **Actual Output:** `"This response is definitely going to exceed the maximum character limit that was set."`

    **Result:** `0.0` :material-close:

    **Explanation:** `"FAILED. Exceeded chars (85/50)."`

</details>

<details markdown="1">
<summary><strong>Scenario 3: Too Few Sentences (Score: 0.0)</strong></summary>

!!! failure "Failed - Below Minimum"

    **Config:** `sentence_range=(3, None)` (at least 3 sentences)

    **Actual Output:** `"Just one sentence."`

    **Result:** `0.0` :material-close:

    **Explanation:** `"FAILED. Sentence count 1 outside (Range: 3-inf)."`

</details>

<details markdown="1">
<summary><strong>Scenario 4: Too Many Sentences (Score: 0.0)</strong></summary>

!!! failure "Failed - Above Maximum"

    **Config:** `sentence_range=(None, 2)` (at most 2 sentences)

    **Actual Output:** `"First point. Second point. Third point. And more!"`

    **Result:** `0.0` :material-close:

    **Explanation:** `"FAILED. Sentence count 4 outside (Range: 0-2)."`

</details>

---

## Signal Details

The metric returns a `LengthResult` object with detailed information:

| Signal | Type | Description |
|--------|------|-------------|
| `char_count` | `int` | Total characters in the output |
| `max_chars_allowed` | `int` | Configured maximum (or `None` if disabled) |
| `sentence_count` | `int` | Number of sentences detected |
| `sentence_range` | `tuple` | Configured `(min, max)` range |
| `passed` | `bool` | Whether all constraints were met |

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">⚡</span>
<strong>Instant Results</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">O(n) computation. No LLM calls required.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📱</span>
<strong>UI Compliance</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensure outputs fit display constraints and character limits.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">💰</span>
<strong>Cost Control</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Prevent overly verbose outputs that increase token costs.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>Conciseness</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Enforce brevity requirements for summaries and chatbots.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Length Constraint** = Does the output meet character and sentence limits?

    - **Use it when:** Enforcing length limits for UI, cost, or conciseness
    - **Score interpretation:** 1.0 = all constraints met, 0.0 = any violation
    - **Key behavior:** Character counting + regex-based sentence detection

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.LengthConstraint`](../../reference/metrics.md#length-constraint)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Latency](./latency.md) - Performance metrics

</div>
