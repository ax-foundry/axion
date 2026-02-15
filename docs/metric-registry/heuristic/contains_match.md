# Contains Match

<div style="border-left: 4px solid #f59e0b; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Check if output contains the expected text as a substring</strong><br>
<span class="badge" style="margin-top: 0.5rem; background: #f59e0b;">Heuristic</span>
<span class="badge" style="background: #0F2440;">Single Turn</span>
<span class="badge" style="background: #ec4899;">Binary</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> or <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Binary pass/fail</small>
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
<code>actual_output</code> <code>expected_output</code><br>
<small style="color: var(--md-text-muted);">Substring to find</small>
</div>

</div>

!!! abstract "What It Measures"
    Contains Match checks whether the **expected output appears as a substring** within the actual output. This is the simplest form of text matching—does the response include the required content anywhere?

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Expected text found in output |
    | **0.0** | :material-close: Expected text not found |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Checking for required keywords/phrases</li>
<li>Validating specific content inclusion</li>
<li>Simple pass/fail tests</li>
<li>Fast sanity checks</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Exact match required (use Exact String Match)</li>
<li>Similarity scoring needed (use BLEU/Levenshtein)</li>
<li>Case variations matter</li>
<li>Semantic matching needed</li>
</ul>
</div>

</div>

!!! tip "See Also: Exact String Match"
    **Contains Match** checks if expected is a *substring* of actual.
    **[Exact String Match](./exact_string_match.md)** checks if they are *identical*.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    Simple substring check after stripping whitespace.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Actual Output]
            B[Expected Output]
        end

        subgraph PROCESS["🔍 Processing"]
            C[Strip whitespace]
            D["Check: expected in actual?"]
        end

        subgraph OUTPUT["📊 Result"]
            E["1.0 = Found"]
            F["0.0 = Not Found"]
        end

        A & B --> C
        C --> D
        D -->|Yes| E
        D -->|No| F

        style INPUT stroke:#f59e0b,stroke-width:2px
        style PROCESS stroke:#3b82f6,stroke-width:2px
        style OUTPUT stroke:#10b981,stroke-width:2px
    ```

=== ":material-code-tags: Logic"

    ```python
    score = 1.0 if expected.strip() in actual.strip() else 0.0
    ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | *(none)* | - | - | No configuration options |

    !!! info "Simple by Design"
        Contains Match is intentionally simple with no configuration. For case-insensitive matching or fuzzy matching, use other metrics.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ContainsMatch
    from axion.dataset import DatasetItem

    metric = ContainsMatch()

    item = DatasetItem(
        actual_output="The capital of France is Paris, a beautiful city.",
        expected_output="Paris",
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - "Paris" is in the output
    ```

=== ":material-close: No Match Example"

    ```python
    from axion.metrics import ContainsMatch

    metric = ContainsMatch()

    item = DatasetItem(
        actual_output="The capital of France is a beautiful city.",
        expected_output="Paris",
    )

    result = await metric.execute(item)
    print(result.score)  # 0.0 - "Paris" not found
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ContainsMatch
    from axion.runners import MetricRunner

    metric = ContainsMatch()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    passed = sum(1 for r in results if r.score == 1.0)
    print(f"Passed: {passed}/{len(results)}")
    ```

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Substring Found (Score: 1.0)</strong></summary>

!!! success "Match Found"

    **Expected:** `"42"`

    **Actual:** `"The answer to life, the universe, and everything is 42."`

    **Result:** `1.0` :material-check-all:

    *The substring "42" exists in the output.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 2: Substring Not Found (Score: 0.0)</strong></summary>

!!! failure "No Match"

    **Expected:** `"Python"`

    **Actual:** `"JavaScript is a popular programming language."`

    **Result:** `0.0` :material-close:

    *"Python" does not appear in the output.*

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 3: Case Sensitivity</strong></summary>

!!! warning "Case Matters"

    **Expected:** `"PARIS"`

    **Actual:** `"The capital is Paris."`

    **Result:** `0.0` :material-close:

    *"PARIS" (uppercase) does not match "Paris" (title case).*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">⚡</span>
<strong>Instant Results</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">O(n) substring search. No external dependencies or LLM calls.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">✅</span>
<strong>Sanity Checks</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Quickly verify required content is present in responses.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Keyword Validation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensure specific terms, codes, or phrases appear in output.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Contains Match** = Does the output contain the expected text?

    - **Use it when:** Checking for required keywords or phrases
    - **Score interpretation:** 1.0 = found, 0.0 = not found
    - **Key behavior:** Case-sensitive substring match

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ContainsMatch`](../../reference/metrics.md#contains-match)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Exact String Match](./exact_string_match.md) · Sentence BLEU · Levenshtein Ratio

</div>
