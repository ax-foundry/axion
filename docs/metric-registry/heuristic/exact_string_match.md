# Exact String Match

<div style="border-left: 4px solid #f59e0b; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Check if output exactly matches expected text</strong><br>
<span class="badge" style="margin-top: 0.5rem; background: #f59e0b;">Heuristic</span>
<span class="badge" style="background: #6B7A3A;">Single Turn</span>
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
<small style="color: var(--md-text-muted);">Full text comparison</small>
</div>

</div>

!!! abstract "What It Measures"
    Exact String Match checks whether the **actual output is identical** to the expected output (after stripping whitespace). This is the strictest form of text matching—the response must be exactly what was expected.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Perfect match—strings are identical |
    | **0.0** | :material-close: No match—any difference fails |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Exact output format is required</li>
<li>Testing deterministic transformations</li>
<li>Validating code generation</li>
<li>Comparing structured outputs (JSON, XML)</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Minor variations are acceptable</li>
<li>Semantic equivalence matters more</li>
<li>Paraphrasing is allowed</li>
<li>Case differences should be ignored</li>
</ul>
</div>

</div>

!!! tip "See Also: Contains Match"
    **Exact String Match** checks if actual *equals* expected.
    **[Contains Match](./contains_match.md)** checks if expected is a *substring* of actual.

    Use Exact Match when precision matters; use Contains Match for keyword validation.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    Simple string equality check after stripping whitespace.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Actual Output]
            B[Expected Output]
        end

        subgraph PROCESS["🔍 Processing"]
            C[Strip whitespace]
            D["Check: actual == expected?"]
        end

        subgraph OUTPUT["📊 Result"]
            E["1.0 = Match"]
            F["0.0 = No Match"]
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
    score = 1.0 if actual.strip() == expected.strip() else 0.0
    ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | *(none)* | - | - | No configuration options |

    !!! info "Simple by Design"
        Exact String Match is intentionally simple with no configuration. For case-insensitive or fuzzy matching, use Levenshtein Ratio.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import ExactStringMatch
    from axion.dataset import DatasetItem

    metric = ExactStringMatch()

    item = DatasetItem(
        actual_output="Hello, World!",
        expected_output="Hello, World!",
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - exact match
    ```

=== ":material-close: No Match Example"

    ```python
    from axion.metrics import ExactStringMatch

    metric = ExactStringMatch()

    item = DatasetItem(
        actual_output="Hello, world!",  # lowercase 'w'
        expected_output="Hello, World!",  # uppercase 'W'
    )

    result = await metric.execute(item)
    print(result.score)  # 0.0 - case mismatch
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import ExactStringMatch
    from axion.runners import MetricRunner

    metric = ExactStringMatch()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    exact_matches = sum(1 for r in results if r.score == 1.0)
    print(f"Exact matches: {exact_matches}/{len(results)}")
    ```

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Perfect Match (Score: 1.0)</strong></summary>

!!! success "Identical Strings"

    **Expected:** `"The answer is 42."`

    **Actual:** `"The answer is 42."`

    **Result:** `1.0` :material-check-all:

    *Strings are character-for-character identical.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 2: Case Mismatch (Score: 0.0)</strong></summary>

!!! failure "Different Case"

    **Expected:** `"PASS"`

    **Actual:** `"Pass"`

    **Result:** `0.0` :material-close:

    *Case sensitivity causes failure.*

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 3: Whitespace Handling</strong></summary>

!!! warning "Leading/Trailing Whitespace"

    **Expected:** `"  Hello  "`

    **Actual:** `"Hello"`

    **Result:** `1.0` :material-check-all:

    *Whitespace is stripped before comparison, so these match.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 4: Extra Content (Score: 0.0)</strong></summary>

!!! failure "Additional Text"

    **Expected:** `"Paris"`

    **Actual:** `"The answer is Paris."`

    **Result:** `0.0` :material-close:

    *Even containing the expected text, the strings aren't equal.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">⚡</span>
<strong>Instant Results</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">O(n) string comparison. No external dependencies or LLM calls.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>Maximum Precision</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">No false positives—only identical strings pass.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔧</span>
<strong>Deterministic Testing</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Perfect for testing deterministic transformations and code generation.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Exact String Match** = Is the output identical to the expected text?

    - **Use it when:** Exact output format is required (code, JSON, IDs)
    - **Score interpretation:** 1.0 = identical, 0.0 = any difference
    - **Key behavior:** Case-sensitive, whitespace-trimmed

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.ExactStringMatch`](../../reference/metrics.md#exact-string-match)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Contains Match](./contains_match.md) · Levenshtein Ratio · Sentence BLEU

</div>
