# Levenshtein Ratio

<div style="border-left: 4px solid #f59e0b; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Calculate character-level string similarity</strong><br>
<span class="badge" style="margin-top: 0.5rem; background: #f59e0b;">Heuristic</span>
<span class="badge" style="background: #6B7A3A;">Single Turn</span>
<span class="badge" style="background: #06b6d4;">Fast</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Character-level similarity</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.2</code><br>
<small style="color: var(--md-text-muted);">Pass/fail cutoff</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code> <code>expected_output</code><br>
<small style="color: var(--md-text-muted);">Text comparison</small>
</div>

</div>

!!! abstract "What It Measures"
    Levenshtein Ratio calculates the **character-level similarity** between two strings using the SequenceMatcher algorithm. It measures how many edits (insertions, deletions, substitutions) are needed to transform one string into another.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Identical strings |
    | **0.8+** | :material-check: Very similar, minor typos |
    | **0.5-0.8** | :material-alert: Moderate similarity |
    | **< 0.5** | :material-close: Significant differences |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Checking for typos or small variations</li>
<li>Fuzzy string matching needed</li>
<li>Comparing names or identifiers</li>
<li>Near-match detection</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Semantic similarity matters</li>
<li>Word-level comparison preferred (use BLEU)</li>
<li>Long texts with different structures</li>
<li>Exact match required</li>
</ul>
</div>

</div>

!!! tip "See Also: Sentence BLEU"
    **Levenshtein Ratio** measures character-level edit distance.
    **[Sentence BLEU](./sentence_bleu.md)** measures word-level n-gram precision.

    Use Levenshtein for typo detection; use BLEU for paraphrase comparison.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    Uses Python's `SequenceMatcher` to calculate the ratio of matching characters.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Actual Output]
            B[Expected Output]
        end

        subgraph PROCESS["🔍 Processing"]
            C[Optional: Convert to lowercase]
            D[Find matching subsequences]
            E[Calculate similarity ratio]
        end

        subgraph OUTPUT["📊 Result"]
            F["Score: 0.0 to 1.0"]
        end

        A & B --> C
        C --> D
        D --> E
        E --> F

        style INPUT stroke:#f59e0b,stroke-width:2px
        style PROCESS stroke:#3b82f6,stroke-width:2px
        style OUTPUT stroke:#10b981,stroke-width:2px
    ```

=== ":material-calculator: Formula"

    The SequenceMatcher ratio is calculated as:

    ```
    ratio = 2.0 * M / T

    where:
    M = number of matches (characters in common)
    T = total number of characters in both strings
    ```

    **Example:**
    ```
    String 1: "hello"
    String 2: "hallo"

    Matches: h, l, l, o = 4 characters match
    Total: 5 + 5 = 10 characters
    Ratio: 2.0 * 4 / 10 = 0.8
    ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `case_sensitive` | `bool` | `False` | Whether comparison is case-sensitive |

    !!! info "Case Sensitivity"
        By default, comparison is case-insensitive (both strings converted to lowercase). Set `case_sensitive=True` for strict character matching.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import LevenshteinRatio
    from axion.dataset import DatasetItem

    metric = LevenshteinRatio()

    item = DatasetItem(
        actual_output="The quick brown fox",
        expected_output="The quick brown fox",
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - identical strings
    ```

=== ":material-tune-variant: Case Sensitive"

    ```python
    from axion.metrics import LevenshteinRatio

    # Case insensitive (default)
    metric = LevenshteinRatio(case_sensitive=False)

    item = DatasetItem(
        actual_output="HELLO",
        expected_output="hello",
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - case ignored

    # Case sensitive
    metric_strict = LevenshteinRatio(case_sensitive=True)
    result_strict = await metric_strict.execute(item)
    print(result_strict.score)  # 0.0 - case matters
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import LevenshteinRatio
    from axion.runners import MetricRunner

    metric = LevenshteinRatio()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    avg_similarity = sum(r.score for r in results) / len(results)
    print(f"Average similarity: {avg_similarity:.2%}")
    ```

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: High Similarity (Score: 0.95)</strong></summary>

!!! success "Minor Typo"

    **Expected:** `"accommodation"`

    **Actual:** `"accomodation"` (missing 'm')

    **Result:** `~0.92` :material-check:

    *Single character difference results in high similarity.*

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Moderate Similarity (Score: 0.67)</strong></summary>

!!! warning "Multiple Differences"

    **Expected:** `"Hello World"`

    **Actual:** `"Helo Wrld"` (missing letters)

    **Result:** `~0.67` :material-alert:

    *Several missing characters reduce similarity.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Low Similarity (Score: 0.2)</strong></summary>

!!! failure "Very Different Strings"

    **Expected:** `"The quick brown fox"`

    **Actual:** `"A lazy dog sleeps"`

    **Result:** `~0.2` :material-close:

    *Completely different content results in low similarity.*

</details>

<details markdown="1">
<summary><strong>✅ Scenario 4: Case Handling</strong></summary>

!!! success "Case Insensitive Match"

    **Expected:** `"OpenAI"`

    **Actual:** `"openai"`

    **Result (default):** `1.0` :material-check-all:

    **Result (case_sensitive=True):** `~0.67` :material-alert:

    *Case sensitivity significantly affects scoring.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">⚡</span>
<strong>Fast & Deterministic</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">No LLM calls needed. Instant, reproducible results.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔤</span>
<strong>Typo Detection</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Perfect for detecting spelling errors and near-matches.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📊</span>
<strong>Gradual Scoring</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Unlike binary metrics, provides nuanced similarity scores.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Levenshtein Ratio** = How similar are the strings at the character level?

    - **Use it when:** Checking for typos or fuzzy matching
    - **Score interpretation:** Higher = more similar characters
    - **Key config:** `case_sensitive` controls case handling

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.LevenshteinRatio`](../../reference/metrics.md#levenshtein-ratio)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Sentence BLEU](./sentence_bleu.md) · Exact String Match · Contains Match

</div>
