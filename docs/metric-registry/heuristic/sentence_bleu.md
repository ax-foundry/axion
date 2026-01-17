# Sentence BLEU

<div style="border-left: 4px solid #f59e0b; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Compute n-gram precision similarity between candidate and reference text</strong><br>
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
<small style="color: var(--md-text-muted);">N-gram precision score</small>
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
<small style="color: var(--md-text-muted);">Reference text required</small>
</div>

</div>

!!! abstract "What It Measures"
    Sentence BLEU (Bilingual Evaluation Understudy) computes the similarity between a candidate text and reference text using **modified n-gram precision** with a brevity penalty. Originally designed for machine translation, it's useful for any task where textual similarity to a reference matters.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Perfect n-gram match with reference |
    | **0.7+** | :material-check: High similarity, minor differences |
    | **0.3-0.7** | :material-alert: Moderate similarity |
    | **< 0.3** | :material-close: Low similarity to reference |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Comparing text to reference translations</li>
<li>Evaluating summarization quality</li>
<li>Fast, deterministic evaluation needed</li>
<li>N-gram overlap is meaningful</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Semantic similarity matters more than wording</li>
<li>Multiple valid phrasings exist</li>
<li>Evaluating creative/generative tasks</li>
<li>Word order flexibility is expected</li>
</ul>
</div>

</div>

!!! tip "See Also: Levenshtein Ratio"
    **Sentence BLEU** measures n-gram precision (word sequences).
    **[Levenshtein Ratio](./levenshtein_ratio.md)** measures character-level edit distance.

    Use BLEU for word-level comparison; use Levenshtein for character-level.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    BLEU calculates n-gram precision with clipping and applies a brevity penalty.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Candidate Text]
            B[Reference Text]
        end

        subgraph NGRAM["🔍 Step 1: N-gram Extraction"]
            C[Extract 1-grams to n-grams]
            D1["1-gram counts"]
            D2["2-gram counts"]
            D3["3-gram counts"]
            DN["n-gram counts"]
        end

        subgraph PRECISION["⚖️ Step 2: Clipped Precision"]
            E[Clip counts to reference max]
            F["Calculate precision per n"]
        end

        subgraph SCORE["📊 Step 3: Final Score"]
            G["Geometric mean of precisions"]
            H["Apply brevity penalty"]
            I["Final BLEU Score"]
        end

        A & B --> C
        C --> D1 & D2 & D3 & DN
        D1 & D2 & D3 & DN --> E
        E --> F
        F --> G
        G --> H
        H --> I

        style INPUT stroke:#f59e0b,stroke-width:2px
        style NGRAM stroke:#3b82f6,stroke-width:2px
        style PRECISION stroke:#8b5cf6,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style I fill:#f59e0b,stroke:#d97706,stroke-width:3px,color:#fff
    ```

=== ":material-calculator: BLEU Formula"

    **Modified Precision:**
    ```
    p_n = Σ min(count(ngram), max_ref_count(ngram)) / Σ count(ngram)
    ```

    **Brevity Penalty (BP):**
    ```
    BP = 1                    if c > r
    BP = exp(1 - r/c)         if c ≤ r

    where c = candidate length, r = reference length
    ```

    **Final Score:**
    ```
    BLEU = BP × exp(Σ w_n × log(p_n))

    where w_n = 1/N (uniform weights)
    ```

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">📝 Clipping</strong>
    <br><small>Prevents gaming by repeating words. Each n-gram counted at most as many times as it appears in reference.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">📏 Brevity Penalty</strong>
    <br><small>Penalizes outputs shorter than reference. Prevents gaming by outputting only high-confidence words.</small>
    </div>

    </div>

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `n_grams` | `int` | `4` | Maximum n-gram length (e.g., 4 for BLEU-4) |
    | `case_sensitive` | `bool` | `False` | Whether comparison is case-sensitive |
    | `smoothing` | `bool` | `True` | Apply smoothing for sentence-level BLEU |

    !!! info "Smoothing"
        Sentence-level BLEU often has zero counts for higher n-grams. Smoothing (add-one) prevents the entire score from becoming zero.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import SentenceBleu
    from axion.dataset import DatasetItem

    metric = SentenceBleu()

    item = DatasetItem(
        actual_output="The cat sat on the mat.",
        expected_output="The cat is sitting on the mat.",
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: ~0.6 (good n-gram overlap with minor differences)
    ```

=== ":material-tune-variant: Custom N-grams"

    ```python
    from axion.metrics import SentenceBleu

    # BLEU-2 for shorter sequences
    metric = SentenceBleu(n_grams=2)

    # Case-sensitive comparison
    metric = SentenceBleu(case_sensitive=True)

    # Without smoothing (corpus-level style)
    metric = SentenceBleu(smoothing=False)
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import SentenceBleu
    from axion.runners import MetricRunner

    metric = SentenceBleu(n_grams=4)
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"BLEU-4: {item_result.score:.3f}")
    ```

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: High BLEU Score (~0.9)</strong></summary>

!!! success "Near-Perfect Match"

    **Reference:**
    > "The quick brown fox jumps over the lazy dog."

    **Candidate:**
    > "The quick brown fox jumped over the lazy dog."

    **Analysis:**

    - 1-grams: 8/9 match (jumped vs jumps)
    - 2-grams: 7/8 match
    - 3-grams: 6/7 match
    - 4-grams: 5/6 match
    - Brevity penalty: ~1.0 (same length)

    **Final Score:** `~0.85` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Moderate BLEU Score (~0.5)</strong></summary>

!!! warning "Partial Overlap"

    **Reference:**
    > "Machine learning models require large datasets for training."

    **Candidate:**
    > "Deep learning needs big data to train properly."

    **Analysis:**

    - Same meaning, different words
    - Few exact n-gram matches
    - "learning" and "train" overlap

    **Final Score:** `~0.3` :material-alert:

    *Semantic similarity high, but n-gram overlap low.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Low BLEU Score (~0.1)</strong></summary>

!!! failure "Minimal Overlap"

    **Reference:**
    > "Paris is the capital of France."

    **Candidate:**
    > "The Eiffel Tower is located in the French capital city."

    **Analysis:**

    - Related topic, completely different wording
    - Almost no n-gram matches

    **Final Score:** `~0.1` :material-close:

    *Semantically related but lexically different.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">⚡</span>
<strong>Fast & Deterministic</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">No LLM calls needed. Instant, reproducible results ideal for CI/CD pipelines.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📊</span>
<strong>Industry Standard</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Widely used in NLP research for translation and summarization evaluation.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔢</span>
<strong>N-gram Precision</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Captures phrase-level similarity, not just word overlap.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Sentence BLEU** = How much does the candidate text overlap with the reference at the n-gram level?

    - **Use it when:** Fast, deterministic text similarity is needed
    - **Score interpretation:** Higher = more n-gram overlap with reference
    - **Key config:** `n_grams` controls phrase length (default 4)

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.SentenceBleu`](../../reference/metrics.md#sentence-bleu)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Levenshtein Ratio](./levenshtein_ratio.md) · Exact String Match · Contains Match

</div>
