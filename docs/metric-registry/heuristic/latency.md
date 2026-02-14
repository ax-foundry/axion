# Latency

<div style="border-left: 4px solid #f59e0b; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Measure and evaluate execution time performance</strong><br>
<span class="badge" style="margin-top: 0.5rem; background: #f59e0b;">Heuristic</span>
<span class="badge" style="background: #0F2440;">Single Turn</span>
<span class="badge" style="background: #8b5cf6;">Performance</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">∞</code><br>
<small style="color: var(--md-text-muted);">Seconds (or normalized 0-1)</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">5.0s</code><br>
<small style="color: var(--md-text-muted);">Target latency</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>latency</code><br>
<small style="color: var(--md-text-muted);">Execution time in seconds</small>
</div>

</div>

!!! abstract "What It Measures"
    The Latency metric evaluates **execution time performance**. It can return raw latency values or normalize them to a 0-1 scale using various decay functions.

    | Mode | Score Interpretation |
    |------|---------------------|
    | **Raw** | Actual latency in seconds (lower is better) |
    | **Normalized** | 0.0-1.0 where 1.0 = instant, 0.0 = very slow |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Monitoring response times</li>
<li>SLA compliance checking</li>
<li>Performance regression testing</li>
<li>Comparing model latencies</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Quality metrics are more important</li>
<li>Latency isn't being tracked</li>
<li>Network conditions are highly variable</li>
<li>Cold start effects dominate</li>
</ul>
</div>

</div>

!!! tip "Inverse Scoring"
    Unlike most metrics where higher is better, **lower latency is better**. The metric is marked as `inverse_scoring_metric = True` for proper aggregation.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric reads the latency value and optionally normalizes it.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Input"]
            A[Latency Value]
            B[Threshold Setting]
        end

        subgraph PROCESS["🔍 Processing"]
            C{Normalize?}
            D[Return raw latency]
            E[Apply normalization function]
        end

        subgraph OUTPUT["📊 Result"]
            F["Raw: seconds"]
            G["Normalized: 0.0-1.0"]
        end

        A & B --> C
        C -->|No| D
        C -->|Yes| E
        D --> F
        E --> G

        style INPUT stroke:#f59e0b,stroke-width:2px
        style PROCESS stroke:#3b82f6,stroke-width:2px
        style OUTPUT stroke:#10b981,stroke-width:2px
    ```

=== ":material-calculator: Normalization Methods"

    Four normalization methods convert raw latency to a 0-1 score:

    | Method | Formula | Characteristics |
    |--------|---------|-----------------|
    | **exponential** | `exp(-latency/threshold)` | Smooth decay, never reaches 0 |
    | **sigmoid** | `1/(1 + exp((latency-threshold)/scale))` | S-curve centered at threshold |
    | **reciprocal** | `threshold/(threshold + latency)` | Hyperbolic decay |
    | **linear** | `max(0, 1 - latency/threshold)` | Linear drop to 0 |

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">📈 Exponential</strong>
    <br><small>Smooth decay. At threshold: ~0.37</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #8b5cf6; padding-left: 1rem;">
    <strong style="color: #8b5cf6;">📉 Sigmoid</strong>
    <br><small>S-curve. At threshold: 0.5</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">📊 Reciprocal</strong>
    <br><small>Hyperbolic. At threshold: 0.5</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">📐 Linear</strong>
    <br><small>Simple. At threshold: 0.0</small>
    </div>

    </div>

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `threshold` | `float` | `5.0` | Target latency in seconds |
    | `normalize` | `bool` | `False` | Whether to normalize to 0-1 range |
    | `normalization_method` | `str` | `exponential` | Method: exponential, sigmoid, reciprocal, linear |

    !!! info "Choosing a Normalization Method"
        - **exponential**: Good default, smooth decay
        - **sigmoid**: Hard cutoff around threshold
        - **reciprocal**: Balanced decay, never hits 0
        - **linear**: Simple, goes to 0 at threshold

---

## Code Examples

=== ":material-play: Basic Usage (Raw)"

    ```python
    from axion.metrics import Latency
    from axion.dataset import DatasetItem

    metric = Latency(threshold=2.0)

    item = DatasetItem(
        query="What is the capital of France?",
        actual_output="Paris",
        latency=1.5,  # 1.5 seconds
    )

    result = await metric.execute(item)
    print(result.score)  # 1.5 (raw latency)
    print(result.explanation)
    # "Raw latency: 1.500s, below threshold (2.0s)."
    ```

=== ":material-tune-variant: Normalized Scoring"

    ```python
    from axion.metrics import Latency

    # Exponential normalization
    metric = Latency(
        threshold=2.0,
        normalize=True,
        normalization_method='exponential'
    )

    item = DatasetItem(latency=1.0)  # 1 second
    result = await metric.execute(item)
    print(f"{result.score:.3f}")  # ~0.607 (exp(-1/2))

    # Linear normalization
    metric_linear = Latency(
        threshold=2.0,
        normalize=True,
        normalization_method='linear'
    )
    result_linear = await metric_linear.execute(item)
    print(f"{result_linear.score:.3f}")  # 0.5 (1 - 1/2)
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import Latency
    from axion.runners import MetricRunner

    metric = Latency(threshold=3.0, normalize=True)
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Latency score: {item_result.score:.2f}")
        print(f"  {item_result.explanation}")
    ```

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Excellent Performance</strong></summary>

!!! success "Below Half Threshold"

    **Threshold:** 5.0s

    **Latency:** 2.0s (40% of threshold)

    **Raw Score:** `2.0`

    **Normalized (exponential):** `~0.67` :material-check:

    **Explanation:** "Latency: 2.000s. Normalized score: 0.670 (threshold: 5.0s, method: exponential). Performance: excellent."

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: At Threshold</strong></summary>

!!! warning "Exactly at Target"

    **Threshold:** 5.0s

    **Latency:** 5.0s

    **Raw Score:** `5.0`

    **Normalized Scores:**

    | Method | Score |
    |--------|-------|
    | exponential | 0.37 |
    | sigmoid | 0.50 |
    | reciprocal | 0.50 |
    | linear | 0.00 |

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Poor Performance</strong></summary>

!!! failure "Above Threshold"

    **Threshold:** 2.0s

    **Latency:** 8.0s (4x threshold)

    **Raw Score:** `8.0`

    **Normalized (exponential):** `~0.02` :material-close:

    **Explanation:** "Latency: 8.000s. Normalized score: 0.018 (threshold: 2.0s, method: exponential). Performance: poor."

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">⏱️</span>
<strong>SLA Compliance</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Track response times against service level agreements.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📈</span>
<strong>Performance Monitoring</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Detect regressions and optimize slow endpoints.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">⚖️</span>
<strong>Quality vs Speed</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Balance model quality against response time requirements.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Latency** = How fast was the response?

    - **Use it when:** Monitoring performance or SLA compliance
    - **Score interpretation:** Raw (seconds) or normalized (0-1, higher = faster)
    - **Key config:** `threshold` sets target, `normalize` enables 0-1 scoring

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.Latency`](../../reference/metrics.md#latency)

- :material-link-variant: **Related Concepts**

    [:octicons-arrow-right-24: MetricRunner](../../deep-dives/runners/metric-runner.md) · Evaluation Strategies

</div>
