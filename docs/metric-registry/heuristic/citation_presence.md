# Citation Presence

<div style="border-left: 4px solid #f59e0b; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Verify responses include properly formatted citations</strong><br>
<span class="badge" style="margin-top: 0.5rem; background: #f59e0b;">Heuristic</span>
<span class="badge" style="background: #667eea;">Knowledge</span>
<span class="badge" style="background: #06b6d4;">Multi-Turn</span>
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
<code>actual_output</code><br>
<small style="color: var(--md-text-muted);">Optional: conversation</small>
</div>

</div>

!!! abstract "What It Measures"
    Citation Presence evaluates whether AI responses include **properly formatted citations**—URLs, DOIs, or academic references. It supports both single-turn responses and multi-turn conversations.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Citations present (in at least one message) |
    | **0.0** | :material-close: No citations found |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Requiring sourced responses</li>
<li>Building research assistants</li>
<li>Enforcing citation policies</li>
<li>Validating knowledge retrieval</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Citations aren't required</li>
<li>Checking citation accuracy (use Faithfulness)</li>
<li>Creative/generative tasks</li>
<li>Simple Q&A without sources</li>
</ul>
</div>

</div>

!!! tip "Citation Presence vs Faithfulness"
    **Citation Presence** checks: *"Are citations included?"*
    **[Faithfulness](../composite/faithfulness.md)** checks: *"Is the content accurate to the source?"*

    Use Citation Presence for format compliance; use Faithfulness for content verification.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric extracts citations using regex patterns and evaluates based on the configured mode.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Input"]
            A[Response Text]
            B[Mode Setting]
        end

        subgraph EXTRACT["🔍 Step 1: Extract Citations"]
            C[Run citation patterns]
            D1["HTTP/HTTPS URLs"]
            D2["DOI references"]
            D3["Academic citations"]
        end

        subgraph EVALUATE["⚖️ Step 2: Mode-Based Evaluation"]
            E{Mode?}
            F["any_citation: Any URL/DOI found?"]
            G["resource_section: Section with citations?"]
        end

        subgraph OUTPUT["📊 Result"]
            H["1.0 = Pass"]
            I["0.0 = Fail"]
        end

        A & B --> C
        C --> D1 & D2 & D3
        D1 & D2 & D3 --> E
        E -->|any_citation| F
        E -->|resource_section| G
        F & G -->|Yes| H
        F & G -->|No| I

        style INPUT stroke:#f59e0b,stroke-width:2px
        style EXTRACT stroke:#3b82f6,stroke-width:2px
        style EVALUATE stroke:#8b5cf6,stroke-width:2px
        style OUTPUT stroke:#10b981,stroke-width:2px
    ```

=== ":material-link-variant: Detected Citation Formats"

    | Format | Pattern | Example |
    |--------|---------|---------|
    | **HTTP/HTTPS URLs** | `https?://...` | `https://docs.python.org/3/` |
    | **WWW URLs** | `www.domain.com` | `www.wikipedia.org` |
    | **DOI References** | `doi:10.xxxx/...` | `doi:10.1000/xyz123` |
    | **Academic** | `(Author, Year)` | `(Smith et al., 2023)` |

=== ":material-cog-outline: Evaluation Modes"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">any_citation</strong> (default)
    <br><small>Pass if any citation appears anywhere in the response.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #8b5cf6; padding-left: 1rem;">
    <strong style="color: #8b5cf6;">resource_section</strong>
    <br><small>Pass only if citations appear in a dedicated Resources/References section.</small>
    </div>

    </div>

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `mode` | `str` | `any_citation` | Evaluation mode: `any_citation` or `resource_section` |
    | `strict` | `bool` | `False` | If True, validates URLs are live |
    | `use_semantic_search` | `bool` | `False` | Use embeddings for fallback detection |
    | `embed_model` | `EmbeddingRunnable` | `None` | Embedding model (required if semantic search enabled) |
    | `resource_similarity_threshold` | `float` | `0.8` | Threshold for semantic matching |
    | `custom_resource_phrases` | `List[str]` | `None` | Custom phrases to identify resource sections |

    !!! info "Strict Mode"
        When `strict=True`, the metric validates that URLs are live by making HEAD requests. This ensures citations point to actual resources but adds latency.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import CitationPresence
    from axion.dataset import DatasetItem

    metric = CitationPresence()

    item = DatasetItem(
        actual_output="Python is a programming language. Learn more at https://python.org",
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - URL citation found
    ```

=== ":material-close: No Citations"

    ```python
    from axion.metrics import CitationPresence

    metric = CitationPresence()

    item = DatasetItem(
        actual_output="Python is a great programming language for beginners.",
    )

    result = await metric.execute(item)
    print(result.score)  # 0.0 - no citations
    print(result.explanation)
    # "Mode: any_citation. FAILURE: No assistant message satisfied the citation requirement."
    ```

=== ":material-tune-variant: Resource Section Mode"

    ```python
    from axion.metrics import CitationPresence

    # Require citations in a dedicated section
    metric = CitationPresence(mode='resource_section')

    item = DatasetItem(
        actual_output="""
        Python is versatile and beginner-friendly.

        For More Information:
        - https://docs.python.org/3/
        - https://realpython.com/
        """,
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - resource section with citations
    ```

=== ":material-chat-processing: Multi-Turn Conversation"

    ```python
    from axion.metrics import CitationPresence
    from axion._core.schema import Conversation, HumanMessage, AIMessage

    metric = CitationPresence()

    item = DatasetItem(
        actual_output="",  # Will check conversation instead
        conversation=Conversation(messages=[
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
            HumanMessage(content="Where can I learn more?"),
            AIMessage(content="Check out https://python.org and https://realpython.com"),
        ]),
    )

    result = await metric.execute(item)
    print(result.score)  # 1.0 - citation in second AI message
    print(result.signals.messages_with_citations)  # [3] (index of 2nd AI message)
    ```

---

## Metric Diagnostics

Every evaluation is **fully interpretable**. Access detailed diagnostic results via `result.signals`.

```python
result = await metric.execute(item)
print(result.pretty())      # Human-readable summary
result.signals              # Full diagnostic breakdown
```

<details markdown="1">
<summary><strong>📊 CitationPresenceResult Structure</strong></summary>

```python
CitationPresenceResult(
{
    "passes_presence_check": True,
    "total_assistant_messages": 2,
    "messages_with_citations": [3]  # 0-indexed message positions
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `passes_presence_check` | `bool` | Whether citation requirement was met |
| `total_assistant_messages` | `int` | Number of AI messages evaluated |
| `messages_with_citations` | `List[int]` | Indices of messages with valid citations |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: URL Citation (Score: 1.0)</strong></summary>

!!! success "HTTP URL Found"

    **Output:**
    > "Machine learning is a subset of AI. See https://scikit-learn.org for tutorials."

    **Citations Detected:** `https://scikit-learn.org`

    **Final Score:** `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>✅ Scenario 2: Academic Citation (Score: 1.0)</strong></summary>

!!! success "Author-Year Format"

    **Output:**
    > "Attention mechanisms transformed NLP (Vaswani et al., 2017)."

    **Citations Detected:** `(Vaswani et al., 2017)`

    **Final Score:** `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: No Citations (Score: 0.0)</strong></summary>

!!! failure "Missing Citations"

    **Output:**
    > "Deep learning uses neural networks with multiple layers to process data."

    **Citations Detected:** None

    **Final Score:** `0.0` :material-close:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 4: Resource Section Required</strong></summary>

!!! warning "Wrong Mode"

    **Mode:** `resource_section`

    **Output:**
    > "Python documentation is at https://python.org which explains everything."

    **Analysis:** URL exists but not in a resource section.

    **Final Score:** `0.0` :material-alert:

    *Switch to `any_citation` mode or add a Resources section.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">📚</span>
<strong>Source Attribution</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensure AI outputs provide proper attribution to sources.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🎓</span>
<strong>Research Quality</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Enforce citation standards for academic or research applications.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">✅</span>
<strong>Policy Compliance</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Verify responses meet organizational citation requirements.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Citation Presence** = Does the response include citations?

    - **Use it when:** Requiring sourced responses or research assistants
    - **Score interpretation:** 1.0 = citations found, 0.0 = none
    - **Key config:** `mode` determines where citations must appear

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.CitationPresence`](../../reference/metrics.md#citation-presence)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Faithfulness](../composite/faithfulness.md) · Contextual Relevancy · Answer Relevancy

</div>
