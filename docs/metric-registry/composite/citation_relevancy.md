# Citation Relevancy

<div style="border-left: 4px solid #1E3A5F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Measure the quality and relevance of citations in AI responses</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Knowledge</span>
<span class="badge" style="background: #8b5cf6;">Multi-Turn</span>
<span class="badge" style="background: #06b6d4;">Citation</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Ratio of relevant citations</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.8</code><br>
<small style="color: var(--md-text-muted);">High bar for citation quality</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>query</code> <code>actual_output</code><br>
<small style="color: var(--md-text-muted);">Optional: <code>conversation</code></small>
</div>

</div>

!!! abstract "What It Measures"
    Citation Relevancy evaluates whether the **citations included in an AI response** are actually relevant to the user's query. It extracts citations using pattern matching, then judges each citation's relevance using an LLM. Essential for research assistants and fact-checking systems.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All citations directly relevant to query |
    | **0.8+** | :material-check: Most citations relevant, minor tangents |
    | **0.5** | :material-alert: Mixed relevance—some helpful, some off-topic |
    | **< 0.5** | :material-close: Mostly irrelevant or unrelated citations |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Building research assistants</li>
<li>Fact-checking systems</li>
<li>Academic writing tools</li>
<li>Any system that generates citations</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Responses don't include citations</li>
<li>Citation format is non-standard</li>
<li>Internal linking (not external sources)</li>
<li>Pure conversational AI</li>
</ul>
</div>

</div>

!!! tip "See Also: Faithfulness"
    **Citation Relevancy** checks if *cited sources* are relevant to the query.
    **[Faithfulness](./faithfulness.md)** checks if *claims* are grounded in retrieved context.

    Use Citation Relevancy for output validation; use Faithfulness for RAG grounding.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric uses regex-based extraction followed by LLM-based relevance judgment.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[AI Response with Citations]
        end

        subgraph EXTRACT["🔍 Step 1: Citation Extraction"]
            C[Regex Pattern Matching]
            D["Extracted Citations"]
        end

        subgraph JUDGE["⚖️ Step 2: Relevance Judgment"]
            E[CitationRelevanceJudge LLM]
            F["Verdict per Citation"]
        end

        subgraph SCORE["📊 Step 3: Scoring"]
            G["Count Relevant"]
            H["Calculate Ratio"]
            I["Final Score"]
        end

        A & B --> C
        C --> D
        D --> E
        A --> E
        E --> F
        F --> G
        G --> H
        H --> I

        style INPUT stroke:#1E3A5F,stroke-width:2px
        style EXTRACT stroke:#3b82f6,stroke-width:2px
        style JUDGE stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style I fill:#1E3A5F,stroke:#0F2440,stroke-width:3px,color:#fff
    ```

=== ":material-link-variant: Supported Citation Formats"

    The metric extracts citations using multiple regex patterns:

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">📝 Markdown Links</strong>
    <br><small><code>[Title](https://example.com)</code></small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">🔗 HTTP/HTTPS URLs</strong>
    <br><small><code>https://example.com/article</code></small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">🌐 WWW URLs</strong>
    <br><small><code>www.example.com/page</code></small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #8b5cf6; padding-left: 1rem;">
    <strong style="color: #8b5cf6;">📚 DOI Patterns</strong>
    <br><small><code>doi:10.1234/example</code></small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ec4899; padding-left: 1rem;">
    <strong style="color: #ec4899;">🎓 Academic Format</strong>
    <br><small><code>(Smith et al., 2023)</code> or <code>(Smith, 2023)</code></small>
    </div>

    </div>

=== ":material-scale-balance: Verdict System"

    Each citation receives a binary relevance verdict.

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ RELEVANT</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1</div>
    <br><small>Citation directly supports answering the user's query.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ IRRELEVANT</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0</div>
    <br><small>Citation is off-topic or doesn't help answer the question.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = relevant_citations / total_citations
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `multi_turn_strategy` | `'last_turn'` \| `'all_turns'` | `'last_turn'` | How to evaluate conversations |
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level |

    !!! info "Multi-Turn Support"
        In multi-turn conversations, citations are associated with their corresponding query context:

        - **`last_turn`**: Only evaluates citations in the final response
        - **`all_turns`**: Evaluates citations across all turns, matching each to its original query

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import CitationRelevancy
    from axion.dataset import DatasetItem

    metric = CitationRelevancy()

    item = DatasetItem(
        query="What are the health benefits of green tea?",
        actual_output="""
        Green tea has numerous health benefits:

        1. Rich in antioxidants [Source](https://healthline.com/green-tea-benefits)
        2. May improve brain function (Smith et al., 2020)
        3. Great for parties! [Party Guide](https://party-planning.com)
        """,
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 0.67 (2 of 3 citations relevant)
    ```

=== ":material-chat-processing: Multi-Turn"

    ```python
    from axion.metrics import CitationRelevancy
    from axion.dataset import DatasetItem, MultiTurnConversation
    from axion.schema import HumanMessage, AIMessage

    conversation = MultiTurnConversation(messages=[
        HumanMessage(content="What causes climate change?"),
        AIMessage(content="Climate change is primarily caused by greenhouse gases. [IPCC Report](https://ipcc.ch/report)"),
        HumanMessage(content="How can I reduce my carbon footprint?"),
        AIMessage(content="You can reduce emissions by using public transport. [EPA Guide](https://epa.gov/guide)"),
    ])

    metric = CitationRelevancy(multi_turn_strategy='all_turns')
    item = DatasetItem(conversation=conversation)

    result = await metric.execute(item)
    print(f"Evaluated {result.signals.total_citations} citations across turns")
    ```

=== ":material-cog-outline: With Runner"

    ```python
    from axion.metrics import CitationRelevancy
    from axion.runners import MetricRunner

    metric = CitationRelevancy()
    runner = MetricRunner(metrics=[metric])
    results = await runner.run(dataset)

    for item_result in results:
        print(f"Score: {item_result.score}")
        print(f"Relevant: {item_result.signals.relevant_citations_count}/{item_result.signals.total_citations}")
        for citation in item_result.signals.citation_breakdown:
            status = "✅" if citation.relevance_verdict else "❌"
            print(f"  {status} {citation.citation_text[:50]}...")
    ```

---

## Metric Diagnostics

Every evaluation is **fully interpretable**. Access detailed diagnostic results via `result.signals` to understand exactly why a score was given—no black boxes.

```python
result = await metric.execute(item)
print(result.pretty())      # Human-readable summary
result.signals              # Full diagnostic breakdown
```

<details markdown="1">
<summary><strong>📊 CitationRelevancyResult Structure</strong></summary>

```python
CitationRelevancyResult(
{
    "relevance_score": 0.67,
    "total_citations": 3,
    "relevant_citations_count": 2,
    "irrelevant_citations_count": 1,
    "citation_breakdown": [
        {
            "citation_text": "[Source](https://healthline.com/green-tea-benefits)",
            "relevance_verdict": true,
            "relevance_reason": "Directly addresses health benefits of green tea",
            "turn_index": 0,
            "original_query": "What are the health benefits of green tea?"
        },
        {
            "citation_text": "(Smith et al., 2020)",
            "relevance_verdict": true,
            "relevance_reason": "Academic source on tea and brain function",
            "turn_index": 0,
            "original_query": "What are the health benefits of green tea?"
        },
        {
            "citation_text": "[Party Guide](https://party-planning.com)",
            "relevance_verdict": false,
            "relevance_reason": "Party planning is unrelated to health benefits",
            "turn_index": 0,
            "original_query": "What are the health benefits of green tea?"
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `relevance_score` | `float` | Ratio of relevant citations (0.0-1.0) |
| `total_citations` | `int` | Total citations extracted |
| `relevant_citations_count` | `int` | Count of relevant citations |
| `irrelevant_citations_count` | `int` | Count of irrelevant citations |
| `citation_breakdown` | `List` | Per-citation verdict details |

### Citation Breakdown Fields

| Field | Type | Description |
|-------|------|-------------|
| `citation_text` | `str` | The extracted citation |
| `relevance_verdict` | `bool` | Whether citation is relevant |
| `relevance_reason` | `str` | Explanation for verdict |
| `turn_index` | `int` | Conversation turn (for multi-turn) |
| `original_query` | `str` | Query that prompted this citation |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: All Citations Relevant (Score: 1.0)</strong></summary>

!!! success "High-Quality Citations"

    **Query:**
    > "What are effective treatments for type 2 diabetes?"

    **AI Response:**
    > "Type 2 diabetes can be managed through:
    >
    > 1. Lifestyle changes including diet and exercise [ADA Guidelines](https://diabetes.org/guidelines)
    > 2. Metformin as first-line medication (Johnson et al., 2021)
    > 3. Regular blood glucose monitoring [CDC Recommendations](https://cdc.gov/diabetes)"

    **Analysis:**

    | Citation | Verdict | Reason |
    |----------|---------|--------|
    | ADA Guidelines | ✅ | Authoritative diabetes organization |
    | Johnson et al., 2021 | ✅ | Academic source on medication |
    | CDC Recommendations | ✅ | Government health authority |

    **Final Score:** `3 / 3 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Mixed Relevance (Score: 0.5)</strong></summary>

!!! warning "Some Citations Off-Topic"

    **Query:**
    > "How does machine learning work?"

    **AI Response:**
    > "Machine learning uses algorithms to learn from data:
    >
    > - Neural networks mimic brain structure [ML Textbook](https://ml-intro.com)
    > - Training requires large datasets [Data Science Blog](https://datascience.com)
    > - My favorite coffee shop uses ML! [Best Coffee](https://coffee-reviews.com)
    > - Check out this unrelated video [Cat Video](https://youtube.com/cats)"

    **Analysis:**

    | Citation | Verdict | Reason |
    |----------|---------|--------|
    | ML Textbook | ✅ | Directly about machine learning |
    | Data Science Blog | ✅ | Relevant to ML data requirements |
    | Best Coffee | ❌ | Coffee reviews unrelated to ML |
    | Cat Video | ❌ | Entertainment, not educational |

    **Final Score:** `2 / 4 = 0.5` :material-alert:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Mostly Irrelevant (Score: 0.25)</strong></summary>

!!! failure "Citation Spam"

    **Query:**
    > "What is the capital of France?"

    **AI Response:**
    > "Paris is the capital of France. Here are some links:
    >
    > - [My Portfolio](https://myportfolio.com)
    > - [Buy Cheap Flights](https://flights.com)
    > - [Wikipedia - France](https://wikipedia.org/wiki/France)
    > - [Dating Site](https://dating.com)"

    **Analysis:**

    | Citation | Verdict | Reason |
    |----------|---------|--------|
    | My Portfolio | ❌ | Self-promotion, irrelevant |
    | Buy Cheap Flights | ❌ | Commercial, off-topic |
    | Wikipedia - France | ✅ | Relevant geographic source |
    | Dating Site | ❌ | Completely unrelated |

    **Final Score:** `1 / 4 = 0.25` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Source Quality</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensures AI-generated citations actually support the response, not random links or self-promotion.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🎓</span>
<strong>Research Integrity</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Critical for academic and research tools where citations must be relevant and authoritative.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">✅</span>
<strong>User Trust</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Users expect citations to be helpful. Irrelevant citations damage credibility and waste time.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Citation Relevancy** = Are the citations actually relevant to the user's question?

    - **Use it when:** AI responses include citations that need quality validation
    - **Score interpretation:** Higher = more citations are relevant
    - **Key feature:** Supports multiple citation formats (URLs, DOIs, academic)

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.CitationRelevancy`](../../reference/metrics.md#citation-relevancy)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Faithfulness](./faithfulness.md) · Answer Relevancy · Factual Accuracy

</div>
