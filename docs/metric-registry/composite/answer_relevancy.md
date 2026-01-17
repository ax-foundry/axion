# Answer Relevancy

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluate how well an AI response addresses the input query</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Knowledge</span>
<span class="badge" style="background: #6B7A3A;">Single Turn</span>
<span class="badge" style="background: #8b5cf6;">Multi-Turn</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Ratio of relevant statements</small>
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
<code>query</code> <code>actual_output</code><br>
<small style="color: var(--md-text-muted);">Optional: <code>conversation</code></small>
</div>

</div>

!!! abstract "What It Measures"
    Answer Relevancy evaluates whether **each statement** in the AI's response directly addresses the user's query. Unlike Faithfulness (which checks factual grounding), this metric measures **topical alignment**—did the AI stay on topic or go off on tangents?

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Every statement directly addresses the query |
    | **0.7+** | :material-check: Mostly relevant with minor tangents |
    | **0.5** | :material-alert: Threshold—mix of relevant and off-topic content |
    | **< 0.5** | :material-close: Significant off-topic or irrelevant content |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Q&A systems & chatbots</li>
<li>Customer support agents</li>
<li>Search result evaluation</li>
<li>Any query-response system</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Open-ended conversations</li>
<li>Exploratory discussions</li>
<li>No clear query/question</li>
<li>Tasks where tangents are valuable</li>
</ul>
</div>

</div>

!!! tip "See Also: Faithfulness"
    **Answer Relevancy** checks if statements *address the user's query* (topical alignment).
    **[Faithfulness](./faithfulness.md)** checks if claims are *grounded in the source context* (factual accuracy).

    Use both together for comprehensive RAG evaluation.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric uses an **Evaluator LLM** to decompose the response into atomic statements, then judge each statement's relevance to the query.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[Query]
            B[AI Response]
        end

        subgraph EXTRACT["🔍 Step 1: Statement Extraction"]
            C[StatementExtractor LLM]
            D["Atomic Statements<br/><small>Self-contained facts</small>"]
        end

        subgraph JUDGE["⚖️ Step 2: Relevancy Judgment"]
            E[RelevancyJudge LLM]
            F["Verdict per Statement<br/><small>yes / no / idk</small>"]
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

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style EXTRACT stroke:#3b82f6,stroke-width:2px
        style JUDGE stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style I fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Verdict System"

    Each extracted statement receives a **verdict** indicating its relevance to the query.

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ YES</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1.0</div>
    <br><small>Statement <strong>directly addresses</strong> the query. Clearly relevant.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #6b7280; padding-left: 1rem;">
    <strong style="color: #6b7280;">❓ IDK</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #6b7280;">1.0*</div>
    <br><small>Ambiguous relevance. <strong>*Configurable</strong>—can be 0.0 with <code>penalize_ambiguity=True</code></small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ NO</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0.0</div>
    <br><small>Statement is <strong>off-topic</strong> or doesn't address the query at all.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = (yes_count + idk_count*) / total_statements

        * idk_count included only if penalize_ambiguity=False (default)
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `relevancy_mode` | `'strict'` \| `'task'` | `'task'` | **strict**: Only direct answers count. **task**: Helpful related info also counts |
    | `penalize_ambiguity` | `bool` | `False` | When `True`, ambiguous (`idk`) verdicts score 0.0 instead of 1.0 |
    | `multi_turn_strategy` | `'last_turn'` \| `'all_turns'` | `'last_turn'` | How to evaluate conversations |
    | `mode` | `EvaluationMode` | `GRANULAR` | Evaluation detail level |

    !!! info "Relevancy Modes"
        - **`task` mode** (default): More lenient—counts closely related, helpful information as relevant
        - **`strict` mode**: Only statements that directly answer the question count as relevant

=== ":material-tune-variant: Strict Configuration"

    For high-precision evaluation where tangential information should be penalized:

    ```python
    from axion.metrics import AnswerRelevancy

    # Strict evaluation: only direct answers, penalize ambiguity
    metric = AnswerRelevancy(
        relevancy_mode='strict',
        penalize_ambiguity=True
    )
    ```

=== ":material-chat-processing: Multi-Turn"

    For conversational AI evaluation:

    ```python
    from axion.metrics import AnswerRelevancy

    # Evaluate all turns in a conversation
    metric = AnswerRelevancy(
        multi_turn_strategy='all_turns'  # or 'last_turn' (default)
    )
    ```

    - **`last_turn`**: Only evaluates the final Human→AI exchange
    - **`all_turns`**: Evaluates every turn and aggregates via micro-averaging

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.metrics import AnswerRelevancy
    from axion.dataset import DatasetItem

    # Initialize with defaults (task mode, lenient)
    metric = AnswerRelevancy()

    item = DatasetItem(
        query="What features does this laptop have?",
        actual_output=(
            "The laptop has a 15-inch Retina display and 16GB of RAM. "
            "It also comes with a 1-year warranty. "
            "Our company was founded in 2010."
        ),
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score ~0.67: warranty is borderline, founding year is irrelevant
    ```

=== ":material-shield-alert: Strict Mode"

    ```python hl_lines="4 5"
    from axion.metrics import AnswerRelevancy

    # Strict: only direct answers count
    metric = AnswerRelevancy(
        relevancy_mode='strict',
        penalize_ambiguity=True
    )

    # Now only "15-inch display" and "16GB RAM" statements count
    # Warranty = ambiguous (0.0), founding year = no (0.0)
    ```

=== ":material-chat-processing: Multi-Turn Conversation"

    ```python
    from axion.metrics import AnswerRelevancy
    from axion.dataset import DatasetItem, MultiTurnConversation
    from axion.schema import HumanMessage, AIMessage

    conversation = MultiTurnConversation(messages=[
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language known for readability."),
        HumanMessage(content="What are its main uses?"),
        AIMessage(content="Python is used for web dev, data science, and automation."),
    ])

    metric = AnswerRelevancy(multi_turn_strategy='all_turns')
    item = DatasetItem(conversation=conversation)

    result = await metric.execute(item)
    print(f"Evaluated {result.signals.evaluated_turns_count} turns")
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
<summary><strong>📊 AnswerRelevancyResult Structure</strong></summary>

```python
AnswerRelevancyResult(
{
    "overall_score": 1.0,
    "explanation": "The score is 1.0 because the response fully and accurately explains...",
    "relevant_statements_count": 2,
    "irrelevant_statements_count": 0,
    "ambiguous_statements_count": 0,
    "total_statements_count": 2,
    "statement_breakdown": [
        {
            "statement": "The infield fly rule prevents the defense from dropping a fly ball.",
            "verdict": "yes",
            "is_relevant": true,
            "turn_index": 0
        },
        {
            "statement": "The rule prevents an easy double play when runners are on base.",
            "verdict": "yes",
            "is_relevant": true,
            "turn_index": 0
        }
    ],
    "evaluated_turns_count": 1
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | `float` | The 0-1 relevancy score |
| `explanation` | `str` | Human-readable summary of why the score was given |
| `relevant_statements_count` | `int` | Count of `yes` verdicts |
| `irrelevant_statements_count` | `int` | Count of `no` verdicts |
| `ambiguous_statements_count` | `int` | Count of `idk` verdicts |
| `total_statements_count` | `int` | Total statements extracted |
| `statement_breakdown` | `List` | Per-statement verdict details |
| `evaluated_turns_count` | `int` | Number of conversation turns evaluated |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Perfect Relevancy (Score: 1.0)</strong></summary>

!!! success "All Statements Relevant"

    **Query:**
    > "What are the health benefits of green tea?"

    **AI Response:**
    > "Green tea contains antioxidants that may reduce inflammation. It also has caffeine which can improve alertness."

    **Analysis:**

    | Statement | Verdict | Score |
    |-----------|---------|-------|
    | Green tea contains antioxidants that may reduce inflammation | yes | 1.0 |
    | Green tea has caffeine which can improve alertness | yes | 1.0 |

    **Final Score:** `2 / 2 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Partial Relevancy (Score: 0.67)</strong></summary>

!!! warning "Mixed Verdicts"

    **Query:**
    > "What features does this laptop have?"

    **AI Response:**
    > "The laptop has a 15-inch display. It has 16GB RAM. Our company has excellent customer service."

    **Analysis:**

    | Statement | Verdict | Score |
    |-----------|---------|-------|
    | The laptop has a 15-inch display | yes | 1.0 |
    | The laptop has 16GB RAM | yes | 1.0 |
    | Our company has excellent customer service | no | 0.0 |

    **Final Score:** `2 / 3 = 0.67` :material-alert:

    *The customer service statement doesn't address laptop features.*

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Poor Relevancy (Score: 0.25)</strong></summary>

!!! failure "Mostly Off-Topic"

    **Query:**
    > "How do I reset my password?"

    **AI Response:**
    > "Our platform uses industry-standard encryption. We were founded in 2015. Password resets can be done via email. We have offices in 3 countries."

    **Analysis:**

    | Statement | Verdict | Score |
    |-----------|---------|-------|
    | Our platform uses industry-standard encryption | no | 0.0 |
    | We were founded in 2015 | no | 0.0 |
    | Password resets can be done via email | yes | 1.0 |
    | We have offices in 3 countries | no | 0.0 |

    **Final Score:** `1 / 4 = 0.25` :material-close:

    *Only one statement actually answers the question.*

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>User Experience</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Users expect direct answers. Off-topic responses frustrate users and reduce trust in your AI system.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">💬</span>
<strong>Conversation Quality</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">For chatbots and assistants, staying on topic is crucial. Tangential responses break conversational flow.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Debug Generation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Identifies when your model goes off-topic—separate from retrieval issues (Faithfulness) or factual errors.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Answer Relevancy** = Does the AI's response actually address what the user asked?

    - **Use it when:** You need to ensure responses stay on topic
    - **Score interpretation:** Higher = more statements directly address the query
    - **Key config:** Use `relevancy_mode='strict'` for precision, `'task'` for lenient evaluation

<div class="grid cards" markdown>

- :material-api: **API Reference**

    [:octicons-arrow-right-24: `axion.metrics.AnswerRelevancy`](../../reference/metrics.md#answer-relevancy)

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Faithfulness](./faithfulness.md) · Answer Completeness · Context Precision

</div>
