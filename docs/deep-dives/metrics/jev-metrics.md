---
icon: custom/cpu
---
# Building Jev-Backed Metrics

`JevMetric` scores an item by asking [TypeSafe Jev](https://typesafe.ai) a bundle of questions about it. Jev is a decision endpoint rather than a chat model: it answers a declared question with a calibrated probability, a label, or a rubric level — fast and cheap enough to run on every turn of a live conversation.

That is the reason it exists alongside the LLM-judge metrics. An LLM judge is the right tool for a nightly evaluation over a sampled dataset, and the wrong tool for a gate that has to fire before the next turn is sent.

## What You'll Learn

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Question Types</p>
<p class="rule-card__desc">Ask for a probability (<code>noul</code>), a label (<code>choice</code>), or a rubric level (<code>score</code>) &mdash; each answered with calibrated confidence.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">Declaring a Metric</p>
<p class="rule-card__desc">Subclass <code>JevMetric</code>, declare <code>questions</code> and a <code>state_template</code>. The call, the scoring, and the sub-metric explosion are handled for you.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">One Call, Not N</p>
<p class="rule-card__desc">Every question travels in a single request, because Jev bills on input tokens and ingests the shared state once per call.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">4</span>
<p class="rule-card__title">Per-Agent Calibration</p>
<p class="rule-card__desc">Thresholds on the class are defaults, not settings. Any instance can be handed different ones, so calibration is configuration rather than a subclass.</p>
</div>
</div>

## Overview

| | LLM judge (`BaseMetric`) | Jev (`JevMetric`) |
|---|---|---|
| Decides by | A chat model reasoning in free text | A decision endpoint answering a declared question |
| Output | Parsed from generated text | A probability, a label, or a rubric level |
| Run-to-run stability | Varies between identical calls | Effectively stable at the same input |
| Good for | Nuanced quality, rich critiques, offline depth | Per-turn gates, high-volume scoring, cheap screening |

Reach for a Jev metric when the question is narrow enough to state precisely and you need the answer on every item. Keep an LLM judge for the questions that need a paragraph of reasoning to answer.

## Configuration

Jev is not reached through the LLM registry, so it is configured on its own settings rather than through a provider:

| Setting | Environment variable | Default |
|---|---|---|
| `typesafe_api_key` | `TYPESAFE_API_KEY` | — (required) |
| `typesafe_base_url` | `TYPESAFE_BASE_URL` | `https://api.typesafe.ai/v1` |
| `typesafe_model` | `TYPESAFE_MODEL` | `jev-latest` |

```bash
export TYPESAFE_API_KEY='...'
```

Point `TYPESAFE_BASE_URL` at a gateway to route through one. Gateways generally rename the model, so set `TYPESAFE_MODEL` to whatever that route calls it.

!!! note "No LLM credentials needed"
    A `JevMetric` subclass sets `requires_llm = False`, so it resolves no model from the registry and configures no LLM client. Constructing one with `OPENAI_API_KEY` unset works fine.

## Question Types

Every question carries `instructions` — the question itself, stated as precisely as you can manage — plus whatever its type needs.

=== "noul"

    A calibrated probability that the statement holds. No criteria and no separate confidence: the number *is* the confidence.

    ```python
    from axion._handlers.jev.schema import NoulQuestion

    NoulQuestion(
        instructions='Is every factual claim in the answer supported by the tool output?'
    )
    ```

    Answers with `noul` in `[0, 1]`.

=== "choice"

    Pick one of a named set of labels. Each key is the label, each value describes when it applies.

    ```python
    from axion._handlers.jev.schema import ChoiceQuestion

    ChoiceQuestion(
        instructions='Which failure, if any, does the answer show?',
        criteria={
            'none': 'Every claim traces to the transcript or the tool output.',
            'fabricated': 'A claim appears nowhere in the transcript or the tool output.',
            'misread': 'A claim contradicts what the tool actually returned.',
        },
    )
    ```

    Answers with `choice` (the label), `confidence`, and `probabilities` over every label.

=== "score"

    Place the item on an ordered rubric. `criteria` is a list, lowest level first, and the order is load-bearing — the score is a probability-weighted expectation across the levels, so it comes back fractional.

    ```python
    from axion._handlers.jev.schema import ScoreQuestion

    ScoreQuestion(
        instructions='How severe is the ungrounded content?',
        criteria=[
            'None.',
            'Cosmetic.',
            'Materially misleading.',
            'Would change a decision.',
        ],
    )
    ```

    Answers with `score` (e.g. `2.37` on a four-level rubric), `confidence`, `legend`, and `probabilities`.

## Declaring a Metric

A subclass declares the questions it wants answered and how the item renders into the state Jev reads:

```python
from axion._handlers.jev.schema import ChoiceQuestion, NoulQuestion
from axion.metrics.base import metric
from axion.metrics.jev import JevMetric, JevRubric


@metric(
    key='grounding',
    name='Grounding',
    description='Is the answer supported by what the tools returned?',
    required_fields=['actual_output'],
    optional_fields=['retrieved_content'],
    default_threshold=0.7,
)
class Grounding(JevMetric):
    state_template = 'Answer: {actual_output}\n\nTool output: {retrieved_content}'

    questions = {
        'supported': JevRubric(
            question=NoulQuestion(
                instructions='Is every factual claim in the answer supported?'
            ),
            threshold=0.8,
        ),
        'failure': JevRubric(
            question=ChoiceQuestion(
                instructions='Which failure, if any, does the answer show?',
                criteria={
                    'none': 'Every claim traces to the tool output.',
                    'fabricated': 'A claim appears nowhere in the tool output.',
                },
            ),
            label_scores={'none': 1.0, 'fabricated': 0.0},
        ),
    }
```

Run it like any other metric:

```python
from axion.dataset import DatasetItem

item = DatasetItem(
    actual_output='I confirmed this against the filed rate pages.',
    retrieved_content=['No rate page was retrieved.'],
)

result = await Grounding().execute(item)
print(result.pretty())
```

### JevRubric

`JevRubric` is the bridge between an answer and a number.

| Field | Meaning |
|---|---|
| `question` | The `NoulQuestion` / `ChoiceQuestion` / `ScoreQuestion` to ask. |
| `threshold` | The bar this question's sub-metric passes at. `None` inherits the parent metric's. |
| `invert` | `True` when a high answer is the bad outcome — a severity rubric, for instance. |
| `label_scores` | Required for a choice question: what each label is worth. Every label in `criteria` needs one, or construction fails. |

The score each answer type produces:

- **noul** — the probability itself.
- **choice** — `label_scores[choice]`.
- **score** — the level normalized onto `[0, 1]`, so level `2.37` of a four-level rubric scores `0.79`.

`invert=True` returns `1 - score`.

### Rendering the State

`state_template` is formatted with the metric's required and optional fields. When the state needs real assembly — a transcript, tool calls interleaved with their results, truncation to fit — override `build_state` instead:

```python
class Grounding(JevMetric):
    def build_state(self, item: DatasetItem) -> str:
        return render_transcript(item.conversation)
```

!!! warning "Put everything a fair answer depends on in the state"
    A question about whether an answer invented something cannot be answered correctly from the current turn alone. A claim carried over from three turns earlier reads as fabricated unless those turns are in the state to carry it — and the answer will be *stably* wrong, not noisily wrong, so retries will not surface the problem.

Jev accepts 64k tokens per request, of which the state may use 32k.

## One Call, Not N

Every question in the bundle travels in a single request, because Jev bills on input tokens and ingests the state once per call. Splitting four questions across four calls re-sends the state four times — roughly 2.7× the input tokens for the same answers, with no latency saving to show for it.

The bundle explodes into one sub-metric per question, each judged against its own threshold. The questions are not commensurable — a grounding probability and a severity rubric do not average into anything meaningful — so the parent score is the **worst part** rather than the mean. Override `aggregate` to change that:

```python
class Grounding(JevMetric):
    def aggregate(self, scores: List[float]) -> Optional[float]:
        return sum(scores) / len(scores) if scores else None
```

A response missing any question it was asked raises `JevError` rather than scoring the parts that arrived.

## Per-Agent Calibration

The thresholds declared on the class are the metric's *defaults*, not its settings. Any instance can be handed different ones, so a single metric can be held to a different bar for each agent it grades without a subclass per agent:

```python
Grounding(question_thresholds={'supported': 0.85})
Grounding(question_thresholds={'supported': 0.7}, state_template=OTHER_TEMPLATE)
```

`None` is itself a calibration: it drops that question's own threshold, so the sub-metric inherits the parent metric's.

Because these are ordinary constructor arguments, calibration travels as configuration wherever metric instances are built from a config dict:

```python
metric_config = {'question_thresholds': {'supported': 0.85}}
instance = metric_class(**metric_config)
```

A threshold naming a question the metric does not ask raises `MetricValidationError` rather than being ignored — a miscalibrated metric that looks correctly configured is the failure that takes longest to notice.

The calibrated rubrics are copied per instance, so calibrating one never moves another's bar, including instances already built for other agents.

## Reusing a Client

By default each call builds and closes its own client, which is the safe behavior where metric instances are built per request and never disposed of. To share one connection pool across many calls, pass a client in:

```python
from axion._handlers.jev.client import JevClient

async with JevClient() as client:
    metric = Grounding(jev_client=client)
    results = [await metric.execute(item) for item in items]
```

## Errors

| Exception | Raised when |
|---|---|
| `JevAuthError` | 401 or 403. Never retried — a bad key does not get better. |
| `JevRateLimitError` | The call was rate limited. |
| `JevError` | Any other failed call, or a response missing a question it was asked. |

429 and 5xx responses are retried with exponential backoff.

---

<div class="ref-nav" markdown="1">

[Creating Custom Metrics :octicons-arrow-right-24:](creating-metrics.md){ .md-button .md-button--primary }
[Metrics Guide :octicons-arrow-right-24:](../../guides/metrics.md){ .md-button }
[Metrics Reference :octicons-arrow-right-24:](../../reference/metrics.md){ .md-button }

</div>
