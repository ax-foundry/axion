# CaliberHQ

CaliberHQ is a tool for calibrating LLM-as-a-judge evaluators against a
human-labeled baseline. Instead of writing criteria in the abstract, you
work backward from real outputs so the evaluator learns what actually
matters for your use case.

## Core Philosophy: Calibrate, Don't Just Create

Aligning AI to human preferences is only half the battle. The other half is
calibrating your criteria to the model's output distribution. Teams often
write elaborate rubrics without looking at the data first, which produces
criteria that are irrelevant or unrealistic. CaliberHQ is built to keep the
data in the loop.

## Workflow

1. **Prepare Data**: Start with a dataset of inputs and generated outputs.
2. **Annotate**: Add a human judgment (pass/fail) to each item.
3. **Configure & Execute**: Define the LLM-as-a-judge instructions and run.
4. **Analyze**: Compare LLM scores to the human baseline to find gaps.

## Using CaliberHQ in Python

CaliberHQ is designed for programmatic workflows and Jupyter notebooks. It
includes an interactive annotation flow and rich analysis outputs.


```
┌─────────────────────────────────────────────────────────────────────────────────────┐
  │                              CaliberHQ Workflow                                      │
  │                   Align LLM Judges with Human Judgment                              │
  └─────────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  STEP 1: UPLOAD                                                                      │
  ├─────────────────────────────────────────────────────────────────────────────────────┤
  │                                                                                      │
  │   ┌──────────────┐         ┌─────────────────────────────────────────────────────┐  │
  │   │   CSV File   │────────▶│  Expected Columns:                                  │  │
  │   │  (drag/drop) │         │  • id, query, actual_output                         │  │
  │   └──────────────┘         │  • llm_score (0/1), reasoning (optional)            │  │
  │                            └─────────────────────────────────────────────────────┘  │
  │                                              │                                       │
  │                                              ▼                                       │
  │                            ┌─────────────────────────────────────────────────────┐  │
  │                            │  Data loaded into calibration store                 │  │
  │                            │  Records ready for human review                     │  │
  │                            └─────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  STEP 2: REVIEW & LABEL                                                              │
  ├─────────────────────────────────────────────────────────────────────────────────────┤
  │                                                                                      │
  │   ┌─────────────────────────────────────────┐    ┌────────────────────────────────┐ │
  │   │         ANNOTATION CARD                 │    │      PROGRESS SIDEBAR          │ │
  │   │  ┌───────────────────────────────────┐  │    │  ┌──────────────────────────┐  │ │
  │   │  │  Query: "User question..."        │  │    │  │  ○ ○ ● ○ ○ ○ ○ ○ ○ ○    │  │ │
  │   │  │  Output: "LLM response..."        │  │    │  │  3 / 50 annotated        │  │ │
  │   │  │  LLM Score: 1 (Accept)            │  │    │  │                          │  │ │
  │   │  │  Reasoning: "Good because..."     │  │    │  │  Legend:                 │  │ │
  │   │  └───────────────────────────────────┘  │    │  │  ● Current   ✓ Done      │  │ │
  │   │                                         │    │  │  ○ Pending   ✗ Rejected  │  │ │
  │   │  ┌─────────────────────────────────┐    │    │  └──────────────────────────┘  │ │
  │   │  │  YOUR JUDGMENT                  │    │    └────────────────────────────────┘ │
  │   │  │                                 │    │                                       │
  │   │  │  ┌─────────┐    ┌─────────┐     │    │    ┌────────────────────────────────┐ │
  │   │  │  │ ACCEPT  │    │ REJECT  │     │    │    │      KEYBOARD SHORTCUTS        │ │
  │   │  │  │  [ A ]  │    │  [ R ]  │     │    │    │  ┌──────────────────────────┐  │ │
  │   │  │  └─────────┘    └─────────┘     │    │    │  │  A = Accept              │  │ │
  │   │  │                                 │    │    │  │  R = Reject              │  │ │
  │   │  │  Notes: ___________________     │    │    │  │  ← → = Navigate          │  │ │
  │   │  │         (optional patterns)     │    │    │  └──────────────────────────┘  │ │
  │   │  └─────────────────────────────────┘    │    └────────────────────────────────┘ │
  │   │                                         │                                       │
  │   │       [ ← Previous ]  [ Next → ]        │                                       │
  │   └─────────────────────────────────────────┘                                       │
  │                                                                                      │
  │   Output: humanAnnotations = { "id1": {score: 1, notes: "..."}, ... }               │
  └─────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  STEP 3: BUILD EVAL                                                                  │
  ├─────────────────────────────────────────────────────────────────────────────────────┤
  │                                                                                      │
  │   ┌─────────────────────────────────────────────────────────────────────────────┐   │
  │   │  1. RUN EVALUATION                                                          │   │
  │   │  ┌───────────────────────────────────────────────────────────────────────┐  │   │
  │   │  │  Configure:  Model [gpt-4o ▼]  System Prompt [...]  Criteria [...]    │  │   │
  │   │  │                                                                       │  │   │
  │   │  │                    [ ▶ Run LLM Evaluation ]                           │  │   │
  │   │  └───────────────────────────────────────────────────────────────────────┘  │   │
  │   └─────────────────────────────────────────────────────────────────────────────┘   │
  │                                               │                                      │
  │                                               ▼                                      │
  │   ┌─────────────────────────────────────────────────────────────────────────────┐   │
  │   │  2. ALIGNMENT METRICS                                                       │   │
  │   │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │   │
  │   │  │  Accuracy  │ │ Precision  │ │   Recall   │ │  F1 Score  │               │   │
  │   │  │   0.82     │ │   0.85     │ │   0.79     │ │   0.82     │               │   │
  │   │  └────────────┘ └────────────┘ └────────────┘ └────────────┘               │   │
  │   │                                                                             │   │
  │   │  ┌─────────────────────────────────┐  Cohen's Kappa: 0.64 (Substantial)    │   │
  │   │  │      CONFUSION MATRIX          │                                        │   │
  │   │  │  ┌─────────────────────────┐   │  Interpretation:                       │   │
  │   │  │  │         │ Human │ Human │   │  • LLM agrees with humans 82% of time  │   │
  │   │  │  │         │ Accept│Reject │   │  • Kappa > 0.6 = good agreement        │   │
  │   │  │  │─────────┼───────┼───────│   │                                        │   │
  │   │  │  │LLM Acc. │  TP   │  FP   │   │                                        │   │
  │   │  │  │LLM Rej. │  FN   │  TN   │   │                                        │   │
  │   │  │  └─────────────────────────┘   │                                        │   │
  │   │  └─────────────────────────────────┘                                        │   │
  │   └─────────────────────────────────────────────────────────────────────────────┘   │
  │                                               │                                      │
  │                                               ▼                                      │
  │   ┌─────────────────────────────────────────────────────────────────────────────┐   │
  │   │  3. MISALIGNMENT ANALYSIS                                                   │   │
  │   │  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │   │
  │   │  │  FALSE POSITIVES (FP)       │  │  FALSE NEGATIVES (FN)               │   │   │
  │   │  │  LLM: Accept, Human: Reject │  │  LLM: Reject, Human: Accept         │   │   │
  │   │  │  → LLM is too lenient       │  │  → LLM is too strict                │   │   │
  │   │  │                             │  │                                     │   │   │
  │   │  │  Examples:                  │  │  Examples:                          │   │   │
  │   │  │  • "Hallucinated facts..."  │  │  • "Minor formatting but correct.." │   │   │
  │   │  │  • "Missed key context..."  │  │  • "Edge case handled well..."      │   │   │
  │   │  └─────────────────────────────┘  └─────────────────────────────────────┘   │   │
  │   └─────────────────────────────────────────────────────────────────────────────┘   │
  │                                               │                                      │
  │                                               ▼                                      │
  │   ┌─────────────────────────────────────────────────────────────────────────────┐   │
  │   │  4. PROMPT OPTIMIZATION                                                     │   │
  │   │  ┌───────────────────────────────────────────────────────────────────────┐  │   │
  │   │  │  Based on misalignment patterns, suggested criteria improvements:     │  │   │
  │   │  │                                                                       │  │   │
  │   │  │  • Add explicit check for factual accuracy                           │  │   │
  │   │  │  • Clarify handling of edge cases                                    │  │   │
  │   │  │  • Reduce strictness on formatting requirements                      │  │   │
  │   │  │                                                                       │  │   │
  │   │  │  [ Copy Optimized Prompt ]  [ Apply & Re-run ]                       │  │   │
  │   │  └───────────────────────────────────────────────────────────────────────┘  │   │
  │   └─────────────────────────────────────────────────────────────────────────────┘   │
  │                                               │                                      │
  │                                               ▼                                      │
  │   ┌─────────────────────────────────────────────────────────────────────────────┐   │
  │   │  5. PATTERN INSIGHTS (from annotation notes)                                │   │
  │   │  ┌───────────────────────────────────────────────────────────────────────┐  │   │
  │   │  │  Discovered patterns from human annotator notes:                      │  │   │
  │   │  │                                                                       │  │   │
  │   │  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐         │  │   │
  │   │  │  │ Hallucinations  │ │ Tone Issues     │ │ Missing Context │         │  │   │
  │   │  │  │ 12 cases        │ │ 8 cases         │ │ 5 cases         │         │  │   │
  │   │  │  │ [expandable]    │ │ [expandable]    │ │ [expandable]    │         │  │   │
  │   │  │  └─────────────────┘ └─────────────────┘ └─────────────────┘         │  │   │
  │   │  └───────────────────────────────────────────────────────────────────────┘  │   │
  │   └─────────────────────────────────────────────────────────────────────────────┘   │
  │                                                                                      │
  │   ┌─────────────────────────────────────────────────────────────────────────────┐   │
  │   │                          [ Export Results ]                                 │   │
  │   └─────────────────────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────────────────┘


                                ┌───────────────────────┐
                                │   FEEDBACK LOOP       │
                                │                       │
                                │  Optimized prompt     │
                                │         │             │
                                │         ▼             │
                                │  Re-run evaluation    │
                                │         │             │
                                │         ▼             │
                                │  Compare metrics      │
                                │         │             │
                                │         ▼             │
                                │  Iterate until        │
                                │  aligned              │
                                └───────────────────────┘

  Summary Flow:

      ┌────────┐      ┌──────────────┐      ┌────────────┐
      │ UPLOAD │ ───▶ │ REVIEW/LABEL │ ───▶ │ BUILD EVAL │
      └────────┘      └──────────────┘      └────────────┘
           │                  │                    │
           ▼                  ▼                    ▼
      CSV with LLM       Human scores         Metrics + Analysis
      judge outputs      (Accept/Reject)      + Optimized Prompts
                         + Notes                    │
                                                    │
                                ┌───────────────────┘
                                ▼
                          Iterate until
                          LLM ≈ Human
```

### Step 1: Create a Dataset

```python
from axion.dataset import Dataset, DatasetItem

items = [
    DatasetItem(
        id="item-1",
        query="What is the capital of France?",
        expected_output="Paris",
        actual_output="Paris.",
    ),
    DatasetItem(
        id="item-2",
        query="What is 2+2?",
        expected_output="4",
        actual_output="5",
    ),
]

dataset = Dataset(items=items)
```

### Step 2: Define a Metric

Define an LLM-as-a-judge metric with a clear instruction. The metric can be
as simple as a binary pass/fail rubric.

```python
from axion.metrics.base import BaseMetric

class PassFailMetric(BaseMetric):
    instruction = (
        "Score 1 if the answer is correct and complete. "
        "Otherwise score 0 and explain why."
    )
```

### Step 3: Run CaliberHQ

```python
from axion.align import CaliberHQ

evaluator = CaliberHQ(dataset, PassFailMetric())
evaluator.annotate()  # optional if judgments are already present
results_df = evaluator.execute()
```

## UI-Friendly Outputs

For web or UI pipelines, use `WebCaliberHQ` to return JSON-serializable
results and attach optional progress callbacks. Both `CaliberHQ` and
`WebCaliberHQ` support `execute(as_dict=...)` for a shared interface.

```python
from axion.align import WebCaliberHQ

web_eval = WebCaliberHQ(dataset, PassFailMetric())
payload = web_eval.execute(
    as_dict=True,
    on_progress=lambda current, total: print(current, total),
)

# payload["results"] -> list of row dicts
# payload["metrics"] -> summary metrics
# payload["confusion_matrix"] -> confusion matrix dict
```

You can also construct a dataset from uploaded records:

```python
web_eval = WebCaliberHQ.from_records(records, PassFailMetric())
payload = web_eval.execute(as_dict=True)
```

## Intelligent Example Selection

When providing few-shot examples to calibrate your LLM judge, use
`ExampleSelector` for smarter selection instead of naive slicing.

```python
from axion.align import ExampleSelector, SelectionStrategy

selector = ExampleSelector(seed=42)

# Balanced selection (50/50 accept/reject)
result = selector.select(records, annotations, count=6)

# Prioritize misaligned cases (requires prior eval results)
result = selector.select(
    records, annotations, count=6,
    strategy=SelectionStrategy.MISALIGNMENT_GUIDED,
    eval_results=prior_results
)

# Cover discovered patterns (requires Pattern Discovery)
result = selector.select(
    records, annotations, count=6,
    strategy=SelectionStrategy.PATTERN_AWARE,
    patterns=discovered_patterns
)
```

Three strategies are available:

| Strategy | Use Case |
|----------|----------|
| `BALANCED` | Default - 50/50 accept/reject sampling |
| `MISALIGNMENT_GUIDED` | Prioritize FP/FN cases from prior eval |
| `PATTERN_AWARE` | Cover discovered failure patterns |

See [Example Selector Deep Dive](../deep-dives/align/example-selector.md) for
detailed usage.

## Renderers

CaliberHQ uses a renderer interface so UIs can plug in without changing
core logic. Out of the box you get:

- `NotebookCaliberHQRenderer` for notebooks (Jupyter)
- `ConsoleCaliberHQRenderer` for terminal usage
- `JsonCaliberHQRenderer` for JSON-first workflows

```python
from axion.align import CaliberHQ, ConsoleCaliberHQRenderer

evaluator = CaliberHQ(dataset, PassFailMetric(), renderer=ConsoleCaliberHQRenderer())
evaluator.execute()
```
