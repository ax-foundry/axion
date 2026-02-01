# CaliberHQ

CaliberHQ is a toolkit for calibrating **LLM-as-a-judge** evaluators against a
**human-labeled baseline**. Instead of writing rubrics in the abstract, you work
backward from real outputs so the evaluator learns what actually matters for your
use case.

## Quick start

```python
from axion.caliber import CalibrationSession

session = CalibrationSession()

# 1) Upload your data
session.upload_records(
    [
        {"id": "r1", "query": "…", "actual_output": "…"},
        {"id": "r2", "query": "…", "actual_output": "…"},
    ]
)

# 2) Add human annotations
session.annotate("r1", score=1, notes="Good response")
session.annotate("r2", score=0, notes="Factually incorrect")

# 3) Run LLM evaluation + alignment metrics
result = await session.evaluate(
    criteria="Score 1 if accurate and helpful, 0 otherwise",
    model_name="gpt-4o",
    llm_provider="openai",
)

print(f"Accuracy: {result.metrics.accuracy:.1%}")
print(f"Cohen's Kappa: {result.metrics.cohen_kappa:.3f}")
```

## The 6-step workflow

1. **Upload**: Load your evaluation data
2. **Annotate**: Add human judgments (Accept=1, Reject=0) + optional notes
3. **Evaluate**: Run the LLM judge and compute alignment metrics
4. **Discover patterns**: Cluster annotation notes into themes (LLM / BERTopic / hybrid)
5. **Analyze misalignments**: False positives / false negatives (judge vs human)
6. **Optimize**: Produce improved evaluation criteria

## Two usage patterns

### Pattern 1: Session-based (recommended)

Use `CalibrationSession` for state management and serialization. Good for scripts,
web APIs, and notebooks.

```python
from axion.caliber import CalibrationSession

session = CalibrationSession()
session.upload_csv("data.csv")
session.annotate("r1", score=1, notes="Good")
result = await session.evaluate(criteria="…")
```

### Pattern 2: Direct components (advanced)

Use individual components for fine-grained control.

```python
from axion.caliber import AnnotationManager, EvaluationRunner, UploadHandler

upload = UploadHandler().from_csv("data.csv")
manager = AnnotationManager(upload.records)
manager.annotate("r1", score=1, notes="Good")

# Run evaluation with your own config (see `EvaluationConfig`)
runner = EvaluationRunner()
result = await runner.run(upload.records, manager.get_annotations_dict())
```

## Key components (what to import)

- **Core session**
  - `CalibrationSession`
- **Step components**
  - `UploadHandler`, `AnnotationManager`, `EvaluationRunner`
- **Analysis tools**
  - `PatternDiscovery`, `MisalignmentAnalyzer`, `PromptOptimizer`, `ExampleSelector`
- **Renderers**
  - `ConsoleCaliberRenderer`, `NotebookCaliberRenderer`, `JsonCaliberRenderer`

## Demo

Run the demo script to see the full workflow in action:

```bash
# Basic demo (no API key needed)
python examples/caliber_demo.py

# Full end-to-end with LLM calls
OPENAI_API_KEY=your-key python examples/caliber_demo.py --full
```


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

### Notes on data shapes (for integrations)

Internally, CaliberHQ uses `DatasetItem` (for the underlying data record) plus
`MetricScore`-like fields (for judge outputs):

- `id`, `query`, `actual_output` (DatasetItem-style)
- `judgment` (human label)
- `score`, `explanation`, `signals` (judge output)

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
