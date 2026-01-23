# AlignEval

AlignEval is a tool for calibrating LLM-as-a-judge evaluators against a
human-labeled baseline. Instead of writing criteria in the abstract, you
work backward from real outputs so the evaluator learns what actually
matters for your use case.

## Core Philosophy: Calibrate, Don’t Just Create

Aligning AI to human preferences is only half the battle. The other half is
calibrating your criteria to the model’s output distribution. Teams often
write elaborate rubrics without looking at the data first, which produces
criteria that are irrelevant or unrealistic. AlignEval is built to keep the
data in the loop.

Inspiration for AlignEval comes from eugeneyan’s aligneval.

## Workflow

1. **Prepare Data**: Start with a dataset of inputs and generated outputs.
2. **Annotate**: Add a human judgment (pass/fail) to each item.
3. **Configure & Execute**: Define the LLM-as-a-judge instructions and run.
4. **Analyze**: Compare LLM scores to the human baseline to find gaps.

## Using AlignEval in Python

AlignEval is designed for programmatic workflows and Jupyter notebooks. It
includes an interactive annotation flow and rich analysis outputs.

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

### Step 3: Run AlignEval

```python
from axion.align import AlignEval

evaluator = AlignEval(dataset, PassFailMetric())
evaluator.annotate()  # optional if judgments are already present
results_df = evaluator.execute()
```

## UI-Friendly Outputs

For web or UI pipelines, use `WebAlignEval` to return JSON-serializable
results and attach optional progress callbacks. Both `AlignEval` and
`WebAlignEval` support `execute(as_dict=...)` for a shared interface.

```python
from axion.align import WebAlignEval

web_eval = WebAlignEval(dataset, PassFailMetric())
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
web_eval = WebAlignEval.from_records(records, PassFailMetric())
payload = web_eval.execute(as_dict=True)

## Renderers

AlignEval uses a renderer interface so UIs can plug in without changing
core logic. Out of the box you get:

- `NotebookAlignEvalRenderer` for notebooks (Jupyter)
- `ConsoleAlignEvalRenderer` for terminal usage
- `JsonAlignEvalRenderer` for JSON-first workflows

```python
from axion.align import AlignEval, ConsoleAlignEvalRenderer

evaluator = AlignEval(dataset, PassFailMetric(), renderer=ConsoleAlignEvalRenderer())
evaluator.execute()
```
```
