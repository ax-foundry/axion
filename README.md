<p align="center">
  <img src="resources/axion_main_transparent.png" width="500" alt="Axion">
</p>



<p align="center">
<strong>A</strong>gent <strong>X</strong>-Ray <strong>I</strong>nspection & <strong>O</strong>ptimization <strong>N</strong>etwork – A modular white box evaluation framework for AI agents.
</p>

<p align="center">
<a href="https://ax-foundry.github.io/axion/"><strong>Documentation</a> |
<a href="https://ax-foundry.github.io/axion/getting-started/installation/">Quick Start</a> |
<a href="https://ax-foundry.github.io/axion/metric-registry/composite/">Metrics</a> |
<a href="https://ax-foundry.github.io/axion/agent_playbook/">Agent Playbook</a>
</strong></p>

<p align="center">
<a href="https://github.com/ax-foundry/axion/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
</p>

<div style="background: linear-gradient(135deg, #8B9F4F 0%, #6B7A3A 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

Moving from passive, black-box observation to active, white-box evaluation—Axion empowers builders with **actionable signals**, **automated pipelines**, and **fully transparent metrics**. See exactly why your agent succeeds or fails.
</div>


## Why Axion?

Evaluating AI agents isn't a traditional ML task. It's a context-driven problem requiring more than generic metrics or off-the-shelf tools. Generative agents are unpredictable and domain-specific—success demands tailored evaluation rooted in deep domain expertise, context-aware error analysis, and metrics tied to real business outcomes.

**There's no silver bullet.** One-size-fits-all evaluation leads to mediocrity. The real question is: *Is this agent improving at the job it was designed to do?*

### Core Philosophy

- **White-box transparency** — Black-box metrics give you numbers; white-box metrics give you understanding
- **Metrics without actionable meaning are worthless** — Dashboards full of numbers that get glanced at once a week don't drive improvement
- **Binary judgments with critiques** — Not vague Likert scales that nobody can agree on
- **Domain-specific evaluation** — A customer support agent and an onboarding agent shouldn't be graded with the same rubric
- **LLM-as-a-Judge done right** — Calibrated against human experts, not used blindly

---

## Key Features

| Feature | Description |
|---------|-------------|
| **30+ Metrics** | Composite, heuristic, and conversational metrics covering RAG, retrieval, response quality, and agent behavior |
| **Metric Registry** | Extensible pattern for registering custom domain-specific metrics |
| **Evaluation Runners** | Parallel batch evaluation with caching and cost estimation |
| **Dataset Management** | `Dataset` and `DatasetItem` classes for single and multi-turn conversations |
| **LLM-as-a-Judge** | Calibrated judge prompts with chain-of-thought reasoning |
| **Synthetic Generation** | Generate evaluation datasets from documents or personas |
| **Hierarchical Scoring** | Multi-level scoring with weighted dimensions and diagnostic drill-down |

---

## Hierarchical Scoring

**What sets Axion apart:** Our scoring framework is hierarchical by design—moving from a single, intuitive overall score down into layered sub-scores. This structure delivers more than just measurement; it provides a diagnostic map of quality.

```
                    ┌─────────────────┐
                    │  Overall Score  │
                    │      0.82       │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  Relevance  │    │  Accuracy   │    │    Tone     │
   │    0.91     │    │    0.78     │    │    0.85     │
   └─────────────┘    └─────────────┘    └─────────────┘
```

**Why this matters:**

- **Instant Root Cause Diagnosis** — Drill down to pinpoint whether issues stem from relevance, accuracy, tone, or other dimensions
- **Strategic Prioritization** — Forces clarity on what really matters for your business
- **Actionable Feedback Loop** — Each layer translates directly into actions (retraining, prompt adjustments, alignment tuning)
- **Customizable to Business Goals** — Weight and expand dimensions to match your unique KPIs

```python
from axion.runners import evaluation_runner
from axion.metrics import AnswerRelevancy
from axion.dataset import DatasetItem

# Define hierarchical scoring configuration
config = {
    'metric': {
        'Relevance': AnswerRelevancy(metric_name='Relevancy'),
    },
    'model': {
        'ANSWER_QUALITY': {'Relevance': 1.0},
    },
    'weights': {
        'ANSWER_QUALITY': 1.0,
    }
}

data_item = DatasetItem(
    query="How do I reset my password?",
    actual_output="To reset your password, click 'Forgot Password' on the login page and follow the email instructions.",
)

results = evaluation_runner(
    evaluation_inputs=[data_item],
    scoring_config=config,  # Or pass path to config.yaml
    evaluation_name="Hierarchical Evaluation"
)

# Generate scorecard with hierarchical breakdown
results.to_scorecard()
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/ax-foundry/axion.git
cd axion
pip install -e .
```

```python
from axion import Dataset, DatasetItem, metric_registry
from axion.metrics import Faithfulness, AnswerRelevancy
from axion.runners import evaluation_runner

# Load your evaluation dataset
dataset = Dataset.read_csv("eval_data.csv")

# Select metrics for your use case
metrics = [Faithfulness(), AnswerRelevancy()]

# Run evaluation
results = evaluation_runner(dataset, evaluation_name="my_eval", scoring_metrics=metrics)
```

### Configuration

Create a `.env` file:

```bash
OPENAI_API_KEY=<your-key>
LOG_LEVEL="INFO"
```

### Development Installation

```bash
# With dev dependencies (for contributing/testing)
pip install -e ".[dev]"

# With tracing support
pip install -e ".[tracing]"
```

---

## Documentation

**[Full Documentation](https://ax-foundry.github.io/axion/)**

- [Agent Evaluation Playbook](https://ax-foundry.github.io/axion/agent_playbook/) — The Analyze-Measure-Improve methodology
- [Metrics Reference](https://ax-foundry.github.io/axion/guides/metrics/) — Complete metric catalog
- [Dataset Guide](https://ax-foundry.github.io/axion/guides/datasets/) — Building effective evaluation datasets

---

## The Analyze-Measure-Improve Cycle

Axion is built around the **AMI methodology**—a disciplined, repeatable process that replaces one-off troubleshooting with a data-driven continuous improvement loop.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    ANALYZE ──────► MEASURE ──────► IMPROVE ──────┐          │
│        ▲                                          │         │
│        └──────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| Phase | Outcome |
|-------|---------|
| **Analyze** | Clear map of what's broken and why |
| **Measure** | Solid baselines and priorities grounded in data |
| **Improve** | Fixes that address root causes—not symptoms |

See the [Agent Evaluation Playbook](https://ax-foundry.github.io/axion/agent_playbook/) for the complete methodology.

---

## Contributing

Axion is the foundation of a growing ecosystem. Your contributions accelerate the entire community.

| Built something useful? | Found a gap? | Want to collaborate? |
|------------------------|--------------|---------------------|
| Submit a PR | Raise an issue | Open a discussion |

```bash
# Run linting
pre-commit run --all-files

# Run tests
pytest
```

---

## Why "Axion"?

**A**gent **X**-Ray **I**nspection & **O**ptimization **N**etwork

The name draws inspiration from the [axion](https://en.wikipedia.org/wiki/Axion)—a hypothetical particle in physics proposed to solve the "strong CP problem" in quantum chromodynamics. Physicists Frank Wilczek and Steven Weinberg named it after a laundry detergent, hoping it would "clean up" their theoretical mess.

Like its namesake, this toolkit provides lightweight, modular components that work together to solve complex problems. Axions in physics are characterized by being incredibly small yet potentially accounting for much of the universe's dark matter through sheer numbers. Similarly, Axion the toolkit offers small, focused tools that combine to tackle the substantial challenge of AI agent evaluation.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
