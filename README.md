# AXION

<strong>A</strong>gent <strong>X</strong>-Ray <strong>I</strong>nspection & <strong>O</strong>ptimization <strong>N</strong>etwork

![header](resources/toolkit_readme.png)


**A modular white box evaluation framework for AI agents.**


<div style="background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #0b3b8c 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">


Axion provides plug-and-play building blocks for evaluating generative AI agents—going beyond surface-level metrics to focus on what truly matters: continuously improving agents to perform in the real world.</p>
</div>


## Why Axion?

Evaluating AI agents isn't a traditional ML task. It's a context-driven problem requiring more than generic metrics or off-the-shelf tools. Generative agents are unpredictable and domain-specific—success demands tailored evaluation rooted in deep domain expertise, context-aware error analysis, and metrics tied to real business outcomes.

**There's no silver bullet.** One-size-fits-all evaluation leads to mediocrity. The real question is: *Is this agent improving at the job it was designed to do?*

### Core Philosophy

- **Metrics without actionable meaning are worthless** — Dashboards full of numbers that get glanced at once a week don't drive improvement
- **Binary judgments with critiques** — Not vague Likert scales that nobody can agree on
- **Domain-specific evaluation** — A customer support agent and an onboarding agent shouldn't be graded with the same rubric
- **LLM-as-a-Judge done right** — Calibrated against human experts, not used blindly

---

## Key Features

| Feature | Description |
|---------|-------------|
| **98+ Metrics** | Composite, heuristic, and conversational metrics covering RAG, retrieval, response quality, and agent behavior |
| **Metric Registry** | Extensible pattern for registering custom domain-specific metrics |
| **Evaluation Runners** | Parallel batch evaluation with caching and cost estimation |
| **Dataset Management** | `Dataset` and `DatasetItem` classes for single and multi-turn conversations |
| **LLM-as-a-Judge** | Calibrated judge prompts with chain-of-thought reasoning |
| **Synthetic Generation** | Generate evaluation datasets from documents or personas |

---

## Quick Start

```bash
pip install axion
```

```python
from axion import Dataset, DatasetItem, metric_registry
from axion.metrics import Faithfulness, AnswerRelevancy
from axion.runners import evaluation_runner

# Load your evaluation dataset
dataset = Dataset.from_json("eval_data.json")

# Select metrics for your use case
metrics = [Faithfulness(), AnswerRelevancy()]

# Run evaluation
results = await evaluation_runner(dataset, metrics)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ax-foundry/axion.git
cd axion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file:

```bash
OPENAI_API_KEY=<your-key>
LOG_LEVEL="INFO"
```

---

## Documentation

**[Full Documentation](https://ax-foundry.github.io/axion/)**

- [Agent Evaluation Playbook](https://ax-foundry.github.io/axion/agent_playbook/) — The Analyze-Measure-Improve methodology
- [Metrics Reference](https://ax-foundry.github.io/axion/metrics/) — Complete metric catalog
- [Dataset Guide](https://ax-foundry.github.io/axion/datasets/) — Building effective evaluation datasets

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

## License

MIT License — see [LICENSE](LICENSE) for details.
