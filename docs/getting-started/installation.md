# Installation

Get started with Axion in minutes.

## Quick Install

```bash
pip install axion
```

## Development Installation

For contributing or running from source:

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

## Configuration

Create a `.env` file in your project root:

```bash
# Required for LLM-based metrics
OPENAI_API_KEY=<your-key>

# Optional settings
LOG_LEVEL="INFO"
```

## Verify Installation

```python
from axion import Dataset, metric_registry

# Check available metrics
print(metric_registry.list_metrics())
```

## Next Steps

- [Working with Datasets](../guides/datasets.md) - Learn how to create evaluation datasets
- [Metrics & Evaluation](../guides/metrics.md) - Understand the metrics system
- [Agent Evaluation Playbook](../agent_playbook.md) - Best practices for agent evaluation
