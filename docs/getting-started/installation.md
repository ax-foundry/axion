# Installation

Get started with Axion in minutes.

## Install from Source

```bash
# Clone the repository
git clone https://github.com/ax-foundry/axion.git
cd axion

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install the package
pip install -e .
```

## Development Installation

For contributing or running tests:

```bash
# Install with dev dependencies
pip install -e ".[dev]"
```

## Optional Dependencies

Install with specific extras for additional functionality:

```bash
# Logfire tracing support
pip install -e ".[logfire]"

# Langfuse tracing support
pip install -e ".[langfuse]"

# All tracing providers
pip install -e ".[tracing]"
```

## Configuration

Create a `.env` file in your project root:

```bash
# Required for LLM-based metrics
OPENAI_API_KEY=<your-key>

# Optional: Logging settings
LOG_LEVEL="INFO"
LOG_RICH="true"

# Optional: Tracing (auto-detects if credentials present)
TRACING_MODE="langfuse"  # or: noop, logfire, otel, opik
LANGFUSE_SECRET_KEY=<your-key>
LANGFUSE_PUBLIC_KEY=<your-key>
```

### Programmatic Configuration

Use `axion.init()` to configure both logging and tracing at once:

```python
import axion

# Initialize with custom settings
axion.init(
    tracing='langfuse',  # or: noop, logfire, otel, opik
    log_level='DEBUG',
    log_rich=True,
)

# Or just let it auto-configure from environment variables
# (no init() call needed - works automatically)
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
