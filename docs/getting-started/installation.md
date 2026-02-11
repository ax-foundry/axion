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

Axion uses optional dependencies to keep the core installation lightweight. Install extras based on what you need:

=== ":material-radar: Tracing Providers"

    ```bash
    # Logfire (OpenTelemetry-based)
    pip install -e ".[logfire]"

    # Langfuse (LLM-specific observability)
    pip install -e ".[langfuse]"

    # Opik
    pip install -e ".[opik]"

    # All tracing providers
    pip install -e ".[tracing]"
    ```

=== ":material-magnify: Search Integrations"

    ```bash
    # Google Search via SerpAPI
    pip install -e ".[search]"
    ```

    Requires `SERPAPI_KEY` environment variable.

=== ":material-puzzle: LlamaIndex Extensions"

    ```bash
    # HuggingFace embeddings and LLMs
    pip install -e ".[huggingface]"

    # Docling document reader (PDF, DOCX, HTML, images)
    pip install -e ".[docling]"
    ```

=== ":material-chart-bar: Visualization"

    ```bash
    # Matplotlib and Seaborn for plotting
    pip install -e ".[plotting]"
    ```

### Combining Extras

Install multiple extras at once:

```bash
# Example: search + tracing + plotting
pip install -e ".[search,tracing,plotting]"
```

## Configuration

=== ":material-file-cog: .env File"

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

=== ":material-language-python: Programmatic"

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

---

<div class="ref-nav" markdown="1">

[Working with Datasets :octicons-arrow-right-24:](../guides/datasets.md){ .md-button .md-button--primary }
[Metrics & Evaluation :octicons-arrow-right-24:](../guides/metrics.md){ .md-button }
[Agent Evaluation Playbook :octicons-arrow-right-24:](../agent_playbook.md){ .md-button }

</div>
