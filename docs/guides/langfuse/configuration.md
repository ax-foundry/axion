---
icon: custom/settings
---
# Langfuse Configuration

This guide covers all setup and configuration options for integrating Axion with Langfuse.

## Prerequisites

- A [Langfuse](https://langfuse.com) account (cloud or self-hosted)
- Langfuse API keys (public and secret)
- Python 3.12+
- Axion installed with Langfuse support: `pip install axion[langfuse]`

## Environment Variables

Configure Langfuse credentials via environment variables:

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `LANGFUSE_PUBLIC_KEY` | Yes | Your Langfuse public key (`pk-lf-...`) | - |
| `LANGFUSE_SECRET_KEY` | Yes | Your Langfuse secret key (`sk-lf-...`) | - |
| `LANGFUSE_BASE_URL` | No | Langfuse host URL | `https://us.cloud.langfuse.com` |
| `LANGFUSE_TAGS` | No | Comma-separated default tags for traces | - |
| `LANGFUSE_ENVIRONMENT` | No | Environment name (e.g., `production`) | - |
| `LANGFUSE_TRACING_ENVIRONMENT` | No | Environment name (Langfuse SDK standard) | - |
| `LANGFUSE_DEFAULT_TAGS` | No | Comma-separated default tags for scores | - |
| `TRACING_MODE` | No | Set to `langfuse` to force Langfuse provider | Auto-detected |

### Example `.env` File

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
LANGFUSE_ENVIRONMENT=production
LANGFUSE_TAGS=prod,v2.0
LANGFUSE_DEFAULT_TAGS=evaluation,automated
```

## Programmatic Configuration

=== "Auto-Detection"

    ```python
    from axion._core.tracing import Tracer

    # Tracer auto-detects Langfuse from LANGFUSE_SECRET_KEY
    tracer = Tracer('llm')
    ```

=== "Explicit Configuration"

    ```python
    from axion._core.tracing import configure_tracing, Tracer

    # Explicitly configure Langfuse
    configure_tracing(provider='langfuse')
    tracer = Tracer('llm')
    ```

=== "Loader with Custom Credentials"

    ```python
    from axion.tracing import LangfuseTraceLoader

    # Override credentials for loader
    loader = LangfuseTraceLoader(
        public_key='pk-lf-...',
        secret_key='sk-lf-...',
        host='https://cloud.langfuse.com',
        default_tags=['evaluation', 'automated']
    )
    ```

=== "Direct LangfuseTracer"

    ```python
    from axion._core.tracing.langfuse.tracer import LangfuseTracer

    # Direct initialization with all options
    tracer = LangfuseTracer(
        tags=['prod', 'v1.0'],
        environment='production'
    )
    ```

## Auto-Detection Behavior

When you create a `Tracer()` instance, Axion automatically detects the appropriate backend:

1. **Check `TRACING_MODE`**: If set to `langfuse`, use Langfuse
2. **Check credentials**: If `LANGFUSE_SECRET_KEY` is set, use Langfuse
3. **Check Logfire**: If Logfire is configured, use OpenTelemetry
4. **Default**: Use NOOP tracer (no overhead)

```python
import os
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-...'
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-...'

from axion._core.tracing import Tracer
tracer = Tracer('llm')  # Automatically uses Langfuse
```

## LangfuseTraceLoader Initialization

The `LangfuseTraceLoader` class accepts these initialization options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `public_key` | `str` | `None` | Override `LANGFUSE_PUBLIC_KEY` |
| `secret_key` | `str` | `None` | Override `LANGFUSE_SECRET_KEY` |
| `host` | `str` | `None` | Override `LANGFUSE_BASE_URL` |
| `default_tags` | `list[str]` | `None` | Tags applied to all scores |
| `request_pacing` | `float` | `0.05` | Delay between API requests |
| `max_retries` | `int` | `3` | Max retry attempts |
| `base_delay` | `float` | `0.5` | Initial retry delay |

```python
from axion.tracing import LangfuseTraceLoader

loader = LangfuseTraceLoader(
    public_key='pk-lf-...',
    secret_key='sk-lf-...',
    host='https://us.cloud.langfuse.com',
    default_tags=['evaluation'],
    request_pacing=0.1,  # Slower for rate limit prone accounts
    max_retries=5,
)
```

## Region Configuration

Langfuse offers different regional endpoints:

```python
import os

# US region (default)
os.environ['LANGFUSE_BASE_URL'] = 'https://us.cloud.langfuse.com'

# EU region
os.environ['LANGFUSE_BASE_URL'] = 'https://cloud.langfuse.com'

# Self-hosted
os.environ['LANGFUSE_BASE_URL'] = 'https://your-langfuse-instance.com'
```

## Tags and Environment

### Tags

Tags help filter and organize traces. They can be set at multiple levels:

```python
import os

# Default tags for all traces (comma-separated)
os.environ['LANGFUSE_TAGS'] = 'prod,v1.0'

# Default tags for scores
os.environ['LANGFUSE_DEFAULT_TAGS'] = 'evaluation,automated'
```

**Tag precedence:**

1. Per-call tags (passed to method)
2. Loader default tags (set at initialization)
3. Environment variable fallback (`LANGFUSE_TAGS`)

### Environment

Environment identifies which deployment created a trace:

```python
import os
os.environ['LANGFUSE_ENVIRONMENT'] = 'production'
```

!!! warning "Environment Limitation"
    Environment **cannot** be set when pushing scores to existing traces. It must be configured at tracer initialization when creating the original traces.

## Troubleshooting

### Connection Issues

If the Langfuse client fails to initialize:

1. **Verify credentials:**
   ```python
   import os
   print(f"Public key: {os.environ.get('LANGFUSE_PUBLIC_KEY', 'NOT SET')}")
   print(f"Secret key set: {bool(os.environ.get('LANGFUSE_SECRET_KEY'))}")
   ```

2. **Check the base URL:**
   ```python
   print(f"Base URL: {os.environ.get('LANGFUSE_BASE_URL', 'default')}")
   ```

3. **Test connection:**
   ```python
   from axion.tracing import LangfuseTraceLoader

   loader = LangfuseTraceLoader()
   traces = loader.fetch_traces(limit=1)
   print(f"Connection successful: {len(traces)} traces")
   ```

### Rate Limiting (429 Errors)

If you encounter rate limiting when fetching many traces:

```python
# Increase delay between requests
loader = LangfuseTraceLoader(
    request_pacing=0.1,  # Increase from default 0.05
    max_retries=5,
    base_delay=1.0,
)

# Or fetch summaries only (fewer API calls)
traces = loader.fetch_traces(
    limit=1000,
    fetch_full_traces=False
)
```

### Credentials Not Found

Ensure environment variables are set **before** importing Axion modules:

```python
import os
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-...'
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-...'

# Now import
from axion._core.tracing import Tracer
```

Or use a `.env` file with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()  # Load before imports

from axion._core.tracing import Tracer
```

## Next Steps

- **[Tracing](tracing.md)**: Create and manage traces
- **[Publishing](publishing.md)**: Publish evaluation results
- **[Overview](overview.md)**: Complete workflow example
