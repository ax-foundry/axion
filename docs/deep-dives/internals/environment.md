# Axion Environment & Settings

This guide details the configuration management system for Axion, powered by pydantic-settings.

## Core Concepts

The configuration system is built on several key principles:

**Schema-First Design**: The `AxionConfig` class serves as the central schema. It's a Pydantic `BaseModel` that defines the shape of the configuration—all available fields, their types, and default values.

**Environment Variable Loading**: Settings are automatically loaded from environment variables and `.env` files using pydantic-settings.

**Centralized Access**: A single, global `settings` object is created at startup. This object is the source of truth for all configuration values throughout the application.

## Configuration Loading

The system loads settings with a clear order of precedence:

1. **Environment Variables**: System environment variables (highest priority)
2. **.env File**: Values loaded from the `.env` file
3. **Default Values**: Default values defined in the `AxionConfig` schema (lowest priority)

### .env File Discovery

The system automatically discovers your `.env` file:

1. First checks for an explicit `ENV_PATH` environment variable
2. Falls back to `find_dotenv()`, which searches the current and parent directories

## Configuration Schema

The following settings are available. Environment variables are case-insensitive.

### General Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `debug` | `DEBUG` | `False` | Enable debug mode for verbose logging and diagnostics. |
| `port` | `PORT` | `8000` | The port the application will run on. |
| `hosts` | `HOSTS` | `['localhost']` | A list of allowed hostnames. |

### LLM Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `llm_provider` | `LLM_PROVIDER` | `'openai'` | Default provider for language models. |
| `embedding_provider` | `EMBEDDING_PROVIDER` | `'openai'` | Default provider for embedding models. |
| `llm_model_name` | `LLM_MODEL_NAME` | `'gpt-4o'` | Default language model name. |
| `embedding_model_name` | `EMBEDDING_MODEL_NAME` | `'text-embedding-ada-002'` | Default embedding model name. |
| `api_base_url` | `API_BASE_URL` | `None` | Optional base URL for OpenAI-compatible APIs. |
| `litellm_verbose` | `LITELLM_VERBOSE` | `False` | Enable verbose logging for LiteLLM debugging. |

### API Key Settings

| Setting | Environment Variable | Description |
|---------|---------------------|-------------|
| `openai_api_key` | `OPENAI_API_KEY` | API key for OpenAI models. |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | API key for Anthropic Claude models. |
| `google_api_key` | `GOOGLE_API_KEY` | API key for Google Gemini models. |

### Google Vertex AI Settings

| Setting | Environment Variable | Description |
|---------|---------------------|-------------|
| `vertex_project` | `VERTEXAI_PROJECT` | GCP project ID for Vertex AI. |
| `vertex_location` | `VERTEXAI_LOCATION` | GCP region for Vertex AI (e.g., `us-central1`). |
| `vertex_credentials` | `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON file. |

### Web Search API Keys

| Setting | Environment Variable | Description |
|---------|---------------------|-------------|
| `serpapi_key` | `SERPAPI_KEY` | API key for SerpAPI. |
| `ydc_api_key` | `YDC_API_KEY` | API key for You.com Search API. |
| `tavily_api_key` | `TAVILY_API_KEY` | API key for Tavily Search API. |

### Knowledge Settings

| Setting | Environment Variable | Description |
|---------|---------------------|-------------|
| `llama_parse_api_key` | `LLAMA_PARSE_API_KEY` | API key for LlamaParse. |
| `google_credentials_path` | `GOOGLE_CREDENTIALS_PATH` | Path to Google Credentials JSON file. |

### Logging Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `log_level` | `LOG_LEVEL` | `'INFO'` | The minimum logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `log_use_rich` | `LOG_USE_RICH` | `True` | Use rich for beautiful console output. |
| `log_format_string` | `LOG_FORMAT_STRING` | `None` | A custom format string for the console logger. |
| `log_file_path` | `LOG_FILE_PATH` | `None` | If set, logs will also be written to this file. |

### Tracing Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `tracing_mode` | `TRACING_MODE` | `'noop'` | Controls tracing provider (see below). |

**Available Tracing Modes:**

| Mode | Description |
|------|-------------|
| `noop` | Disabled (zero overhead). Default. |
| `logfire` | Pydantic Logfire for OpenTelemetry-based observability. |
| `otel` | Generic OpenTelemetry exporter. |
| `langfuse` | Langfuse LLM observability with cost tracking. |
| `opik` | Comet Opik LLM observability. |

### Logfire Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `logfire_token` | `LOGFIRE_TOKEN` | `None` | API token for Logfire hosted mode. |
| `logfire_service_name` | `LOGFIRE_SERVICE_NAME` | `'axion'` | The service name that appears in Logfire. |
| `logfire_project_name` | `LOGFIRE_PROJECT` | `None` | Optional project name for Logfire. |
| `logfire_distributed_tracing` | `DISTRIBUTED_TRACING` | `True` | Toggles distributed tracing. |
| `logfire_console_logging` | `CONSOLE_LOGGING` | `False` | Toggles Logfire's console logging. |
| `otel_endpoint` | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `None` | Custom OpenTelemetry endpoint. |

### Langfuse Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `langfuse_public_key` | `LANGFUSE_PUBLIC_KEY` | `None` | Langfuse public key for authentication. |
| `langfuse_secret_key` | `LANGFUSE_SECRET_KEY` | `None` | Langfuse secret key for authentication. |
| `langfuse_base_url` | `LANGFUSE_BASE_URL` | `'https://cloud.langfuse.com'` | Langfuse API endpoint. |

### Opik Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `opik_api_key` | `OPIK_API_KEY` | `None` | Opik API key for authentication. |
| `opik_workspace` | `OPIK_WORKSPACE` | `None` | Opik workspace name. |
| `opik_project_name` | `OPIK_PROJECT_NAME` | `'axion'` | Opik project name for grouping traces. |
| `opik_base_url` | `OPIK_URL_OVERRIDE` | `'https://www.comet.com/opik/api'` | Opik API endpoint. |

## Usage in Code

### Accessing Settings

Import the global `settings` object to access configuration values:

```python
from axion._core.environment import settings

def some_function():
    # Access settings directly
    if settings.debug:
        print("Debug mode is enabled.")

    print(f"Using LLM Provider: {settings.llm_provider}")
    print(f"Default Model: {settings.llm_model_name}")
```

### Resolving API Keys

Use `resolve_api_key()` to get API keys with proper fallback:

```python
from axion._core.environment import resolve_api_key

# Prioritizes direct argument, falls back to settings
api_key = resolve_api_key(
    api_key=None,  # or pass explicit key
    key_name='tavily_api_key',
    service_name='Tavily'
)
```

### Auto-Detecting Tracing Provider

The system can auto-detect the tracing provider from environment variables:

```python
from axion._core.environment import detect_tracing_provider, list_tracing_providers

# Auto-detect based on which API keys are set
provider = detect_tracing_provider()
print(f"Detected provider: {provider}")

# List all available providers
providers = list_tracing_providers()
# ['noop', 'logfire', 'otel', 'langfuse', 'opik']
```

**Auto-detection priority:**

1. Explicit `TRACING_MODE` environment variable
2. `LANGFUSE_SECRET_KEY` set → `'langfuse'`
3. `OPIK_API_KEY` set → `'opik'`
4. `LOGFIRE_TOKEN` set → `'logfire'`
5. `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` set → `'otel'`
6. Default → `'noop'`

## Environment Variable Examples

=== ":material-cog: Basic"

    ```bash
    # .env file
    DEBUG=false
    LOG_LEVEL=INFO
    LLM_PROVIDER=openai
    LLM_MODEL_NAME=gpt-4o
    OPENAI_API_KEY=sk-your-api-key
    ```

=== ":material-chart-timeline: Langfuse"

    ```bash
    # .env file
    OPENAI_API_KEY=sk-your-api-key
    LANGFUSE_PUBLIC_KEY=pk-lf-xxx
    LANGFUSE_SECRET_KEY=sk-lf-xxx
    # TRACING_MODE is auto-detected from LANGFUSE_SECRET_KEY
    ```

=== ":material-fire: Logfire"

    ```bash
    # .env file
    OPENAI_API_KEY=sk-your-api-key
    LOGFIRE_TOKEN=your-logfire-token
    LOGFIRE_SERVICE_NAME=my-app
    # TRACING_MODE is auto-detected from LOGFIRE_TOKEN
    ```

=== ":material-swap-horizontal: Multi-Provider"

    ```bash
    # .env file
    OPENAI_API_KEY=sk-your-openai-key
    ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
    GOOGLE_API_KEY=your-google-key

    # Vertex AI
    VERTEXAI_PROJECT=my-gcp-project
    VERTEXAI_LOCATION=us-central1
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    ```

=== ":material-magnify: Search"

    ```bash
    # .env file
    TAVILY_API_KEY=tvly-xxx
    SERPAPI_KEY=xxx
    YDC_API_KEY=xxx
    ```

## User Extension Namespace

The settings object includes an `ext` dictionary for custom user settings:

```python
from axion._core.environment import settings

# Access custom settings
custom_value = settings.ext.get('my_custom_setting', 'default')
```

---

<div class="ref-nav" markdown="1">

[Logging :octicons-arrow-right-24:](logging.md){ .md-button .md-button--primary }
[Installation :octicons-arrow-right-24:](../../getting-started/installation.md){ .md-button }
[LLM Providers :octicons-arrow-right-24:](../../guides/llm-providers.md){ .md-button }

</div>
