# Axion Environment & Settings

This guide details the flexible configuration management system for Axion, powered by pydantic-settings. The system is designed to work seamlessly whether Axion is running as a standalone package or as a plugin within a host application like MLPTK.

## Core Concepts

The configuration system is built on several key principles:

**Schema-First Design**: The `AxionConfig` class serves as the central schema. It's a pydantic.BaseModel that defines the shape of the configuration—all available fields, their types, and default values. It does not load any environment variables itself.

**Dynamic Operating Modes**: A single environment variable, `AXION_MODE`, determines how the settings are loaded, allowing the package to adapt to its environment without any code changes.

**Centralized Loading**: A single, global settings object is created at startup. This object is the source of truth for all configuration values throughout the application.


## Operating Modes

The configuration system has two modes, controlled by the `AXION_MODE` environment variable.

### 1. Standalone Mode

This is the mode for running Axion as a self-contained application.

**How to Activate**: Set the environment variable `AXION_MODE=standalone`.

**Behavior**: The settings loader will look for environment variables prefixed with `AXION_`.

**.env Example**:

```bash
# Used when AXION_MODE=standalone
AXION_DEBUG=true
AXION_LLM_PROVIDER="llm_gateway"
AXION_GATEWAY_API_KEY="your-secret-key"
```

### 2. Plugin Mode (Default)

This is the default mode, designed for when Axion is used as a plugin inside a host application like MLPTK.

**How to Activate**: This is the default behavior. You can explicitly set `AXION_MODE=plugin` or simply leave the variable unset.

**Behavior**: The settings are expected to be nested under the host application's prefix (`MLPTK_`). Pydantic uses a double underscore (`__`) as a delimiter for nesting.

**.env Example**:

```bash
# Used when running in plugin mode (default)
MLPTK_AXION__DEBUG=true
MLPTK_AXION__LLM_PROVIDER="llm_gateway"
MLPTK_AXION__GATEWAY_API_KEY="your-secret-key"
```

## Configuration Schema (AxionConfig)

The following settings are available. They are loaded from environment variables (case-insensitive) or a `.env` file.

### General Settings

| Setting | Environment Variable Suffix | Description |
|---|---|---|
| `debug` | `_DEBUG` | Enable debug mode for verbose logging and diagnostics. |
| `port` | `_PORT` | The port the application will run on. |
| `hosts` | `_HOSTS` | A list of allowed hostnames. |

### LLM & Embedding Settings

| Setting | Environment Variable Suffix | Description |
|---|---|---|
| `llm_provider` | `_LLM_PROVIDER` | Default provider for language models. |
| `embedding_provider` | `_EMBEDDING_PROVIDER` | Default provider for embedding models. |
| `llm_model_name` | `_LLM_MODEL_NAME` | Default language model name. |
| `embedding_model_name` | `_EMBEDDING_MODEL_NAME` | Default embedding model name. |

### API Key Settings

| Setting                   | Environment Variable Suffix | Description                                |
|---------------------------|-----------------------------|--------------------------------------------|
| `api_base_url`            | `_API_BASE_URL`             | Optional base URL for LLM compatible APIs. |
| `gateway_api_key`         | `_GATEWAY_API_KEY`          | API key for the gateway or platform.       |
| `serpapi_key`             | `_SERPAPI_KEY`              | API key for SerpAPI.                       |
| `ydc_api_key`             | `_YDC_API_KEY`              | API key for You.com Search API.            |
| `tavily_api_key`          | `_TAVILY_API_KEY`           | API key for Tavily Search API.             |
| `llama_parse_api_key`     | `_LLAMA_PARSE_API_KEY`      | API key for LlamaParse.                    |
| `google_credentials_path` | `_GOOGLE_CREDENTIALS_PATH`  | Path to Google Credentials JSON file.      |

### Logging Settings

| Setting | Environment Variable Suffix | Description |
|---|---|---|
| `log_level` | `_LOG_LEVEL` | The minimum logging level (e.g., 'DEBUG'). |
| `log_use_rich` | `_LOG_USE_RICH` | Use rich for beautiful console output. |
| `log_format_string` | `_LOG_FORMAT_STRING` | A custom format string for the console logger. |
| `log_file_path` | `_LOG_FILE_PATH` | If set, logs will also be written to this file. |

### Tracing & Logfire Settings

| Setting | Environment Variable Suffix | Description |
|---|---|---|
| `tracing_mode` | `_TRACING_MODE` | Controls tracing: 'noop', 'logfire_local', 'logfire_hosted', or 'logfire_otel'. |
| `otel_endpoint` | `_OTEL_ENDPOINT` | Custom OpenTelemetry endpoint for logfire_otel mode. |
| `logfire_token` | `_LOGFIRE_TOKEN` | API token required for logfire_hosted mode. |
| `logfire_service_name` | `_LOGFIRE_SERVICE_NAME` | The service name that appears in Logfire. |
| `logfire_project_name` | `_LOGFIRE_PROJECT_NAME` | Optional project name for logfire_hosted mode. |
| `logfire_distributed_tracing` | `_LOGFIRE_DISTRIBUTED_TRACING` | Toggles distributed tracing. |
| `logfire_console_logging` | `_LOGFIRE_CONSOLE_LOGGING` | Toggles Logfire's console logging. |

## Usage in Code

### Accessing Settings

Always import the global settings object to access configuration values.

```python
from axion._core.environment import settings, mode

def some_function():
    # Access settings directly
    if settings.debug:
        print("Debug mode is enabled.")

    api_key = settings.gateway_api_key
    print(f"Using LLM Provider: {settings.llm_provider}")

    # In plugin mode, you can also access host-specific settings
    if mode == 'plugin':
        from axion._core.environment import host_settings
        print(f"Running on MLPTK Version: {host_settings.mlptk_version}")
```

## Configuration Loading Order

The system loads settings with a clear order of precedence:

1. **Direct Arguments**: Values passed directly to the settings class constructor (highest priority)
2. **Environment Variables**: System environment variables override .env file values
3. **.env File**: Values loaded from the .env file
4. **Default Values**: Default values defined in the AxionConfig schema (lowest priority)

## Environment Variable Examples

### Standalone Mode

```bash
# .env file for standalone mode
AXION_DEBUG=true
AXION_LOG_LEVEL=DEBUG
AXION_LLM_PROVIDER=llm_gateway
AXION_GATEWAY_API_KEY=your-api-key
```

### Plugin Mode

```bash
# .env file for plugin mode (default)
MLPTK_AXION__DEBUG=true
MLPTK_AXION__LOG_LEVEL=DEBUG
MLPTK_AXION__LLM_PROVIDER=llm_gateway
MLPTK_AXION__GATEWAY_API_KEY=your-api-key
```
