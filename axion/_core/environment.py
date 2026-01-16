import os
from typing import Any, List, Optional

from dotenv import find_dotenv
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from pydantic_settings import BaseSettings, SettingsConfigDict

from axion._core.schema import RichEnum

###################################
# .env File Loading Logic
# 1. It first checks for an environment variable `ENV_PATH` for an explicit file path.
# 2. If `ENV_PATH` is not set or the file doesn't exist, it falls back to `find_dotenv()`,
#    which automatically searches for a `.env` file in the current and parent directories.
###################################


env_path_from_var = os.getenv('ENV_PATH')
dotenv_path = (
    env_path_from_var
    if env_path_from_var and os.path.exists(env_path_from_var)
    else find_dotenv()
)


class TracingMode(str, RichEnum):
    """
    Enum to control tracing behavior.

    Available providers:
        - noop: No-op tracer (default, zero overhead)
        - logfire: Pydantic Logfire (local console or hosted with token)
        - otel: Generic OpenTelemetry exporter
        - langfuse: Langfuse LLM observability
        - opik: Comet Opik LLM observability

    Note: The old logfire_local, logfire_hosted, logfire_otel modes are deprecated.
          Use 'logfire' or 'otel' instead - behavior is auto-detected from env vars.
    """

    NOOP = 'noop'
    LOGFIRE = 'logfire'
    OTEL = 'otel'
    LANGFUSE = 'langfuse'
    OPIK = 'opik'

    # Deprecated aliases (kept for backward compatibility)
    LOGFIRE_LOCAL = 'logfire'  # Maps to logfire
    LOGFIRE_HOSTED = 'logfire'  # Maps to logfire
    LOGFIRE_OTEL = 'otel'  # Maps to otel


class Port(int):
    """Custom type for a network port, ensuring the value is within the valid range."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        def validate_port(v: int) -> int:
            if not 1 <= v <= 65535:
                raise ValueError('Port must be between 1 and 65535')
            return v

        return core_schema.no_info_after_validator_function(
            validate_port, core_schema.int_schema()
        )


class LogLevel(str):
    """Custom type for log levels, ensuring the value is one of the standard levels."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        def validate_log_level(v: str) -> str:
            valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
            v_upper = v.upper()
            if v_upper not in valid_levels:
                raise ValueError(f'Log level must be one of: {valid_levels}')
            return v_upper

        return core_schema.no_info_after_validator_function(
            validate_log_level, core_schema.str_schema()
        )


###################################
# Core Configuration Schema
###################################
class AxionConfig(BaseModel):
    """Defines the configuration schema for the Axion package.
    This class does not load from the environment; it only defines the data shape.
    """

    # App Settings
    debug: bool = Field(
        default=False,
        description='Enable debug mode for verbose logging and diagnostics.',
    )
    port: Port = Field(
        default=8000, description='The port the application will run on.'
    )
    hosts: List[str] = Field(
        default_factory=lambda: ['localhost'],
        description='A list of allowed hostnames.',
    )

    # LLM
    llm_provider: str = Field(
        default='openai', description='Default provider for language models.'
    )
    embedding_provider: str = Field(
        default='openai', description='Default provider for embedding models.'
    )
    llm_model_name: str = Field(
        default='gpt-4o', description='Default language model name.'
    )
    embedding_model_name: str = Field(
        default='text-embedding-ada-002', description='Default embedding model name.'
    )
    api_base_url: Optional[str] = Field(
        default=None, description='Optional base URL for compatible APIs.'
    )

    litellm_verbose: bool = Field(
        default=False,
        description='Enable verbose logging for LiteLLM. Set to True for debugging LLM calls.',
    )

    openai_api_key: Optional[str] = Field(
        default=None, description='API key for OpenAI models (used by LiteLLM).'
    )
    anthropic_api_key: Optional[str] = Field(
        default=None, description='API key for Anthropic Claude models.'
    )
    google_api_key: Optional[str] = Field(
        default=None, description='API key for Google Gemini models.'
    )

    # Google Vertex AI
    vertex_project: Optional[str] = Field(
        default=None,
        validation_alias='VERTEXAI_PROJECT',
        description='GCP project ID for Vertex AI.',
    )
    vertex_location: Optional[str] = Field(
        default=None,
        validation_alias='VERTEXAI_LOCATION',
        description='GCP region for Vertex AI (e.g., us-central1).',
    )
    vertex_credentials: Optional[str] = Field(
        default=None,
        validation_alias='GOOGLE_APPLICATION_CREDENTIALS',
        description='Path to GCP service account JSON for Vertex AI.',
    )

    # Web Search
    serpapi_key: Optional[str] = Field(default=None, description='API key for SerpAPI.')
    ydc_api_key: Optional[str] = Field(
        default=None, description='API key for You.com Search API.'
    )
    tavily_api_key: Optional[str] = Field(
        default=None, description='API key for Tavily Search API.'
    )

    # Knowledge
    llama_parse_api_key: Optional[str] = Field(
        default=None, description='API key for LlamaParse.'
    )
    google_credentials_path: Optional[str] = Field(
        default=None, description='Path to Google Credentials JSON file.'
    )

    # Logging Settings
    log_level: LogLevel = Field(
        default='INFO', description='The minimum logging level.'
    )
    log_use_rich: bool = Field(
        default=True, description='Use rich for beautiful, formatted logging output.'
    )
    log_format_string: Optional[str] = Field(
        default=None, description='A custom format string for the console logger.'
    )
    log_file_path: Optional[str] = Field(
        default=None, description='If set, logs will also be written to this file.'
    )

    # Tracing Settings
    tracing_mode: TracingMode = Field(
        default=TracingMode.NOOP,
        description=(
            "Controls tracing provider: 'noop' (disabled), 'logfire' (Pydantic Logfire), "
            "'otel' (OpenTelemetry), 'langfuse' (Langfuse), or 'opik' (Comet Opik). "
            'If not set, auto-detects from provider-specific env vars.'
        ),
    )
    otel_endpoint: Optional[str] = Field(
        default=None, alias='OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'
    )
    logfire_token: Optional[str] = Field(default=None)
    logfire_service_name: str = Field(default='axion')
    logfire_project_name: Optional[str] = Field(default=None, alias='LOGFIRE_PROJECT')
    logfire_distributed_tracing: bool = Field(default=True, alias='DISTRIBUTED_TRACING')
    logfire_console_logging: bool = Field(default=False, alias='CONSOLE_LOGGING')

    # Langfuse Settings
    langfuse_public_key: Optional[str] = Field(
        default=None,
        alias='LANGFUSE_PUBLIC_KEY',
        description='Langfuse public key for authentication.',
    )
    langfuse_secret_key: Optional[str] = Field(
        default=None,
        alias='LANGFUSE_SECRET_KEY',
        description='Langfuse secret key for authentication.',
    )
    langfuse_base_url: str = Field(
        default='https://cloud.langfuse.com',
        alias='LANGFUSE_BASE_URL',
        description='Langfuse API endpoint (cloud.langfuse.com for EU, us.cloud.langfuse.com for US).',
    )

    # Opik (Comet) Settings
    opik_api_key: Optional[str] = Field(
        default=None,
        alias='OPIK_API_KEY',
        description='Opik API key for authentication.',
    )
    opik_workspace: Optional[str] = Field(
        default=None,
        alias='OPIK_WORKSPACE',
        description='Opik workspace name.',
    )
    opik_project_name: Optional[str] = Field(
        default='axion',
        alias='OPIK_PROJECT_NAME',
        description='Opik project name for grouping traces.',
    )
    opik_base_url: str = Field(
        default='https://www.comet.com/opik/api',
        alias='OPIK_URL_OVERRIDE',
        description='Opik API endpoint.',
    )


###################################
# Settings Initialization
###################################
class AppSettings(BaseSettings, AxionConfig):
    """Application settings that load from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix='',
        case_sensitive=False,
        validate_assignment=True,
        extra='allow',
        env_file=dotenv_path,
        env_file_encoding='utf-8',
    )

    # User extension namespace
    ext: dict[str, Any] = Field(default_factory=dict)


# Global Settings
settings = AppSettings()


def _configure_litellm() -> None:
    """Configure LiteLLM logging based on settings."""
    import logging

    try:
        import litellm

        litellm.set_verbose = settings.litellm_verbose
        litellm.suppress_debug_info = not settings.litellm_verbose

        # Set LiteLLM logger level based on verbose setting
        litellm_logger = logging.getLogger('LiteLLM')
        if settings.litellm_verbose:
            litellm_logger.setLevel(logging.DEBUG)
        else:
            litellm_logger.setLevel(logging.WARNING)

    except ImportError:
        pass  # LiteLLM not installed, skip configuration


# Apply LiteLLM configuration on module load
_configure_litellm()


def resolve_api_key(
    api_key: Optional[str], key_name: str, service_name: Optional[str] = None
) -> str:
    """
    Resolves an API key by prioritizing a direct argument over global settings.

    Args:
        api_key: The API key passed directly to a function/class.
        key_name: The name of the key attribute in the global settings
            object (e.g., 'tavily_api_key').
        service_name: The user-friendly name of the service for error
            messages (e.g., 'Tavily').

    Returns:
        The resolved API key.
    """
    # 1. Prioritize the direct argument
    if api_key:
        return api_key

    # 2. Fall back to the global settings object
    key_name_lower = key_name.lower()
    resolved_key = getattr(settings, key_name_lower, None)
    if resolved_key:
        return resolved_key

    # 3. If still not found, construct an error message
    display_service_name = (
        service_name or key_name_lower.replace('_key', '').replace('_', ' ').title()
    )

    env_var_suggestion = key_name.upper()

    raise ValueError(
        f'{display_service_name} API key not found. Please provide it as an '
        f'argument or set the {env_var_suggestion} environment variable.'
    )


def detect_tracing_provider() -> str:
    """
    Auto-detect tracing provider from environment variables.

    This function checks for provider-specific environment variables and returns
    the appropriate provider name. It enables zero-config usage where users only
    need to set their provider's API key.

    Priority order:
    1. Explicit TRACING_MODE env var (if set to non-default value)
    2. Auto-detect from provider-specific keys:
       - LANGFUSE_SECRET_KEY → 'langfuse'
       - OPIK_API_KEY → 'opik'
       - LOGFIRE_TOKEN → 'logfire'
       - OTEL_EXPORTER_OTLP_TRACES_ENDPOINT → 'otel'
    3. Default to 'noop'

    Returns:
        Provider name string: 'noop', 'logfire', 'otel', 'langfuse', or 'opik'

    Example:
        >>> # With LANGFUSE_SECRET_KEY set in environment
        >>> detect_tracing_provider()
        'langfuse'
    """
    # Check if explicit tracing mode is set (not default noop)
    explicit_mode = os.getenv('TRACING_MODE') or os.getenv('AXION_TRACING_MODE')
    if explicit_mode and explicit_mode.lower() != 'noop':
        return explicit_mode.lower()

    # Auto-detect from provider-specific env vars
    if os.getenv('LANGFUSE_SECRET_KEY'):
        return 'langfuse'
    if os.getenv('OPIK_API_KEY'):
        return 'opik'
    if os.getenv('LOGFIRE_TOKEN'):
        return 'logfire'
    if os.getenv('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'):
        return 'otel'

    # Default to noop
    return 'noop'


def list_tracing_providers() -> list[str]:
    """
    List all available tracing providers.

    Returns:
        List of provider names that can be used with configure_tracing().

    Example:
        >>> list_tracing_providers()
        ['noop', 'logfire', 'otel', 'langfuse', 'opik']
    """
    return ['noop', 'logfire', 'otel', 'langfuse', 'opik']
