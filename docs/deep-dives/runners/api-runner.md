# API Runner

The API Runner is a centralized orchestration system for managing multiple API clients within Axion. It provides a unified interface for executing single queries and batch operations across different API endpoints with built-in retry, concurrency control, error handling, and standardized response formatting.


### APIRunner Class

The main orchestrator that manages multiple API clients and provides a unified execution interface.

**Key Features:**

- Registry-based API management
- Concurrent execution with semaphore control
- Configuration-driven initialization
- Standardized response handling
- Retry Logic


## Available API Runners

To explore all available API runners and their configuration options, use the built-in discovery method:

```python title="View Available Runners"
from axion.runners import APIRunner

# Display all registered API runners with their options
APIRunner.display()
```

## Configuration

API Runner accepts configuration in multiple formats:

```python
# Dictionary format
config = {
    'miaw': {...},
    'agent_api': {...}
}

# YAML file path
config = "/path/to/config.yaml"
```

## Usage Patterns

### Basic Usage

```python
from axion.runners import APIRunner

# Initialize with configuration
runner = APIRunner(config=config, max_concurrent=5)

# Execute single query
response = runner.execute('miaw', "What is Data Cloud?")
print(response.actual_output)

# List available APIs
available_apis = runner.list_available_apis()
print(f"Available APIs: {available_apis}")
```

### Batch Processing

```python
# Prepare multiple queries
queries = [
    "What is Data Cloud?",
    "How does Einstein Analytics work?",
    "Explain Salesforce Flow"
]

# Execute batch asynchronously
responses = await runner.execute_batch('miaw', queries)

# Process responses
for response in responses:
    print(f"Query: {response.query}")
    print(f"Response: {response.actual_output}")
    print(f"Latency: {response.latency}s")
    print("---")
```

### Direct API Usage

```python
from axion.runners import MIAWRunner

# Initialize specific runner directly
api_runner = MIAWRunner(
    config={...},
    max_concurrent=3
)

# Execute query
response = api_runner.execute("Your query here")
```

## Available API Runners

### MIAW Runner

**Registry Key:** `miaw`
**Class:** `MIAWRunner`
**Purpose:** Interface for MIAW (Messaging and In App Wep) API


**Usage Example:**
```python
config = {
    'miaw': {
        'domain': 'your-domain.salesforce.com',
        'org_id': 'your-org-id',
        'deployment_name': 'production',
        'ignore_responses': ['error', 'timeout']
    }
}

runner = APIRunner(config=config)
response = runner.execute('miaw', "Explain Salesforce Data Cloud")
```

### Agent API Runner

**Registry Key:** `agent_api`
**Class:** `AgentAPIRunner`
**Purpose:** Interface for Agent API services

**Usage Example:**
```python
config = {
    'agent_api': {
        'domain': 'your-domain.salesforce.com',
        'consumer_key': '...',
        'consumer_secret': '...',
        'agent_id': '...',
    }
}

runner = APIRunner(config=config)
response = runner.execute('agent_api', "Process this request")
```

### Prompt Template API Runner

**Registry Key:** `prompt_template`
**Class:** `PromptTemplateAPIRunner`
**Purpose:** Knowledge Prompt Template processing with retrieval capabilities


**Usage Example:**
```python
config = {
    'prompt_template': {
        'domain': 'your-domain.salesforce.com',
        'retriever_name': '...',
        '..': '...'
    }
}

runner = APIRunner(config=config)
response = runner.execute('prompt_template', "Your query")
print(f"Retrieved: {response.retrieved_content}")
print(f"Response: {response.actual_output}")
```

## Response Format

All API runners return standardized `APIResponseData` objects containing:

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Original query string |
| `actual_output` | `str` | Primary response content |
| `retrieved_content` | `List[str]` | Retrieved context chunks (if applicable) |
| `latency` | `float` | Response time in seconds |
| `trace` | `Dict[str, Any]` | Debug and trace information |
| `additional_output` | `Dict[str, Any]` | Additional response data |
| `status` | `str` | Execution status (default: 'success') |
| `timestamp` | `str` | ISO-formatted response timestamp |

## Advanced Configuratio


### Concurrency Control

```python
# Set global concurrency limit
runner = APIRunner(config=config, max_concurrent=10)

# Or configure per API when using direct instantiation
api_runner = MIAWRunner(config=api_config, max_concurrent=3)
```

### Retry Control

```python
from axion.runners.api import RetryConfig
# Set global retry control
runner = APIRunner(config=config, retry_config=RetryConfig(max_attempts=5))

# Or configure per API when using direct instantiation
api_runner = MIAWRunner(config=api_config, retry_config=RetryConfig(enabled=False))
```


### Registry Management

```python
# View available APIs
APIRunner.display()

# Check registered APIs
available = runner.list_available_apis()

# Access specific executor
miaw_executor = runner['miaw']
```

## Creating Custom API Runners

You can extend the API Runner system by creating your own custom API implementations. There are two methods for registering custom runners: using the decorator or manual registration.

### Method 1: Decorator Registration

Use the `@APIRunner.register()` decorator to automatically register your custom runner. Adding `@api_retry` will allow for retries and can be configured.

```python
from axion.runners.api import api_retry, BaseAPIRunner, APIResponseData, APIRunner

@APIRunner.register('custom_api')
class CustomAPIRunner(BaseAPIRunner):
    """Runner for CustomAPI."""
    name = 'custom_api'

    @api_retry('Custom API call')
    def execute(self, query: str, **kwargs) -> APIResponseData:
        # Your implementation here
        return APIResponseData(
            query=query,
            actual_output=f"Processed: {query}",
            status='success'
        )
```

### Method 2: Manual Registration

For dynamic registration or when you prefer explicit control:

```python
from axion.runners.api import APIRunner, BaseAPIRunner

class AnotherCustomRunner(BaseAPIRunner):
    """Another custom runner implementation."""

    def execute(self, query: str, **kwargs) -> APIResponseData:
        # Your implementation here
        return APIResponseData(
            query=query,
            actual_output=f"Processed: {query}",
            status='success'
        )

# Manual registration
APIRunner.manual_register('another_custom', AnotherCustomRunner)
```

### Implementation Requirements

When creating custom runners, ensure your implementation:

1. **Inherits from `BaseAPIRunner`**: This provides the standard interface and batch processing capabilities
2. **Implements the `execute()` method**: This is the core method that handles single query execution
3. **Returns APIResponseData objects**: Use the standardized response format for consistency


## API Reference

::: axion.runners.api
