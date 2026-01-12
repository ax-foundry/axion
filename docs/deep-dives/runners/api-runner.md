# API Runner

The API Runner is a centralized orchestration system for managing multiple API clients within Axion. It provides a unified interface for executing single queries and batch operations across different API endpoints with built-in retry, concurrency control, error handling, and standardized response formatting.

## Key Features

- **Registry-based API management** — Register any API client with a simple decorator
- **Concurrent execution** — Semaphore-controlled parallelism for batch operations
- **Configuration-driven** — YAML or dictionary-based initialization
- **Standardized responses** — Consistent `APIResponseData` format across all APIs
- **Built-in retry logic** — Configurable retry behavior with exponential backoff

---

## Creating Custom API Runners

The API Runner system is designed to be extensible. Register your own API implementations to integrate any service into Axion's evaluation pipelines.

### Method 1: Decorator Registration (Recommended)

Use the `@APIRunner.register()` decorator to automatically register your custom runner:

```python
from axion.runners.api import api_retry, BaseAPIRunner, APIResponseData, APIRunner

@APIRunner.register('my_chatbot')
class MyChatbotRunner(BaseAPIRunner):
    """Runner for your custom chatbot API."""
    name = 'my_chatbot'

    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        self.api_url = config.get('api_url')
        self.api_key = config.get('api_key')

    @api_retry('Chatbot API call')
    def execute(self, query: str, **kwargs) -> APIResponseData:
        import requests
        import time

        start_time = time.time()

        response = requests.post(
            self.api_url,
            headers={'Authorization': f'Bearer {self.api_key}'},
            json={'message': query}
        )
        result = response.json()

        return APIResponseData(
            query=query,
            actual_output=result.get('response', ''),
            latency=time.time() - start_time,
            status='success'
        )
```

### Example: OpenAI Chat Runner

A real-world example integrating OpenAI's Chat API:

```python
from axion.runners.api import api_retry, BaseAPIRunner, APIResponseData, APIRunner

@APIRunner.register('openai_chat')
class OpenAIChatRunner(BaseAPIRunner):
    """Custom runner for OpenAI Chat API."""
    name = 'openai_chat'

    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        self.model = config.get('model', 'gpt-4')

    @api_retry('OpenAI Chat API call')
    def execute(self, query: str, **kwargs) -> APIResponseData:
        from openai import OpenAI
        import time

        client = OpenAI()
        start_time = time.time()

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}]
        )

        return APIResponseData(
            query=query,
            actual_output=response.choices[0].message.content,
            latency=time.time() - start_time,
            additional_output={'model': self.model, 'usage': dict(response.usage)},
            status='success'
        )
```

### Implementation Requirements

When creating custom runners, ensure your implementation:

1. **Inherits from `BaseAPIRunner`** — Provides the standard interface and batch processing capabilities
2. **Implements the `execute()` method** — Core method that handles single query execution
3. **Returns `APIResponseData` objects** — Use the standardized response format for consistency
4. **Handles errors gracefully** — Use try/except and return appropriate status

---

## Usage Patterns

### Basic Usage

```python
from axion.runners import APIRunner

# Configuration for your registered APIs
config = {
    'my_chatbot': {
        'api_url': 'https://api.example.com/chat',
        'api_key': 'your-api-key'
    }
}

# Initialize with configuration
runner = APIRunner(config=config, max_concurrent=5)

# Execute single query
response = runner.execute('my_chatbot', "How do I reset my password?")
print(response.actual_output)

# List available APIs
available_apis = runner.list_available_apis()
print(f"Available APIs: {available_apis}")
```

### Batch Processing

```python
# Prepare multiple queries
queries = [
    "How do I reset my password?",
    "What are the payment options?",
    "How do I contact support?"
]

# Execute batch asynchronously
responses = await runner.execute_batch('my_chatbot', queries)

# Process responses
for response in responses:
    print(f"Query: {response.query}")
    print(f"Response: {response.actual_output}")
    print(f"Latency: {response.latency}s")
    print("---")
```

### Direct API Usage

```python
# Initialize a specific runner directly
chatbot_runner = MyChatbotRunner(
    config={'api_url': '...', 'api_key': '...'},
    max_concurrent=3
)

# Execute query
response = chatbot_runner.execute("Your query here")
```

---

## Configuration

API Runner accepts configuration in multiple formats:

```python
# Dictionary format
config = {
    'my_chatbot': {
        'api_url': 'https://api.example.com/chat',
        'api_key': 'your-api-key'
    },
    'openai_chat': {
        'model': 'gpt-4'
    }
}

# Or load from YAML file
config = "/path/to/config.yaml"

runner = APIRunner(config=config)
```

### YAML Configuration Example

```yaml
# config.yaml
my_chatbot:
  api_url: https://api.example.com/chat
  api_key: ${CHATBOT_API_KEY}  # Environment variable substitution

openai_chat:
  model: gpt-4
```

---

## Response Format

All API runners return standardized `APIResponseData` objects:

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

---

## Advanced Configuration

### Concurrency Control

```python
# Set global concurrency limit
runner = APIRunner(config=config, max_concurrent=10)

# Or configure per API when using direct instantiation
api_runner = MyChatbotRunner(config=api_config, max_concurrent=3)
```

### Retry Control

```python
from axion.runners.api import RetryConfig

# Set global retry control
runner = APIRunner(
    config=config,
    retry_config=RetryConfig(max_attempts=5, backoff_factor=2.0)
)

# Or disable retries for specific runner
api_runner = MyChatbotRunner(
    config=api_config,
    retry_config=RetryConfig(enabled=False)
)
```

### Registry Management

```python
# View all registered APIs and their options
APIRunner.display()

# Check registered APIs at runtime
available = runner.list_available_apis()

# Access specific executor
chatbot_executor = runner['my_chatbot']
```

---

## Integration with Evaluation Runner

Use API runners as tasks in the evaluation pipeline:

```python
from axion.runners import evaluation_runner, APIRunner
from axion.metrics import AnswerRelevancy, Faithfulness
from axion.dataset import DatasetItem

# Configure your API
config = {'my_chatbot': {'api_url': '...', 'api_key': '...'}}
api_runner = APIRunner(config=config)

# Create evaluation dataset (without actual_output - the task will generate it)
dataset = [
    DatasetItem(
        query="How do I reset my password?",
        expected_output="Navigate to login, click 'Forgot Password', follow the email link."
    ),
    DatasetItem(
        query="What are your business hours?",
        expected_output="We are open Monday-Friday, 9 AM to 5 PM EST."
    )
]

# Define task that calls your API
def chatbot_task(item: DatasetItem) -> dict:
    response = api_runner.execute('my_chatbot', item.query)
    return {
        'response': response.actual_output,
        'latency': response.latency
    }

# Run evaluation with task
results = evaluation_runner(
    evaluation_inputs=dataset,
    task=chatbot_task,
    scoring_metrics=[AnswerRelevancy(), Faithfulness()],
    evaluation_name="Chatbot Evaluation"
)
```

---

## API Reference

::: axion.runners.api
