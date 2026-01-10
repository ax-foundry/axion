# Working with Datasets

Axion uses `Dataset` and `DatasetItem` classes to manage evaluation data for both single-turn and multi-turn conversations.

## Quick Start

```python
from axion import Dataset, DatasetItem

# Create a single evaluation item
item = DatasetItem(
    query="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    expected_output="Paris is the capital of France.",
    retrieved_content=["France is a country in Europe. Paris is its capital."]
)

# Create a dataset
dataset = Dataset(items=[item])
```

## DatasetItem Fields

| Field | Description | Required |
|-------|-------------|----------|
| `query` | The user's question or input | Yes |
| `actual_output` | The agent's response | Yes |
| `expected_output` | Ground truth / expected answer | For some metrics |
| `retrieved_content` | Retrieved context (for RAG) | For retrieval metrics |
| `conversation` | Multi-turn conversation history | For conversational metrics |
| `tools_called` | Tools/functions called by agent | For tool metrics |
| `expected_tools` | Expected tool calls | For tool metrics |

## Loading Datasets

```python
# From JSON file
dataset = Dataset.from_json("eval_data.json")

# From pandas DataFrame
dataset = Dataset.from_dataframe(df)

# From list of dicts
dataset = Dataset.from_records([
    {"query": "...", "actual_output": "..."},
    {"query": "...", "actual_output": "..."}
])
```

## Multi-Turn Conversations

```python
item = DatasetItem(
    conversation=[
        {"role": "user", "content": "Hello, I need help with my order"},
        {"role": "assistant", "content": "I'd be happy to help! What's your order number?"},
        {"role": "user", "content": "It's #12345"},
        {"role": "assistant", "content": "I found your order. How can I assist?"}
    ]
)
```

## Serialization

```python
# Save to JSON
dataset.to_json("output.json")

# Convert to DataFrame
df = dataset.to_dataframe()
```

## Next Steps

- [Metrics & Evaluation](metrics.md) - Learn how to evaluate your dataset
- [API Reference: Dataset](../reference/dataset.md) - Full API documentation
