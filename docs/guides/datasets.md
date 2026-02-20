---
icon: custom/database
---
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
dataset = Dataset.create(name="my-eval-dataset", items=[item])
```

## DatasetItem Fields

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Core Fields</p>
<p class="rule-card__desc">Query, actual/expected output, conversation history, and unique identifiers.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">Retrieval & RAG</p>
<p class="rule-card__desc">Retrieved content, actual/expected rankings for IR evaluation.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">Tool & Agent</p>
<p class="rule-card__desc">Tool calls made, expected tool calls, and custom user tags.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">4</span>
<p class="rule-card__title">Evaluation & Metadata</p>
<p class="rule-card__desc">Judgments, critiques, acceptance criteria, latency, traces, and metadata.</p>
</div>
</div>

### Core Fields

| Field | Description | Required |
|-------|-------------|----------|
| `id` | Unique identifier (auto-generated UUID7 if not provided) | No |
| `query` | The user's question or input | Yes* |
| `actual_output` | The agent's response | Yes |
| `expected_output` | Ground truth / expected answer | For some metrics |
| `conversation` | Multi-turn conversation history | Yes* |

*Either `query` or `conversation` is required.

### Retrieval & RAG Fields

| Field | Description |
|-------|-------------|
| `retrieved_content` | List of retrieved context strings |
| `actual_ranking` | Ordered list of retrieved items with scores |
| `expected_ranking` | Ground truth ranking for IR evaluation |

### Tool & Agent Fields

| Field | Description |
|-------|-------------|
| `tools_called` | Tools/functions actually called by the agent |
| `expected_tools` | Expected tool calls for evaluation |
| `user_tags` | Custom tags applied to all tool calls |

### Evaluation Fields

| Field | Description |
|-------|-------------|
| `judgment` | Binary/categorical decision (e.g., pass/fail, 1/0) |
| `critique` | Detailed explanation supporting the judgment |
| `acceptance_criteria` | List of criteria for acceptable responses |
| `latency` | Response time in seconds |

### Metadata & Tracing

| Field | Description |
|-------|-------------|
| `metadata` | Additional metadata as JSON string |
| `additional_input` | Extra key-value pairs for evaluation context |
| `additional_output` | Extra outputs from the system |
| `trace` | Execution trace information (JSON string) |
| `trace_id` | Trace ID from tracing provider |
| `observation_id` | Observation ID from tracing provider |

## Creating Datasets

### Factory Method

```python
# Create with initial items (recommended)
dataset = Dataset.create(
    name="qa-evaluation",
    description="Customer support QA pairs",
    items=[
        {"query": "How do I reset my password?", "expected_output": "..."},
        {"query": "What are your hours?", "expected_output": "..."},
    ]
)

# Create from simple query strings
dataset = Dataset.create(
    name="queries-only",
    items=["What is Python?", "How does async work?"]  # Strings become queries
)
```

### Adding Items

```python
dataset = Dataset(name="my-dataset")

# Add a single item
dataset.add_item({
    "query": "What is machine learning?",
    "expected_output": "Machine learning is..."
})

# Add multiple items at once
dataset.add_items([
    {"query": "Question 1", "actual_output": "Answer 1"},
    {"query": "Question 2", "actual_output": "Answer 2"},
])

# Ignore extra keys in your data
dataset.add_item(
    {"query": "...", "some_extra_field": "ignored"},
    ignore_extra_keys=True
)
```

## Loading Datasets

```python
# From CSV file
dataset = Dataset.read_csv("eval_data.csv")

# From CSV with column mapping
dataset = Dataset.read_csv(
    "data.csv",
    column_mapping={"question": "query", "answer": "expected_output"}
)

# From JSON file
dataset = Dataset.read_json("eval_data.json")

# From pandas DataFrame
dataset = Dataset.read_dataframe(df, name="from-pandas")
```

## Saving Datasets

```python
# Save to JSON (preserves all metadata)
dataset.to_json("output.json")

# Save to CSV
dataset.to_csv("output.csv")

# Convert to DataFrame for analysis
df = dataset.to_dataframe()

# Flatten nested JSON fields into separate columns
df = dataset.to_dataframe(flatten_nested_json=True)
```

## Multi-Turn Conversations

```python
from axion._core.schema import HumanMessage, AIMessage, ToolCall, ToolMessage
from axion.dataset_schema import MultiTurnConversation

# Using dict format (simple)
item = DatasetItem(
    conversation=[
        {"role": "user", "content": "Hello, I need help with my order"},
        {"role": "assistant", "content": "I'd be happy to help! What's your order number?"},
        {"role": "user", "content": "It's #12345"},
        {"role": "assistant", "content": "I found your order. How can I assist?"}
    ]
)

# Using typed messages (with tool calls)
conversation = MultiTurnConversation(
    messages=[
        HumanMessage(content="What's the weather in Paris?"),
        AIMessage(
            content="Let me check that for you.",
            tool_calls=[ToolCall(name="get_weather", args={"city": "Paris"})]
        ),
        ToolMessage(
            tool_call_id="...",
            content="Sunny, 22C"
        ),
        AIMessage(content="The weather in Paris is sunny and 22C.")
    ],
    reference_text="Expected: sunny weather information"  # Optional ground truth
)

item = DatasetItem(conversation=conversation)
```

### Conversation Extraction Strategy

Control how `query` and `actual_output` are extracted from multi-turn conversations:

```python
# Default: extract from last messages
item = DatasetItem(
    conversation=[...],
    conversation_extraction_strategy="last"  # Default
)

# Extract from first messages instead
item = DatasetItem(
    conversation=[...],
    conversation_extraction_strategy="first"
)
```

### Conversation Utilities

```python
# Get conversation statistics
stats = item.conversation_stats
# {'turn_count': 4, 'user_message_count': 2, 'ai_message_count': 2, 'tool_call_count': 1}

# Get the agent's execution path (tool names in order)
trajectory = item.agent_trajectory  # ['search', 'retrieve', 'summarize']

# Check for errors in tool messages
if item.has_errors:
    print("Conversation contains tool errors")

# Convert to readable transcript
transcript = item.to_transcript()
# User: What's the weather?
# Assistant: Let me check.
#   Tool Call: get_weather({"city": "Paris"})
# ...
```

## Working with DatasetItems

### Dict-like Access

```python
item = DatasetItem(query="Hello", actual_output="Hi there")

# Access fields like a dictionary
query = item["query"]
query = item.get("query", default="")

# Check if field exists
if "expected_output" in item:
    print(item["expected_output"])

# Iterate over fields
for key in item.keys():
    print(f"{key}: {item[key]}")

# Get all (key, value) pairs
for key, value in item.items():
    print(f"{key}: {value}")
```

### Creating Subsets

```python
# Get item with only specific fields
subset = item.subset(["query", "expected_output"])

# Keep the original ID
subset = item.subset(["query", "actual_output"], keep_id=True)

# Also copy judgment/critique annotations
subset = item.subset(["query"], copy_annotations=True)

# Get just evaluation-relevant fields
eval_item = item.evaluation_fields()
```

### Updating Items

```python
# Update from another item or dictionary
item.update({"actual_output": "New response", "latency": 1.5})

# Update without overwriting existing values
item.update(other_item, overwrite=False)

# Update only runtime fields (actual_output, latency, retrieved_content, etc.)
item.update_runtime(
    actual_output="Response from API",
    latency=0.5,
    retrieved_content=["Context 1", "Context 2"]
)

# Merge additional metadata
item.merge_metadata({"model": "gpt-4", "temperature": 0.7})
```

### Tool Extraction by Tag

Tool calls are automatically tagged based on their names (RAG, GUARDRAIL, LLM, DATABASE):

```python
# Add custom tags to all tool calls
item = DatasetItem(
    conversation=[...],
    user_tags=["production", "v2"]
)

# Extract tool interactions by tag
rag_interactions = item.extract_by_tag("RAG")
for tool_call, tool_message in rag_interactions:
    print(f"Called: {tool_call.name}")
    print(f"Result: {tool_message.content if tool_message else 'No response'}")
```

## Dataset Operations

### Filtering

```python
# Filter items based on a condition
failed_items = dataset.filter(
    lambda item: item.judgment == "fail",
    dataset_name="failed-cases"
)

# Filter items with errors
error_dataset = dataset.filter(lambda item: item.has_errors)

# Filter by latency
slow_items = dataset.filter(lambda item: item.latency and item.latency > 2.0)
```

### Iteration and Access

```python
# Get dataset length
print(f"Dataset has {len(dataset)} items")

# Iterate over items
for item in dataset:
    print(item.query)

# Access by index
first_item = dataset[0]
last_item = dataset[-1]

# Get item by ID
item = dataset.get_item_by_id("some-uuid")
```

### Summary Statistics

```python
# Get summary as dictionary
summary = dataset.get_summary()
# {
#     'name': 'my-dataset',
#     'total_items': 100,
#     'single_turn_items': 80,
#     'multi_turn_items': 20,
#     'has_actual_output': 100,
#     'has_expected_output': 75,
#     'created_at': '2024-01-15 10:30:00',
#     'version': '1.0'
# }

# Print formatted summary table
dataset.get_summary_table(title="Evaluation Dataset Summary")
```

## Batch API Execution

Execute queries against an API and populate `actual_output`:

```python
dataset.execute_dataset_items_from_api(
    api_name="my-rag-api",
    config="config.yaml",
    max_concurrent=5,
    show_progress=True,
    require_success=True  # Remove failed items from dataset
)
```

## Synthetic Data Generation

Generate QA pairs from documents:

```python
from axion.synthetic import GenerationParams

params = GenerationParams(
    num_questions=10,
    difficulty="medium"
)

dataset.synthetic_generate_from_directory(
    directory_path="./documents",
    llm=my_llm,
    params=params,
    embed_model=my_embedder,
    max_concurrent=3
)

# Access raw synthetic data
raw_data = dataset.synthetic_data
```

---

[Metrics & Evaluation :octicons-arrow-right-24:](metrics.md){ .md-button .md-button--primary }
[Dataset Reference :octicons-arrow-right-24:](../reference/dataset.md){ .md-button }
