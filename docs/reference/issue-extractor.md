# Issue Extractor Reference

API reference for the issue extraction and signal analysis module.

## Module: `axion.reporting`

```python
from axion.reporting import (
    IssueExtractor,
    ExtractedIssue,
    IssueExtractionResult,
    IssueGroup,
    LLMSummaryInput,
    MetricSignalAdapter,
    SignalAdapterRegistry,
)
```

---

## IssueExtractor

::: axion.reporting.issue_extractor.IssueExtractor
    options:
      show_source: false
      members:
        - __init__
        - extract_from_evaluation
        - extract_from_test_result
        - extract_from_metric_score
        - to_llm_input
        - to_prompt_text
        - to_grouped_prompt_text
        - to_grouped_prompt_text_async
        - summarize
        - summarize_sync

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_threshold` | `float` | `0.0` | Signals with scores at or below this threshold are considered issues |
| `include_nan` | `bool` | `False` | Whether to include signals with NaN scores as issues |
| `include_context_fields` | `List[str]` | `['query', 'actual_output', 'expected_output']` | Fields to include from test case context |
| `metric_filters` | `List[str]` | `None` | Only extract from these metrics. If None, all metrics are processed |
| `max_issues` | `int` | `None` | Hard limit on number of issues to return |
| `sample_rate` | `float` | `None` | Deterministic sampling rate (0.0-1.0) by test_case_id |

### Methods

#### extract_from_evaluation

```python
def extract_from_evaluation(
    self,
    result: EvaluationResult
) -> IssueExtractionResult
```

Extract all issues from an evaluation result.

**Parameters:**
- `result`: The EvaluationResult from running an evaluation

**Returns:** `IssueExtractionResult` containing all extracted issues

---

#### to_prompt_text

```python
def to_prompt_text(
    self,
    result: IssueExtractionResult,
    max_issues: int = 50
) -> str
```

Generate a text prompt for LLM-based issue summarization.

**Parameters:**
- `result`: The extraction result to convert
- `max_issues`: Maximum number of detailed issues to include

**Returns:** Formatted markdown prompt text

---

#### to_grouped_prompt_text

```python
def to_grouped_prompt_text(
    self,
    result: IssueExtractionResult,
    llm: Optional[LLMRunnable] = None,
    max_groups: int = 20,
    max_examples_per_group: int = 2,
) -> str
```

Generate a grouped prompt with optional LLM summarization.

Groups similar issues together and shows representative examples, reducing context size while preserving signal quality.

**Parameters:**
- `result`: The extraction result to convert
- `llm`: Optional LLM for generating group summaries
- `max_groups`: Maximum number of issue groups to include
- `max_examples_per_group`: Representative examples per group

**Returns:** Formatted prompt text with grouped issues

---

#### to_grouped_prompt_text_async

```python
async def to_grouped_prompt_text_async(
    self,
    result: IssueExtractionResult,
    llm: Optional[LLMRunnable] = None,
    max_groups: int = 20,
    max_examples_per_group: int = 2,
) -> str
```

Async version of `to_grouped_prompt_text`. Use this when working with async LLMs.

---

#### summarize

```python
async def summarize(
    self,
    result: IssueExtractionResult,
    llm: LLMRunnable,
    prompt_template: Optional[str] = None,
    max_issues: int = 100,
) -> IssueSummary
```

Generate a complete LLM-powered summary of evaluation issues.

**Parameters:**
- `result`: The IssueExtractionResult to summarize
- `llm`: The LLM to use for generating the summary (must have `acomplete` method)
- `prompt_template`: Custom prompt template. If None, uses `DEFAULT_SUMMARY_PROMPT`. Template must include `{overview}` and `{issue_data}` placeholders.
- `max_issues`: Maximum number of issues to include in the prompt (default 100)

**Returns:** `IssueSummary` containing the LLM's analysis

**Example:**
```python
from axion.reporting import IssueExtractor
from axion.llm_registry import LLMRegistry

extractor = IssueExtractor()
issues = extractor.extract_from_evaluation(eval_result)

reg = LLMRegistry('anthropic')
llm = reg.get_llm('claude-sonnet-4-20250514')

summary = await extractor.summarize(issues, llm=llm)
print(summary.text)
```

---

#### summarize_sync

```python
def summarize_sync(
    self,
    result: IssueExtractionResult,
    llm: LLMRunnable,
    prompt_template: Optional[str] = None,
    max_issues: int = 100,
) -> IssueSummary
```

Synchronous version of `summarize()`. Use this in non-async contexts.

**Parameters:** Same as `summarize()`

**Returns:** `IssueSummary` containing the LLM's analysis

---

## SignalAdapterRegistry

Registry for metric signal adapters. Provides centralized registration and lookup of adapters.

!!! tip "Best Practice"
    Define custom adapters in the same file as your metric class. This keeps the signal schema and adapter definition in sync. See the [guide](../guides/issue-extraction.md#register-a-custom-adapter) for a complete example.

### Class Methods

#### register

```python
@classmethod
def register(cls, metric_key: str) -> Callable
```

Decorator to register a signal adapter for a metric.

**Parameters:**
- `metric_key`: The metric identifier (case-insensitive, spaces/hyphens normalized)

**Example:**
```python
@SignalAdapterRegistry.register('my_metric')
def my_adapter():
    return MetricSignalAdapter(
        metric_key='my_metric',
        headline_signals=['passed'],
        issue_values={'passed': [False]},
        context_signals=['reason'],
    )
```

---

#### register_adapter

```python
@classmethod
def register_adapter(
    cls,
    metric_key: str,
    adapter: MetricSignalAdapter,
) -> None
```

Directly register a MetricSignalAdapter for a metric.

**Parameters:**
- `metric_key`: The metric identifier
- `adapter`: The MetricSignalAdapter instance

---

#### get

```python
@classmethod
def get(cls, metric_name: str) -> Optional[MetricSignalAdapter]
```

Get the adapter for a metric by name.

**Parameters:**
- `metric_name`: The metric name (case-insensitive, spaces/hyphens normalized)

**Returns:** MetricSignalAdapter if found, None otherwise

---

#### list_adapters

```python
@classmethod
def list_adapters(cls) -> List[str]
```

List all registered adapter keys.

**Returns:** List of registered metric keys

---

## Data Classes

### ExtractedIssue

Represents a single low-score signal extracted from metric evaluation results.

```python
@dataclass
class ExtractedIssue:
    test_case_id: str           # Unique identifier for the test case
    metric_name: str            # Name of the metric that produced this signal
    signal_group: str           # Group name (e.g., "claim_0", "aspect_Coverage")
    signal_name: str            # Name of the signal (e.g., "is_covered")
    value: Any                  # Original value (False, "CONTRADICTORY", etc.)
    score: float                # Numeric score (0.0 for failures)
    description: Optional[str]  # Optional description of the signal
    reasoning: Optional[str]    # LLM reasoning from sibling signal
    item_context: Dict[str, Any]  # Context from test case
    source_path: str            # Path for debugging
    raw_signal: Dict[str, Any]  # Original signal dict
```

---

### IssueExtractionResult

Aggregated result of issue extraction from an evaluation run.

```python
@dataclass
class IssueExtractionResult:
    run_id: str                                    # Evaluation run ID
    evaluation_name: Optional[str]                 # Name of the evaluation
    total_test_cases: int                          # Total test cases analyzed
    total_signals_analyzed: int                    # Total signals analyzed
    issues_found: int                              # Total issues found
    issues_by_metric: Dict[str, List[ExtractedIssue]]  # Issues by metric name
    issues_by_type: Dict[str, List[ExtractedIssue]]    # Issues by signal type
    all_issues: List[ExtractedIssue]               # Flat list of all issues
```

---

### IssueGroup

Represents a group of similar issues for summarization.

```python
@dataclass
class IssueGroup:
    metric_name: str                          # The metric that produced these issues
    signal_name: str                          # The signal name
    total_count: int                          # Total issues in this group
    unique_values: List[Any]                  # Unique failure values
    representative_issues: List[ExtractedIssue]  # Sample issues with context
    affected_test_cases: List[str]            # List of affected test case IDs
    llm_summary: Optional[str]                # Optional LLM-generated summary
```

---

### IssueSummary

LLM-generated summary of evaluation issues, returned by `summarize()`.

```python
@dataclass
class IssueSummary:
    text: str                    # The full LLM-generated analysis
    prompt_used: str             # The prompt that was sent to the LLM
    issues_analyzed: int         # Number of issues included in analysis
    evaluation_name: Optional[str]  # Name of the evaluation
```

---

### LLMSummaryInput

Structured input for LLM-based issue summarization.

```python
@dataclass
class LLMSummaryInput:
    evaluation_name: Optional[str]        # Name of the evaluation
    total_test_cases: int                 # Total test cases analyzed
    issues_found: int                     # Total issues found
    issues_by_metric: Dict[str, int]      # Count by metric
    issues_by_type: Dict[str, int]        # Count by issue type
    detailed_issues: List[Dict[str, Any]] # Detailed issue dicts
```

---

### MetricSignalAdapter

Adapter defining how to extract issues from a specific metric's signals.

```python
@dataclass
class MetricSignalAdapter:
    metric_key: str                      # Metric identifier
    headline_signals: List[str]          # Signals that indicate pass/fail
    issue_values: Dict[str, List[Any]]   # Signal name -> failure values
    context_signals: List[str]           # Sibling signals for context
```

---

## Built-in Adapters

The following adapters are pre-registered:

| Adapter Key | Headline Signals | Issue Values |
|-------------|------------------|--------------|
| `faithfulness` | `faithfulness_verdict` | `CONTRADICTORY`, `NO_EVIDENCE` |
| `answer_criteria` | `is_covered`, `concept_coverage` | `False` |
| `answer_relevancy` | `is_relevant`, `verdict` | `False`, `no` |
| `answer_completeness` | `is_covered`, `is_addressed` | `False` |
| `factual_accuracy` | `is_correct`, `accuracy_score` | `False`, `0` |
| `answer_conciseness` | `conciseness_score` | (score-based) |
| `contextual_relevancy` | `is_relevant` | `False` |
| `contextual_recall` | `is_attributable`, `is_supported` | `False` |
| `contextual_precision` | `is_useful`, `map_score` | `False` |
| `contextual_utilization` | `is_utilized` | `False` |
| `contextual_sufficiency` | `is_sufficient` | `False` |
| `contextual_ranking` | `is_correctly_ranked` | `False` |
| `citation_relevancy` | `relevance_verdict` | `False` |
| `pii_leakage` | `pii_verdict` | `yes` |
| `tone_style_consistency` | `is_consistent` | `False` |
| `persona_tone_adherence` | `persona_match` | `False` |
| `conversation_efficiency` | `efficiency_score` | (score-based) |
| `conversation_flow` | `final_score` | (score-based) |
| `goal_completion` | `is_completed`, `goal_achieved` | `False` |
| `citation_presence` | `presence_check_passed` | `False` |
| `latency` | `latency_score` | (threshold-based) |
| `tool_correctness` | `all_tools_correct` | `False` |

---

## See Also

- [Issue Extraction Guide](../guides/issue-extraction.md) - Usage examples and best practices
- [Running Evaluations](../guides/evaluation.md) - How to run evaluations
- [Creating Custom Metrics](../deep-dives/metrics/creating-metrics.md) - Build your own metrics
