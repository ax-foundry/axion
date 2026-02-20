---
icon: custom/search
---
# Issue Extraction & Analysis

Extract and analyze low-score signals from evaluation results for debugging, reporting, and LLM-powered issue summarization.

## Overview

After running evaluations, you often need to understand **why** certain test cases failed. The `IssueExtractor` automatically identifies failing signals across all metrics.

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Unified Extraction</p>
<p class="rule-card__desc">Works with any metric, built-in or custom.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Grouped Analysis</p>
<p class="rule-card__desc">Similar issues consolidated for pattern detection.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">LLM-Ready Prompts</p>
<p class="rule-card__desc">Generate summaries with optional AI analysis.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Extensible Adapters</p>
<p class="rule-card__desc">Register custom signal detection for your metrics.</p>
</div>
</div>

## Quick Start

```python
from axion.reporting import IssueExtractor
from axion.runners import evaluation_runner

# Run your evaluation
results = await evaluation_runner(dataset, metrics)

# Extract issues (signals with score <= 0)
extractor = IssueExtractor()
issues = extractor.extract_from_evaluation(results)

# View summary
print(f"Found {issues.issues_found} issues across {issues.total_test_cases} test cases")

# Generate LLM prompt for analysis
prompt = extractor.to_prompt_text(issues)
```

## Basic Usage

### Extract Issues from Evaluation Results

```python
from axion.reporting import IssueExtractor

# Default: extract signals with score <= 0.0
extractor = IssueExtractor()
issues = extractor.extract_from_evaluation(evaluation_result)

# Access extracted issues
for issue in issues.all_issues:
    print(f"Test: {issue.test_case_id}")
    print(f"Metric: {issue.metric_name}")
    print(f"Signal: {issue.signal_name} = {issue.value}")
    print(f"Score: {issue.score}")
    print(f"Reasoning: {issue.reasoning}")
    print("---")
```

### Customize Extraction

```python
extractor = IssueExtractor(
    # Extract signals with score <= 0.5 (not just 0.0)
    score_threshold=0.5,

    # Include NaN scores as issues
    include_nan=True,

    # Only extract from specific metrics
    metric_filters=['Faithfulness', 'Answer Relevancy'],

    # Limit total issues extracted
    max_issues=100,

    # Sample 10% of test cases (deterministic by ID)
    sample_rate=0.1,

    # Context fields to include from test cases
    include_context_fields=['query', 'actual_output', 'expected_output', 'retrieved_content']
)
```

### Access Issues by Category

```python
issues = extractor.extract_from_evaluation(results)

# Issues grouped by metric
for metric_name, metric_issues in issues.issues_by_metric.items():
    print(f"{metric_name}: {len(metric_issues)} issues")

# Issues grouped by type (metric:signal combination)
for issue_type, type_issues in issues.issues_by_type.items():
    print(f"{issue_type}: {len(type_issues)} issues")
```

## Generating LLM Prompts

### Basic Prompt Generation

```python
# Generate a detailed prompt listing all issues
prompt = extractor.to_prompt_text(issues, max_issues=50)
print(prompt)
```

Output:
```markdown
## Evaluation Issues Summary

**Evaluation:** RAG Quality Assessment
**Test Cases Analyzed:** 100
**Issues Found:** 23

### Issue Breakdown by Metric
- Faithfulness: 12 issues (8 CONTRADICTORY, 4 NO_EVIDENCE)
- Answer Relevancy: 7 issues (7 False)
- Answer Criteria: 4 issues (4 False)

### Detailed Issues

#### Issue 1: Faithfulness - faithfulness_verdict
- **Test Case:** test_case_42
- **Signal Group:** claim_0
- **Value:** CONTRADICTORY
- **Score:** 0.0
- **Reasoning:** "Context states Python 3.8+ required, contradicting the claim about 3.6 support"
- **Query:** "What Python versions are supported?"
- **Actual Output:** "Our product supports Python 3.6 and above..."

...

## Task
Analyze the quality issues found in this evaluation. Provide:
1. **Critical Failure Patterns:** What are the most common/severe issue types?
2. **Root Cause Analysis:** What systemic problems might be causing these failures?
3. **Recommended Improvements:** Specific actions to improve quality
4. **Priority Ranking:** Which issues should be addressed first?
```

### Grouped Prompts (Token-Efficient)

For large evaluations, use grouped prompts to reduce token usage:

```python
# Group similar issues together with representative examples
grouped_prompt = extractor.to_grouped_prompt_text(
    issues,
    max_groups=20,           # Limit number of issue groups
    max_examples_per_group=2  # Examples per group
)
```

Output:
```markdown
## Evaluation Issues Summary (Grouped)

**Total Issues Found:** 156
**Issue Groups:** 8

### Issue Groups Overview

| Metric | Signal | Count | Values |
|--------|--------|-------|--------|
| Faithfulness | faithfulness_verdict | 45 | CONTRADICTORY, NO_EVIDENCE |
| Answer Criteria | is_covered | 38 | False |
| Answer Relevancy | is_relevant | 31 | False |

### Detailed Issue Groups

#### Group 1: Faithfulness - faithfulness_verdict
- **Total Issues:** 45
- **Failure Values:** CONTRADICTORY, NO_EVIDENCE
- **Affected Tests:** test_12, test_45, test_67, ... (+42 more)

**Representative Examples:**

*Example 1:*
- Test: test_12
- Value: CONTRADICTORY
- Reasoning: "Claim contradicts source documentation"
- Query: "What are the system requirements?"

*Example 2:*
- Test: test_45
- Value: NO_EVIDENCE
- Reasoning: "No supporting evidence found in context"
- Query: "Does it support Windows?"
```

### LLM-Powered Group Summaries

Add AI-generated pattern summaries for each issue group:

```python
from axion.llm_registry import LLMRegistry

# Get an LLM instance
reg = LLMRegistry('anthropic')
llm = reg.get_llm()

# Generate grouped prompt with LLM summaries (async)
grouped_prompt = await extractor.to_grouped_prompt_text_async(
    issues,
    llm=llm,  # LLM generates 1-2 sentence summary per group
    max_groups=15
)
```

Output includes AI-generated pattern analysis:
```markdown
#### Group 1: Faithfulness - faithfulness_verdict
- **Total Issues:** 45
- **Pattern Summary:** Claims about version compatibility and system requirements
  consistently contradict the official documentation, suggesting outdated training data
  or hallucination patterns around technical specifications.
```

### Full LLM Analysis with `summarize()`

For a complete automated analysis (like copy-pasting to ChatGPT/Gemini, but automated), use the `summarize()` method:

```python
from axion.llm_registry import LLMRegistry
from axion.reporting import IssueExtractor

# Extract issues
extractor = IssueExtractor()
issues = extractor.extract_from_evaluation(results)

# Get LLM
reg = LLMRegistry('anthropic')
llm = reg.get_llm('claude-sonnet-4-20250514')

# Generate complete analysis
summary = await extractor.summarize(issues, llm=llm)

print(summary.text)
```

The default prompt generates:
- **Executive Summary** - 2-3 sentence overview
- **Missing Concepts** - Topics the AI consistently missed
- **Failure Categories Table** - Structured breakdown with counts and examples
- **Root Cause Analysis** - Systemic issues causing failures
- **Recommended Actions** - Prioritized improvements

#### Custom Prompts

Override the default prompt with your own template:

```python
custom_prompt = '''
Analyze these evaluation failures:

{overview}

{issue_data}

Provide:
1. Top 3 failure patterns
2. Quick wins to fix them
3. A table with Category, Count, and Example
'''

summary = await extractor.summarize(
    issues,
    llm=llm,
    prompt_template=custom_prompt,
    max_issues=50  # Limit issues in prompt
)
```

The template must include `{overview}` and `{issue_data}` placeholders.

#### Sync Version

For non-async code:

```python
summary = extractor.summarize_sync(issues, llm=llm)
print(summary.text)
```

### Structured Output for Programmatic Use

```python
# Get structured data instead of text
llm_input = extractor.to_llm_input(issues, max_issues=50)

print(llm_input.evaluation_name)      # "RAG Quality Assessment"
print(llm_input.total_test_cases)     # 100
print(llm_input.issues_found)         # 23
print(llm_input.issues_by_metric)     # {'Faithfulness': 12, ...}

# Access detailed issue dicts
for issue_dict in llm_input.detailed_issues:
    print(issue_dict['metric'])
    print(issue_dict['signal_name'])
    print(issue_dict['value'])
    print(issue_dict['context']['query'])
```

## Signal Adapter Registry

The `SignalAdapterRegistry` defines how to extract issues from each metric's signals. Axion includes adapters for all built-in metrics, but you can register custom adapters for your own metrics.

### How Adapters Work

Each adapter specifies:

| Field | Description | Example |
|-------|-------------|---------|
| `headline_signals` | Signals that indicate pass/fail | `['is_relevant', 'verdict']` |
| `issue_values` | Values that indicate failures | `{'is_relevant': [False], 'verdict': ['no']}` |
| `context_signals` | Related signals for context | `['statement', 'reason', 'turn_index']` |

### Built-in Adapters

```python
from axion.reporting import SignalAdapterRegistry

# List all registered adapters
print(SignalAdapterRegistry.list_adapters())
# ['faithfulness', 'answer_criteria', 'answer_relevancy', 'answer_completeness',
#  'contextual_relevancy', 'contextual_recall', 'contextual_precision',
#  'factual_accuracy', 'pii_leakage', 'tool_correctness', ...]
```

### Register a Custom Adapter

**Best Practice:** Define the adapter in the same file as your custom metric. This keeps the signal schema and adapter in sync.

```python
# my_metrics/quality_checker.py

from axion.metrics import BaseMetric
from axion.reporting import SignalAdapterRegistry, MetricSignalAdapter

# 1. Define your metric
class QualityChecker(BaseMetric):
    name = "Quality Checker"

    async def a_score(self, item):
        # Your scoring logic...
        return MetricScore(
            name=self.name,
            score=score,
            signals={
                'quality_verdict': {'value': verdict, 'score': 1.0 if verdict == 'PASS' else 0.0},
                'issues_found': {'value': issues, 'score': 1.0},
                'reasoning': {'value': reason},
            }
        )

# 2. Register adapter alongside the metric
@SignalAdapterRegistry.register('quality_checker')
def _quality_checker_adapter():
    return MetricSignalAdapter(
        metric_key='quality_checker',
        headline_signals=['quality_verdict'],
        issue_values={'quality_verdict': ['FAIL', 'PARTIAL']},
        context_signals=['issues_found', 'reasoning']
    )
```

The adapter registers automatically when your metric module is imported.

#### Alternative: Direct Registration

For quick registration without a decorator:

```python
SignalAdapterRegistry.register_adapter(
    'another_metric',
    MetricSignalAdapter(
        metric_key='another_metric',
        headline_signals=['is_valid'],
        issue_values={'is_valid': [False, 'INVALID', 'ERROR']},
        context_signals=['validation_errors', 'field_name']
    )
)
```

### What If No Adapter Exists?

Unregistered metrics still work - the extractor falls back to **score-based extraction**:

```python
# Your custom metric without a registered adapter
# Signals with score <= threshold are automatically detected as issues
```

| Feature | Without Adapter | With Adapter |
|---------|-----------------|--------------|
| Score-based detection | Yes | Yes |
| Value-based detection | No | Yes |
| Explicit headline signals | No | Yes |
| Context signal extraction | Basic | Full |

Register an adapter when you want richer issue detection beyond just scores.

## Complete Example

```python
import asyncio
from axion import Dataset
from axion.metrics import Faithfulness, AnswerRelevancy, AnswerCriteria
from axion.runners import evaluation_runner
from axion.reporting import IssueExtractor, SignalAdapterRegistry, MetricSignalAdapter
from axion.llm_registry import LLMRegistry

async def analyze_evaluation():
    # 1. Run evaluation
    dataset = Dataset.from_csv("test_cases.csv")
    results = await evaluation_runner(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), AnswerCriteria()]
    )

    # 2. Extract issues
    extractor = IssueExtractor(
        score_threshold=0.0,
        include_context_fields=['query', 'actual_output', 'expected_output']
    )
    issues = extractor.extract_from_evaluation(results)

    # 3. Quick summary
    print(f"Evaluation: {issues.evaluation_name}")
    print(f"Test cases: {issues.total_test_cases}")
    print(f"Issues found: {issues.issues_found}")
    print()

    for metric, metric_issues in issues.issues_by_metric.items():
        print(f"  {metric}: {len(metric_issues)} issues")

    # 4. Generate LLM analysis prompt
    if issues.issues_found > 0:
        # Use grouped prompt for efficiency
        reg = LLMRegistry('anthropic')
        llm = reg.get_llm('claude-sonnet-4-20250514')

        prompt = await extractor.to_grouped_prompt_text_async(
            issues,
            llm=llm,
            max_groups=10
        )

        # Send to LLM for analysis
        analysis = await llm.acomplete(prompt)
        print("\n=== LLM Analysis ===")
        print(analysis.text)

asyncio.run(analyze_evaluation())
```

## API Reference

### IssueExtractor

```python
class IssueExtractor:
    def __init__(
        self,
        score_threshold: float = 0.0,      # Signals <= this are issues
        include_nan: bool = False,          # Include NaN scores
        include_context_fields: List[str] = ['query', 'actual_output', 'expected_output'],
        metric_filters: List[str] = None,   # Only these metrics
        max_issues: int = None,             # Hard limit
        sample_rate: float = None,          # 0.0-1.0 sampling
    ): ...

    def extract_from_evaluation(self, result: EvaluationResult) -> IssueExtractionResult: ...
    def extract_from_test_result(self, test_result: TestResult, index: int) -> List[ExtractedIssue]: ...
    def extract_from_metric_score(self, metric_score: MetricScore, ...) -> List[ExtractedIssue]: ...

    def to_llm_input(self, result: IssueExtractionResult, max_issues: int = 50) -> LLMSummaryInput: ...
    def to_prompt_text(self, result: IssueExtractionResult, max_issues: int = 50) -> str: ...
    def to_grouped_prompt_text(self, result: IssueExtractionResult, llm=None, max_groups=20, max_examples_per_group=2) -> str: ...
    async def to_grouped_prompt_text_async(self, result: IssueExtractionResult, llm=None, ...) -> str: ...
```

### SignalAdapterRegistry

```python
class SignalAdapterRegistry:
    @classmethod
    def register(cls, metric_key: str) -> decorator: ...  # Decorator for registration

    @classmethod
    def register_adapter(cls, metric_key: str, adapter: MetricSignalAdapter) -> None: ...

    @classmethod
    def get(cls, metric_name: str) -> Optional[MetricSignalAdapter]: ...

    @classmethod
    def list_adapters(cls) -> List[str]: ...
```

### Data Classes

```python
@dataclass
class ExtractedIssue:
    test_case_id: str
    metric_name: str
    signal_group: str
    signal_name: str
    value: Any
    score: float
    description: Optional[str]
    reasoning: Optional[str]
    item_context: Dict[str, Any]
    source_path: str
    raw_signal: Dict[str, Any]

@dataclass
class IssueExtractionResult:
    run_id: str
    evaluation_name: Optional[str]
    total_test_cases: int
    total_signals_analyzed: int
    issues_found: int
    issues_by_metric: Dict[str, List[ExtractedIssue]]
    issues_by_type: Dict[str, List[ExtractedIssue]]
    all_issues: List[ExtractedIssue]

@dataclass
class MetricSignalAdapter:
    metric_key: str
    headline_signals: List[str]
    issue_values: Dict[str, List[Any]]
    context_signals: List[str]
```

## Best Practices

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Start with Defaults</p>
<p class="rule-card__desc"><code>IssueExtractor()</code> works well for most cases out of the box.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">Grouped Prompts</p>
<p class="rule-card__desc">Use grouped prompts for large evals &mdash; reduces tokens by 50&ndash;90%.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">Register Adapters</p>
<p class="rule-card__desc">Custom adapters enable value-based detection for your metrics.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">4</span>
<p class="rule-card__title">Sample Large Datasets</p>
<p class="rule-card__desc">Use <code>sample_rate</code> to manage volume on very large evaluations.</p>
</div>
</div>
