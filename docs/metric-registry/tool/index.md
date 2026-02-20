---
icon: custom/overview
---
# Tool Metrics

<div style="border-left: 4px solid #8b5cf6; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluate AI agent tool calling correctness and effectiveness</strong><br>
<span class="badge" style="margin-top: 0.5rem; background: #8b5cf6;">1 Metric</span>
<span class="badge" style="background: #667eea;">Agent</span>
</div>

Tool metrics evaluate the correctness and effectiveness of tool usage in AI agent workflows. These metrics assess whether agents correctly invoke the right tools with appropriate parameters.

---

## Available Metrics

<div class="grid-container">

<div class="grid-item">
<strong><a href="tool_correctness/">Tool Correctness</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Evaluate if expected tools were correctly called</p>
<code>tools_called</code> <code>expected_tools</code>
</div>

</div>

---

## Quick Reference

| Metric | Score Range | Threshold | Key Question |
|--------|-------------|-----------|--------------|
| **Tool Correctness** | 0.0 – 1.0 | 0.5 | Were the right tools called correctly? |

---

## Usage Example

```python
from axion.metrics import ToolCorrectness
from axion.runners import MetricRunner
from axion.dataset import DatasetItem
from axion._core.schema import ToolCall

# Create evaluation item
item = DatasetItem(
    tools_called=[
        ToolCall(name="search", args={"query": "weather in Paris"}),
        ToolCall(name="format", args={"style": "brief"}),
    ],
    expected_tools=[
        ToolCall(name="search", args={"query": "weather in Paris"}),
        ToolCall(name="format", args={"style": "brief"}),
    ],
)

# Initialize metric
metric = ToolCorrectness(
    check_parameters=True,
    parameter_matching_strategy='exact'
)

# Run evaluation
runner = MetricRunner(metrics=[metric])
results = await runner.run([item])

print(f"Tool Correctness: {results[0].score:.2f}")
# Output: Tool Correctness: 1.00
```

---

## Evaluation Modes

Tool Correctness supports multiple evaluation strategies:

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #3b82f6;">
<strong style="color: #3b82f6;">Name Only (Default)</strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Just verify the correct tools were called. Parameters are ignored.</p>
```python
metric = ToolCorrectness()
```
</div>

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">With Parameters</strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Validate both tool names and their arguments.</p>
```python
metric = ToolCorrectness(
    check_parameters=True,
    parameter_matching_strategy='exact'
)
```
</div>

<div class="grid-item" style="border-left: 4px solid #f59e0b;">
<strong style="color: #f59e0b;">Strict Order</strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Tools must be called in the exact expected sequence.</p>
```python
metric = ToolCorrectness(strict_order=True)
```
</div>

<div class="grid-item" style="border-left: 4px solid #8b5cf6;">
<strong style="color: #8b5cf6;">Fuzzy Parameters</strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Allow similar (but not identical) parameter values.</p>
```python
metric = ToolCorrectness(
    check_parameters=True,
    parameter_matching_strategy='fuzzy',
    fuzzy_threshold=0.8
)
```
</div>

</div>

---

## Why Tool Metrics?

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🤖</span>
<strong>Agent Evaluation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Verify AI agents select the right tools for tasks.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔧</span>
<strong>Function Calling</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Test LLM function calling capabilities.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📊</span>
<strong>Workflow Validation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensure multi-step workflows execute correctly.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🧪</span>
<strong>Regression Testing</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Catch breaking changes in agent behavior.</p>
</div>

</div>
