# Axion

![header](assets/toolkit_readme.png)


<div style="background: linear-gradient(135deg, #8B9F4F 0%, #6B7A3A 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
Moving from passive, black-box observation to active, <strong>white-box evaluation</strong>—Axion empowers builders with <strong>actionable signals</strong>, <strong>automated pipelines</strong>, and <strong>fully transparent metrics</strong>. See exactly why your agent succeeds or fails.
</p>
</div>

---

<p align="center">
<strong>White-Box Transparency</strong> |
<strong>Modular by Design</strong> |
<strong>Built for Scale</strong>
</p>




## **Component Arsenal**

<table>
<tr>
<td width="50%" valign="top">

<h3><strong>Core Primitives</strong></h3>
<strong>Structured Handlers & Tool Abstractions</strong>

<p>Build composable toolchains with pre-defined base classes for structured LLMs, tools, and knowledge retrieval. Eliminate boilerplate, enforce consistency, and focus on your logic.</p>

</td>
<td width="50%" valign="top">

<h3><strong>API Integrations</strong></h3>
<strong>Extensible Backend Access</strong>

<p>Base API classes with built-in tracing and authentication support. Build your own API integrations with ease or extend the provided abstractions.</p>

</td>
</tr>
<tr>
<td width="50%" valign="top">

<h3><strong>Evaluation Engine & Metric Suite</strong></h3>
<strong>Built-in & Open-Source Friendly</strong>

<p>Define experiments, run batch evaluations, calibrate judges, and score using our native metrics—or integrate with open libraries for broader experimentation coverage.</p>

</td>
<td width="50%" valign="top">

<h3><strong>RAG Toolbox</strong></h3>
<strong>Everything Retrieval—Chunking, Grounding, Response Assembly</strong>

<p>End-to-end support for grounding pipelines with modular components you can reuse across use cases.</p>

</td>
</tr>
<tr>
<td width="50%" valign="top">

<h3><strong>Observability at Its Core</strong></h3>
<strong>Trace, Log, Debug with Confidence</strong>

<p>Native support for Logfire, structured logging, and run tracking gives you production-grade visibility across every step of your AI pipeline.</p>

</td>
<td width="50%" valign="top">

<h3><strong>Designed for Scale</strong></h3>
<strong>Async-Native, Pydantic-Validated, Error Resilient</strong>

<p>Async support everywhere. Predictable, structured I/O with Pydantic validation. Robust error handling out-of-the-box.</p>

</td>
</tr>
</table>

---

## **Hierarchical Scoring**

<div style="background: rgba(139, 159, 79, 0.1); border-left: 3px solid #8B9F4F; padding: 16px; margin: 20px 0;">
<strong>What sets Axion apart:</strong> Our scoring framework is hierarchical by design—moving from a single overall score down into layered sub-scores. This delivers a <em>diagnostic map</em> of quality, not just a number.
</div>

```
                ┌─────────────────┐
                │  Overall Score  │
                │      0.82       │
                └────────┬────────┘
                         │
      ┌──────────────────┼──────────────────┐
      ▼                  ▼                  ▼
┌───────────┐      ┌───────────┐      ┌───────────┐
│ Relevance │      │ Accuracy  │      │   Tone    │
│   0.91    │      │   0.78    │      │   0.85    │
└───────────┘      └───────────┘      └───────────┘
```

<table>
<tr>
<td width="50%" valign="top">

**Instant Root Cause Diagnosis**

Drill down to pinpoint whether issues stem from relevance, accuracy, tone, or other dimensions—no more guessing from flat scores.

</td>
<td width="50%" valign="top">

**Strategic Prioritization**

Forces clarity on what really matters for your business by breaking quality into weighted layers.

</td>
</tr>
<tr>
<td width="50%" valign="top">

**Actionable Feedback Loop**

Each layer translates directly into actions—retraining, prompt adjustments, or alignment tuning.

</td>
<td width="50%" valign="top">

**Customizable to Business Goals**

Weight and expand dimensions to match your unique KPIs. Define what "good AI" means for you.

</td>
</tr>
</table>

```python
from axion.runners import evaluation_runner
from axion.metrics import AnswerRelevancy
from axion.dataset import DatasetItem

# Define hierarchical scoring configuration
config = {
    'metric': {
        'Relevance': AnswerRelevancy(metric_name='Relevancy'),
    },
    'model': {
        'ANSWER_QUALITY': {'Relevance': 1.0},
    },
    'weights': {
        'ANSWER_QUALITY': 1.0,
    }
}

results = evaluation_runner(
    evaluation_inputs=[data_item],
    scoring_config=config,  # Or pass path to config.yaml
)

# Generate scorecard with hierarchical breakdown
results.to_scorecard()
```

[Learn more about Hierarchical Scoring →](guides/hierarchical-scoring.md)

---

<div align="center" markdown="1">

### **Think of Axion as your Lego set for AI systems:**



<div align="center" markdown="1">

[![Modular](https://img.shields.io/badge/-Modular-8B9F4F?style=for-the-badge)](#)
[![Composable](https://img.shields.io/badge/-Composable-6B7A3A?style=for-the-badge)](#)
[![Production Ready](https://img.shields.io/badge/-Production%20Ready-A4B86C?style=for-the-badge)](#)
[![Open Source](https://img.shields.io/badge/-Open%20Source-B8C78A?style=for-the-badge)](#)

</div>
