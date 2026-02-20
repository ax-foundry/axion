---
icon: custom/overview
---
# Axion

<figure markdown="span">
  ![Axion](assets/axion_main_transparent.png){ width="600" }
</figure>


<div class="callout-dark" markdown="0">
<p><strong>White-box evaluation</strong>—Axion empowers builders with <strong>actionable signals</strong>, <strong>automated pipelines</strong>, and <strong>fully transparent metrics</strong>. See exactly why your agent succeeds or fails.</p>
</div>

<div class="seatbelt-hero">
  <div class="seatbelt-hero__text">
    <span class="seatbelt-hero__label">Philosophy</span>
    <p class="seatbelt-hero__quote">
      Agents are sports cars.<br>
      <em>Evals are the seatbelt.</em>
    </p>
    <p class="seatbelt-hero__sub">
      It won't make you faster. It won't win you anything. But it will keep a bad release from turning into a public incident. If you skip evals because you're a "good driver," you're not a serious person.
    </p>
  </div>
  <div class="seatbelt-hero__visual">
    <img src="assets/seatbelt.png" alt="Safety gear, not horsepower" loading="lazy">
  </div>
</div>

<div class="rule-grid">
  <div class="rule-card">
    <span class="rule-card__number">1</span>
    <p class="rule-card__title">Evals are adult supervision</p>
    <p class="rule-card__desc">Not rocket science. Simple checks that prevent simple disasters. Start here, stay here, and expand only when the basics are solid.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">2</span>
    <p class="rule-card__title">Define "good" in one sentence</p>
    <p class="rule-card__desc">If you can't articulate what success looks like, no framework will save you. Clarity first, tooling second.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">3</span>
    <p class="rule-card__title">Start with pass/fail</p>
    <p class="rule-card__desc">Add nuance only after you've earned it. A binary gate catches more failures than a sophisticated rubric you never run.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">4</span>
    <p class="rule-card__title">Calibrate your judges</p>
    <p class="rule-card__desc">LLM judges are useful. They're also liars with confidence. Calibrate them against humans or don't pretend you measured anything.</p>
  </div>
</div>

[Evaluation Flywheel :octicons-arrow-right-24:](evaluation_flywheel.md){ .md-button .md-button--primary }
[Why Ground Truth Matters :octicons-arrow-right-24:](why_ground_truth_matters.md){ .md-button }

---

## Component Arsenal

<div class="feature-grid" markdown="0">
<div class="feature-card">
<h3>Core Primitives</h3>
<strong>Structured Handlers & Tool Abstractions</strong>
<p>Build composable toolchains with pre-defined base classes for structured LLMs, tools, and knowledge retrieval. Eliminate boilerplate, enforce consistency, and focus on your logic.</p>
</div>
<div class="feature-card">
<h3>API Integrations</h3>
<strong>Extensible Backend Access</strong>
<p>Base API classes with built-in tracing and authentication support. Build your own API integrations with ease or extend the provided abstractions.</p>
</div>
<div class="feature-card">
<h3>Evaluation Engine & Metric Suite</h3>
<strong>Built-in & Open-Source Friendly</strong>
<p>Define experiments, run batch evaluations, calibrate judges, and score using our native metrics—or integrate with open libraries for broader experimentation coverage.</p>
</div>
<div class="feature-card">
<h3>RAG Toolbox</h3>
<strong>Everything Retrieval—Chunking, Grounding, Response Assembly</strong>
<p>End-to-end support for grounding pipelines with modular components you can reuse across use cases.</p>
</div>
<div class="feature-card">
<h3>Observability at Its Core</h3>
<strong>Trace, Log, Debug with Confidence</strong>
<p>Native support for Logfire, structured logging, and run tracking gives you production-grade visibility across every step of your AI pipeline.</p>
</div>
<div class="feature-card">
<h3>Designed for Scale</h3>
<strong>Async-Native, Pydantic-Validated, Error Resilient</strong>
<p>Async support everywhere. Predictable, structured I/O with Pydantic validation. Robust error handling out-of-the-box.</p>
</div>
</div>

---

## **Hierarchical Scoring**

!!! tip "What sets Axion apart"
    Our scoring framework is hierarchical by design—moving from a single overall score down into layered sub-scores. This delivers a *diagnostic map* of quality, not just a number.

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

<div class="rule-grid" markdown="0">
  <div class="rule-card">
    <p class="rule-card__title">Instant Root Cause Diagnosis</p>
    <p class="rule-card__desc">Drill down to pinpoint whether issues stem from relevance, accuracy, tone, or other dimensions—no more guessing from flat scores.</p>
  </div>
  <div class="rule-card">
    <p class="rule-card__title">Strategic Prioritization</p>
    <p class="rule-card__desc">Forces clarity on what really matters for your business by breaking quality into weighted layers.</p>
  </div>
  <div class="rule-card">
    <p class="rule-card__title">Actionable Feedback Loop</p>
    <p class="rule-card__desc">Each layer translates directly into actions—retraining, prompt adjustments, or alignment tuning.</p>
  </div>
  <div class="rule-card">
    <p class="rule-card__title">Customizable to Business Goals</p>
    <p class="rule-card__desc">Weight and expand dimensions to match your unique KPIs. Define what "good AI" means for you.</p>
  </div>
</div>

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

<div class="etymology" markdown="0">
<div class="etymology__text">
<span class="etymology__label">Origin</span>
<h2 class="etymology__heading">Why "Axion"?</h2>
<p class="etymology__acronym"><strong>A</strong>gent <strong>X</strong>-Ray <strong>I</strong>nspection &amp; <strong>O</strong>ptimization <strong>N</strong>etwork</p>
<p class="etymology__body">The name draws inspiration from the <a href="https://en.wikipedia.org/wiki/Axion">axion</a>&mdash;a hypothetical particle in physics proposed to solve the "strong CP problem" in quantum chromodynamics. Physicists Frank Wilczek and Steven Weinberg named it after a laundry detergent, hoping it would "clean up" their theoretical mess.</p>
</div>
<div class="etymology__aside">
<span class="etymology__parallel-label">Particle &harr; Toolkit</span>
<table class="etymology__table">
<tr>
<td class="etymology__parallel-icon">&#x2736;</td>
<td>
<p class="etymology__parallel-title">Incredibly small, immensely powerful</p>
<p class="etymology__parallel-desc">Axions may account for the universe's dark matter through sheer numbers. This toolkit offers small, focused tools that combine to tackle AI evaluation at scale.</p>
</td>
</tr>
<tr>
<td class="etymology__parallel-icon">&#x2727;</td>
<td>
<p class="etymology__parallel-title">Designed to clean things up</p>
<p class="etymology__parallel-desc">Named after a detergent to "clean up" a theoretical mess. Built to bring clarity and structure to the messy problem of agent evaluation.</p>
</td>
</tr>
<tr>
<td class="etymology__parallel-icon">&#x2B22;</td>
<td>
<p class="etymology__parallel-title">Modular by nature</p>
<p class="etymology__parallel-desc">Lightweight components that work together to solve complex problems. Composable building blocks, not a monolithic framework.</p>
</td>
</tr>
</table>
</div>
</div>
