---
icon: custom/evaluate
---
# The Evaluation Flywheel

<div class="seatbelt-hero" markdown="0">
<div class="seatbelt-hero__text">
<span class="seatbelt-hero__label">Lifecycle</span>
<p class="seatbelt-hero__quote">Build. Test. Deploy.<br><em>Learn. Repeat.</em></p>
<p class="seatbelt-hero__sub">The Evaluation Flywheel is a continuous process for building, testing, deploying, and improving AI models. It's called a "flywheel" because feedback from production accelerates improvements in development, building momentum over time.</p>
</div>
<div class="seatbelt-hero__visual">
<img src="../assets/flywheel.png" alt="Evaluation Flywheel" loading="lazy">
</div>
</div>

The lifecycle consists of two interconnected loops:

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Pre-Production (The Lab)</p>
<p class="rule-card__desc">Validate before release. Test challengers against baselines using golden datasets and ground truth.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">Post-Production (The Real World)</p>
<p class="rule-card__desc">Confirm value in practice. Monitor drift, measure business impact, and collect user feedback.</p>
</div>
</div>

---

## Loop 1: Pre-Production

A controlled environment where you test models without affecting real users.

**Goal:** Validate that the new model is better than the current one—and hasn't broken anything that used to work.

### Process

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Design & Update</p>
<p class="rule-card__desc">Create new model versions to address needs or fix problems.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">Run Experiments</p>
<p class="rule-card__desc">Test the "Challenger" model against the "Baseline" using golden datasets.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">Measure</p>
<p class="rule-card__desc">Quantify results against ground truth with targeted metrics.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">4</span>
<p class="rule-card__title">Analyze</p>
<p class="rule-card__desc">Check for improvements, regressions, and safety issues.</p>
</div>
</div>

### What You Need

- **Golden Datasets** — Curated examples with known correct answers
- **Ground Truth** — The definitive "right answer" for each test case

### Key Metrics

| Metric | What It Measures |
|--------|------------------|
| Accuracy | How often the model is correct |
| Relevance | How well answers match the question |
| Groundedness | Whether answers are based on facts |
| Safety | Whether outputs avoid harmful content |

### Exit Criteria

A model leaves this loop only when it:

- Passes all safety checks
- Shows accuracy improvements
- Has zero regressions on existing capabilities

---

## The Release Gate

<div class="callout-dark" markdown="0">
<span class="callout-dark__label">Deployment Decision</span>
<p>Between the two loops sits a mandatory checkpoint. Models cannot move to production unless they meet all Loop 1 criteria. Failed models return to the design phase. Passing models get promoted.</p>
</div>

---

## Loop 2: Post-Production

The live environment where real users interact with your model.

**Goal:** Confirm that Lab results translate to real-world value, and maintain parity between offline and live performance.

### Process

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Deploy & Adapt</p>
<p class="rule-card__desc">Release the model and handle real traffic at scale.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">Monitor</p>
<p class="rule-card__desc">Watch for drift when real-world data diverges from training data.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">Evaluate Value</p>
<p class="rule-card__desc">Measure business impact and user outcomes against expectations.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">4</span>
<p class="rule-card__title">Integrate Feedback</p>
<p class="rule-card__desc">Collect user signals and analyze usage patterns for the next cycle.</p>
</div>
</div>

### What You Need

- **Session Traces** — Real interaction data from live users
- **Feedback Channels** — User ratings, support tickets, behavioral signals

### Key Metrics

| Metric | What It Measures |
|--------|------------------|
| Business KPIs | Revenue, conversion, retention impact |
| Usage | Adoption, engagement, feature utilization |
| Efficacy | Whether users actually solve their problems |

### Critical Check: Prod-Test Parity

Ask: *"Are live scores matching Lab scores?"*

If Lab accuracy was 95% but production accuracy is 70%, something is wrong. This gap signals a problem with your testing methodology or data distribution.

### Success Criteria

- Positive ROI
- Production metrics match offline predictions

---

## The Bridge: Closing the Loop

The arrow at the bottom of the diagram—**The Bridge**—is what makes this a flywheel. It feeds real-world data back into the Lab.

```mermaid
graph LR
    A["Design & Update"] --> B["Run Experiments"]
    B --> C["Measure & Analyze"]
    C --> D{"Release Gate"}
    D -->|Pass| E["Deploy & Monitor"]
    D -->|Fail| A
    E --> F["Evaluate & Feedback"]
    F -->|"Bridge"| A
```

- **Sampled logs** become training and tuning data
- **Production failures** become new test cases

!!! tip "Failures Are Assets"
    Every production failure gets added to your golden datasets. This ensures the next model version is specifically tested against that scenario—preventing the same mistake twice.

<div class="callout-dark" markdown="0">
<p>This continuous feedback loop drives constant improvement. Each cycle through the flywheel makes your evaluation more comprehensive and your models more robust.</p>
</div>

[Agent Evaluation Playbook :octicons-arrow-right-24:](agent_playbook.md){ .md-button .md-button--primary }
[Why Ground Truth Matters :octicons-arrow-right-24:](why_ground_truth_matters.md){ .md-button }
