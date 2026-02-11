# Why Ground Truth Matters

<div class="seatbelt-hero" markdown="0">
<div class="seatbelt-hero__text">
<span class="seatbelt-hero__label">Principle</span>
<p class="seatbelt-hero__quote">Without ground truth,<br><em>you're grading on vibes.</em></p>
<p class="seatbelt-hero__sub">Unlike traditional ML &mdash; where labeled data fuels both training and validation &mdash; AI agents often operate without predefined answers. Curating a high-quality "golden" test set isn't just important, it's essential. Ground truth turns subjective performance into an objective, repeatable benchmark.</p>
</div>
<div class="seatbelt-hero__text" style="background: linear-gradient(160deg, #162030 0%, #1a2840 100%);">
<span class="seatbelt-hero__label">Why It Matters</span>
<p class="seatbelt-hero__quote">Approximation<br><em>is not enough.</em></p>
<p class="seatbelt-hero__sub">Language models can approximate quality, but in high-stakes and domain-specific environments, approximation is not enough. Ground truth anchors evaluation to a consistent standard: the expected answer (and, when applicable, the expected evidence).</p>
</div>
</div>

---

## What Goes Wrong Without Ground Truth

Even strong evaluation frameworks can fail if they are not anchored to expected outcomes. Three common failure modes:

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Plausible but wrong</p>
<p class="rule-card__desc">The answer sounds correct, but is outdated, incomplete, or subtly incorrect for the domain.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">The gullible judge</p>
<p class="rule-card__desc">LLM judges over-reward "safe" answers or fluent answers, scoring linguistic plausibility rather than correctness.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">The helpful liar</p>
<p class="rule-card__desc">Retrieval returns noisy context. The model synthesizes a convincing answer from noise. Similarity-based evals still pass it.</p>
</div>
</div>

---

## The Core Reasons Ground Truth Matters

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Factual Accuracy</p>
<p class="rule-card__desc">Reveals whether an answer is actually correct &mdash; not merely plausible or well-written.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Relevance & Completeness</p>
<p class="rule-card__desc">Clarifies what must be covered and what is irrelevant, preventing "good-sounding" partial responses from passing.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Retrieval Correctness</p>
<p class="rule-card__desc">Enables objective checks that the system found and cited the right documents &mdash; separating retrieval failure from generation failure.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Determinism</p>
<p class="rule-card__desc">LLM judging is inherently variable. Ground truth reduces subjectivity and makes scoring repeatable across runs, models, and prompt iterations.</p>
</div>
</div>

<div class="rule-grid" style="grid-template-columns: 1fr; max-width: 50%;" markdown="0">
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Benchmarking & Iteration</p>
<p class="rule-card__desc">A fixed test set lets you run fair A/B tests, track regressions, and measure improvements with confidence.</p>
</div>
</div>

---

## Ground Truth Dataset Lifecycle

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Formation</p>
<p class="rule-card__desc">Curate high-value, real-world utterances. Validate expected outcomes with domain experts.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">Maintenance</p>
<p class="rule-card__desc">Establish review cycles and governance to keep answers current as products, policies, and knowledge evolve.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">Expansion</p>
<p class="rule-card__desc">Grow coverage intentionally &mdash; edge cases, failure clusters, controlled synthetic generation &mdash; without compromising quality.</p>
</div>
</div>

---

## A Practical Evaluation Approach

For reliable evaluation, structure matters:

- **Decompose the workflow** into discrete steps (e.g., routing, retrieval, tool-use, generation).
- **Build test cases per step**, not just end-to-end: you want to isolate where failures occur.
- **Use hierarchical gates**:

```mermaid
graph TD
    R["Retrieval Correctness"] -->|"Did we find the right info?"| G["Generation Quality"]
    G -->|"Did we use it correctly?"| O["Overall Correctness"]
    O -->|"Only meaningful if prior gates pass"| V["Validated Result"]
```

---

## Inconsistent Scoring

!!! warning "Why LLM-Only Evaluation Fails"
    LLMs are non-deterministic, and LLM judges can disagree—even on the same answer. Without ground truth, evaluations drift toward subjective heuristics like fluency, verbosity, or "sounds right."

<div style="background: linear-gradient(135deg, #2a3320 0%, #1e2618 100%); padding: 24px; border-radius: 8px; color: white; margin: 20px 0;" markdown="0">
<p style="margin: 0; font-size: 14px; line-height: 1.7;">Ground truth is the only way to anchor evaluation to objective, expected outcomes. Without it, you're measuring style. With it, you're measuring substance.</p>
</div>

[Evaluation Flywheel :octicons-arrow-right-24:](evaluation_flywheel.md){ .md-button .md-button--primary }
[Agent Evaluation Playbook :octicons-arrow-right-24:](agent_playbook.md){ .md-button }
