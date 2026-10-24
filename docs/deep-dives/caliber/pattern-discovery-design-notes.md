---
icon: custom/calibration
---
# Pattern Discovery: Design Notes

Two capabilities the pipeline does not have yet, written up so the shape is agreed before anyone builds them.
Both touch the identity of a pattern over time or over size, which the current one-call, one-run design sidesteps.

## Batching Beyond One Clustering Call

### The problem

LLM clustering is a single call over at most `max_items` items.
Above that, evidence is sampled away, and the `recurrence_balanced_sampler` only guarantees that parents stay represented, not that the specific items which establish a pattern survive.
A single long call is also the known collapse mode for LLM batch output: the model returns assignments for a handful of items and omits the rest, which coverage accounting now detects but cannot repair.

### Proposed shape

Two stages, both deterministic given a seed:

1. **Chunk and cluster.** Split the corpus into chunks of at most `max_items`, balanced by recurrence key the same way the sampler balances, and cluster each chunk independently with the existing call. Each chunk yields patterns with full membership.
2. **Merge.** One call that receives, for every chunk pattern, its category, description, member count, distinct-key count, and a small round-robin sample of member items with ids. It returns groups of chunk-pattern ids that describe the same theme, plus a merged category and description per group. Membership of a merged pattern is the union of its chunk patterns' members.

### Constraints that must hold

- **Merge on evidence, not labels.** Labels and descriptions alone are too lossy to consolidate on; two chunks will name the same theme differently and name different themes the same. The merge input carries representative items with ids for every chunk pattern.
- **Membership is never re-derived by the model.** The merge call groups pattern ids; it does not emit item ids. Item membership flows from the chunk stage, where it was validated against submitted ids.
- **Accounting composes.** Every item is submitted to exactly one chunk, so chunk-level coverage buckets sum to corpus-level buckets. `omitted_by_model` stays per chunk and is reported per chunk and in total.
- **Chunk patterns that merge with nothing survive** as their own pattern. The recurrence pre-filter then applies to the merged membership, so a theme that appears once per chunk across several chunks is correctly counted as recurring.
- **Opt-in.** `batching='auto'` or similar; the default stays one call with sampling so existing callers see no change in cost or output.

### Open questions

- Whether the merge stage needs a second pass when the number of chunk patterns itself exceeds what one call handles well.
- Whether to run distillation on merged patterns only, or also keep per-chunk learnings as counterexamples to the merged one.

## Identity Across Runs

### The problem

Each run clusters from scratch.
The only bridge between runs is the deduper, which answers "have I seen a learning like this" and nothing else.
A consumer that ranks patterns over a rolling window needs a pattern to have a stable identity so its distinct-key count can grow as new parents arrive, without double-counting parents that were already counted.

### What stays outside the library

Persistence and merge policy are consumer-owned.
Counts over time must derive from the union of stable parent identities, kept idempotently by the consumer; the pipeline never increments a stored counter, because replaying an overlapping window would count the same parent twice.
Which parent identity is canonical (a ticket number, a conversation id with suffixes stripped) is also the consumer's rule.

### Option A: assign-then-cluster

The pipeline receives a registry of existing patterns (category, description, representative items with ids).
New evidence is first assigned to an existing pattern where it fits, and only the remainder is clustered fresh.
Learnings distilled from an existing pattern carry that pattern's id in provenance.

This is the complete solution and the expensive one: it adds a second model call shape, a registry input model, and a policy for when an existing pattern should split or retire.
Build it only after Option B has shown which parts consumers actually use.

### Option B: matching that says what it matched

The `Deduper` protocol today returns a boolean and mutates its own state inside `is_duplicate`, before any sink write succeeds.
A failed write therefore leaves the artifact remembered as seen.

The groundwork is a split protocol:

- `find_duplicate(artifact) -> Optional[str]` returns the id of the matched target without recording anything.
- `register(artifact, target_id)` records a new artifact after the pipeline has persisted it.

The pipeline calls `find_duplicate` where it calls `is_duplicate` today.
A match sets `LearningRecord.merged_into` to the target id and the record is reported, not dropped, so the consumer can attach the new parents to the existing pattern under its own idempotent rules.
`register` runs after a successful sink write.
Dedupers that implement only `is_duplicate` keep working unchanged; the pipeline feature-detects the new methods the way it already detects `reset_per_run`.

No automatic merging happens in the library.
What the pipeline exposes is the candidate, the matched target, and the evidence behind the candidate; what to do with that is the consumer's.

### Open questions

- Whether `merged_into` records should still be written to the sink (as a new version of the target) or only reported.
- Whether the embedding deduper's similarity threshold is the right matching criterion for identity, or whether identity should also require overlapping recurrence keys.
