---
icon: custom/analytics
---
# Session Collection

`SessionCollection` is the session-level counterpart to [`TraceCollection`](trace-collection.md). Where a `TraceCollection` works with individual traces, a `SessionCollection` works with Langfuse **sessions** — groups of traces that together form a multi-turn conversation (e.g. an assistant answering a user across many turns, interleaved with pipeline runs).

- **Session-level metadata**: Preserve the full session record (id, environment, and any extra fields), not just per-trace data
- **Conversation reconstruction**: Build a `MultiTurnConversation` from the session's traces, with reliable turn selection
- **Composed traces**: Each session exposes its traces as a `TraceCollection`, so all trace tooling still applies
- **Cross-trace aggregation**: Pull tools/observations across every trace in a session (or across every session)
- **Dataset conversion**: One multi-turn `DatasetItem` per session
- **Fast or full**: Skip per-trace enrichment for conversation-only workflows, or fetch full observations when you need them

---

## Mental Model

```
SessionCollection
└── Session                      # session-level metadata + conversation
    ├── metadata (id, environment, project_id, ...)
    ├── conversation()           # MultiTurnConversation across turns
    └── traces  -> TraceCollection
        ├── Trace                # one conversational turn / pipeline run
        ├── Trace
        └── ...
```

A `Session` *composes* a `TraceCollection`, so `session.traces`, `session[0]`, and iteration all return the same `Trace` objects documented in the [Trace Collection guide](trace-collection.md).

---

## Quick Start

```python
from axion.tracing import SessionCollection, LangfuseTraceLoader

loader = LangfuseTraceLoader()

sc = SessionCollection.from_langfuse(
    session_ids=['01KSQTKNK4YEWEFYA1ATEQ9QN1'],
    loader=loader,
)

session = sc[0]
print(session.id, session.environment)   # session-level metadata
print(session.turn_count)                 # number of conversational turns

conv = session.conversation()             # MultiTurnConversation
for msg in conv.messages:
    print(msg.role, '->', msg.content)

dataset = sc.to_dataset()                 # one multi-turn DatasetItem per session
```

---

## Loading Sessions

### From Langfuse

`from_langfuse()` fetches each session (and its traces) and wraps them:

```python
from axion.tracing import SessionCollection, LangfuseTraceLoader

loader = LangfuseTraceLoader(
    public_key='pk-lf-...',
    secret_key='sk-lf-...',
)

sc = SessionCollection.from_langfuse(
    session_ids=['session-a', 'session-b'],
    loader=loader,
)
```

!!! note "Explicit session IDs"
    v1 fetches by explicit `session_ids` only. Not-found ids are **skipped** (logged at WARNING), so the collection contains only sessions that exist.

For a single session, use `Session.from_langfuse()`:

```python
from axion.tracing import Session

session = Session.from_langfuse('session-a', loader=loader)
```

!!! info "Single vs. collection — missing-session behavior"
    `Session.from_langfuse()` returns an empty but *identifiable* `Session` (its `id` is set, with no traces) for a missing id, since single-id callers usually want a usable handle back. `SessionCollection.from_langfuse()` **skips** missing ids.

### Fast Mode: `enrich=False`

By default, `from_langfuse()` fetches **full** trace details (with observations) — one API call per trace. Session trace stubs already carry trace-level `input`/`output`, so for conversation-only workflows you can skip enrichment entirely with `enrich=False` (a single API call per session):

```python
sc = SessionCollection.from_langfuse(
    session_ids=['session-a', 'session-b'],
    loader=loader,
    enrich=False,           # skip per-trace fetch
)

sc[0].conversation()        # still works — built from stub trace-level I/O
```

| | `enrich=True` (default) | `enrich=False` |
|---|---|---|
| API calls per session | 1 + N (one per trace) | 1 |
| `conversation()` / `to_dataset()` | ✅ | ✅ |
| `by_type()` / `tools()` / `find_all()` | ✅ | returns `[]` (no observations on stubs) |

!!! warning "`enrich=False` + `include_tools=True`"
    Tool reconstruction needs `TOOL` observations, which stubs don't carry. Combining the two yields a conversation with no tool messages. The conversation **text** is unaffected.

### From JSON

```python
sc = SessionCollection.load_json('sessions.json')
```

---

## Reconstructing the Conversation

`session.conversation()` returns a `MultiTurnConversation` built from the traces that qualify as conversational **turns** — each turn contributes a `HumanMessage` (from trace input) and an `AIMessage` (from trace output), in chronological order.

```python
conv = session.conversation()
if conv is not None:
    print(len(conv.messages))
    print(conv.metadata)        # full session metadata is attached
```

!!! note "Returns `None` when empty"
    If no traces qualify as turns, `conversation()` returns `None` (an empty conversation is never constructed).

### Turn Selection — Reliable vs. Best-Effort

Sessions often mix conversational traces (text I/O) with pipeline/workflow traces (dict I/O). How a turn is identified matters, and there are reliable and best-effort paths:

=== "By name (reliable)"

    ```python
    # Only traces with this exact name become turns.
    conv = session.conversation(name='athena-chat')
    ```

=== "By predicate (reliable)"

    ```python
    # Arbitrary logic; wins over name=.
    conv = session.conversation(
        is_turn=lambda trace: trace.name in {'athena-chat', 'athena-chat-v2'}
    )
    ```

=== "Auto-detect (best-effort)"

    ```python
    # No selector: auto-detection picks the dominant text-I/O trace name.
    conv = session.conversation()
    ```

!!! warning "Auto-detection is best-effort"
    Auto-detection requires both input and output to extract to real (key-matched) text, then keeps the **dominant** qualifying trace name. It excludes typical pipeline traces (dict blobs with no message keys), but a workflow trace whose payload happens to contain `input`/`output`/`message`/`result` keys can be misclassified. For guaranteed selection use `name=` or `is_turn=`.

Inspect what auto-detection chose:

```python
session.turn_trace_name     # dominant qualifying name (deterministic tie-break)
session.turn_trace_names    # all distinct qualifying names
session.turn_count          # number of traces counted as turns
```

### Setting a Default Turn Selector

Rather than passing `name=` on every call, pin a default selector once at fetch time. It applies to `conversation()`, `to_dataset()`, and `turn_count`, and is still overridable per call:

```python
sc = SessionCollection.from_langfuse(
    session_ids=['session-a'],
    loader=loader,
    turn_name='athena-chat',          # default for every session
)

sc[0].conversation()                  # uses 'athena-chat' automatically
sc[0].conversation(name='other')      # per-call override still wins
```

`turn_predicate=` is also accepted (and wins over `turn_name=`):

```python
sc = SessionCollection.from_langfuse(
    session_ids=['session-a'],
    loader=loader,
    turn_predicate=lambda t: t.name == 'athena-chat',
)
```

Selector priority, highest first:

1. Per-call `conversation(is_turn=...)`
2. Per-call `conversation(name=...)`
3. Session-level `turn_predicate=`
4. Session-level `turn_name=`
5. Best-effort auto-detection

### Including Tool Calls

Tool reconstruction is **off by default**. Opt in with `include_tools=True` to attach `tool_calls` to each `AIMessage` and a paired `ToolMessage` per tool, reconstructed from each turn's `TOOL` observations:

```python
conv = session.conversation(name='athena-chat', include_tools=True)
```

!!! note "Best-effort, skip-malformed"
    A `ToolCall` is built only when the observation has a name (args default to `{}`; JSON-string args are decoded). The synthesized `tool_call_id` is shared with its paired `ToolMessage`. Observations missing a name are silently skipped — a tool call is never fabricated.

---

## Session Metadata

`Session` preserves the **full** session record, not just the core fields (the Langfuse SDK session model is `extra='allow'`):

```python
session.id              # session id
session.created_at      # tolerates camelCase 'createdAt'
session.project_id      # tolerates camelCase 'projectId'
session.environment
session.metadata         # full dict of all captured fields
```

The complete metadata dict is also attached to the reconstructed conversation as `conversation.metadata`.

---

## Aggregating Across Traces

A session aggregates observation-level data across every one of its traces (requires `enrich=True`):

```python
# All tool observations across the whole session
tools = session.tools()              # == session.by_type('TOOL')

# Any observation type
generations = session.by_type('GENERATION')

# Union of observation types present anywhere in the session
session.observation_types            # e.g. ['SPAN', 'GENERATION', 'TOOL']

# Every node matching a name and/or type, across all traces
nodes = session.find_all(name='search_db', type='TOOL')
```

A `SessionCollection` aggregates one level higher — across every trace in every session:

```python
sc.by_type('TOOL')    # all TOOL observations in the whole collection
sc.tools()
```

---

## Filtering

```python
# Lambda filter
prod = sc.filter(lambda s: s.environment == 'production')
multi_turn = sc.filter(lambda s: s.turn_count > 1)

# Attribute equality
staging = sc.filter_by(environment='staging')
```

Both return a new `SessionCollection` (preserving any default turn selector).

---

## Converting to Dataset

`to_dataset()` produces one multi-turn `DatasetItem` per session (sessions with no turns are skipped):

```python
dataset = sc.to_dataset(name='session-eval')
# each item.multi_turn_conversation is a MultiTurnConversation
```

For a single session:

```python
dataset = session.to_dataset()
```

### Custom Transform

Pass a `transform(session)` returning a `DatasetItem`, a `dict`, **or a list** of those (lists are flattened — useful for emitting one item per turn):

```python
from axion.dataset import DatasetItem

def per_turn(session):
    conv = session.conversation(name='athena-chat')
    if conv is None:
        return None
    return DatasetItem(
        id=session.id,
        multi_turn_conversation=conv,
    )

dataset = sc.to_dataset(name='session-eval', transform=per_turn)
```

---

## Serialization

```python
# Save the whole collection (list of session dicts)
sc.save_json('sessions/snapshot.json')

# Load later
loaded = SessionCollection.load_json('sessions/snapshot.json')

# Raw access: each session as a JSON-able dict
dicts = sc.to_list()

# A single session round-trips through to_dict()
payload = session.to_dict()          # full metadata + JSON-able traces
restored = Session(payload)
```

`to_dict()` is round-trip safe: timestamps normalize to UTC-aware on reload, so a reloaded session still sorts chronologically.

---

## End-to-End Example

```python
from axion.tracing import SessionCollection, LangfuseTraceLoader
from axion.metrics import AnswerRelevancy
from axion.runners import evaluation_runner

# 1. Fetch sessions, pinning the conversational trace name
loader = LangfuseTraceLoader()
sc = SessionCollection.from_langfuse(
    session_ids=['01KSQTKNK4YEWEFYA1ATEQ9QN1'],
    loader=loader,
    turn_name='athena-chat',
)

# 2. Explore
for session in sc:
    print(session.id, session.environment, 'turns:', session.turn_count)
    print('  tools used:', [t.name for t in session.tools()])

# 3. Convert to a multi-turn dataset (one item per session)
dataset = sc.to_dataset(name='athena-conversations')

# 4. Evaluate
result = await evaluation_runner(
    evaluation_inputs=dataset,
    scoring_metrics=[AnswerRelevancy()],
    evaluation_name='Athena Multi-Turn Evaluation',
)

# 5. Publish scores back to Langfuse
result.publish_to_observability()
```

---

## API Reference

### SessionCollection

| Method | Description |
|--------|-------------|
| `from_langfuse(session_ids, loader, prompt_patterns, show_progress, enrich, turn_name, turn_predicate)` | Fetch sessions from Langfuse and wrap |
| `load_json(path, prompt_patterns, turn_name, turn_predicate)` | Load from a JSON file |
| `filter(condition)` | Filter by lambda, returns new `SessionCollection` |
| `filter_by(**kwargs)` | Filter by attribute equality |
| `by_type(type_str)` | All observations of a type across every session |
| `tools()` | All `TOOL` observations across every session |
| `to_dataset(name, transform)` | Convert to axion `Dataset` (one item per session) |
| `to_list()` | Each session as a JSON-able dict |
| `save_json(path)` | Serialize to JSON file |
| `len(sc)` / `sc[i]` / iteration | Sequence protocol over `Session`s |

### Session

| Property / Method | Description |
|-------------------|-------------|
| `session.id` | Session id |
| `session.created_at` | Session creation time (tolerates `createdAt`) |
| `session.project_id` | Project id (tolerates `projectId`) |
| `session.environment` | Session environment |
| `session.metadata` | Full session metadata dict (all captured fields) |
| `session.traces` | The session's traces as a `TraceCollection` |
| `session[i]` / iteration | Access traces (chronologically sorted) |
| `session.conversation(name, is_turn, include_tools)` | Reconstruct `MultiTurnConversation`, or `None` |
| `session.turn_count` | Number of traces counted as turns under the default selector |
| `session.turn_trace_name` | Dominant auto-detected turn name (deterministic) |
| `session.turn_trace_names` | All distinct auto-detected turn names |
| `session.by_type(type_str)` | All observations of a type across all traces |
| `session.observation_types` | First-seen-ordered union of observation types |
| `session.tools()` | All `TOOL` observations across all traces |
| `session.find_all(name, type)` | All matching nodes across all traces |
| `session.to_dataset(name, transform)` | Convert to a `Dataset` (one multi-turn item) |
| `session.to_dict()` | Full metadata + JSON-able traces (round-trippable) |
| `Session.from_langfuse(session_id, loader, prompt_patterns, show_progress, enrich, turn_name, turn_predicate)` | Fetch a single session and wrap |

---

## Next Steps

- **[Trace Collection](trace-collection.md)**: Per-trace exploration, steps, observation tree, prompt variable extraction
- **[Tracing](tracing.md)**: Creating traces with `@trace` and session ids
- **[Publishing](publishing.md)**: Publish evaluation scores back to Langfuse
