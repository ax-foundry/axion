"""Tests for the Session class (cross-trace Langfuse sessions)."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List
from unittest.mock import MagicMock

from axion._core.tracing.collection.session import Session
from axion._core.tracing.collection.trace import Trace
from axion._core.tracing.collection.trace_collection import TraceCollection

# ---------------------------------------------------------------------------
# Fakes (mirrors test_trace_collection.py conventions; no live Langfuse)
# ---------------------------------------------------------------------------


@dataclass
class FakeObservation:
    name: str = 'step1'
    type: str = 'GENERATION'
    input: Any = ''
    output: Any = ''


@dataclass
class FakeRawTrace:
    id: str = 'trace-1'
    name: str = 'my-trace'
    input: Any = None
    output: Any = None
    observations: List[Any] = field(default_factory=list)
    timestamp: Any = None
    latency: float = 0.5
    tags: List[str] = field(default_factory=list)


def _chat_trace(i, ts=None, name='athena-chat', observations=None):
    """A conversational turn: plain-text trace-level input/output."""
    return FakeRawTrace(
        id=f'chat-{i}',
        name=name,
        input=f'question {i}',
        output=f'answer {i}',
        timestamp=ts,
        observations=observations or [],
    )


def _pipeline_trace(i, ts=None):
    """A non-conversational pipeline run: dict I/O with no query/output keys."""
    return FakeRawTrace(
        id=f'pipe-{i}',
        name='athena',
        input={'case_id': f'C{i}', 'execution_id': f'E{i}'},
        output={'outcome': 'done', 'workflow_status': 'complete'},
        timestamp=ts,
        observations=[FakeObservation(name='plan', type='SPAN')],
    )


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


class TestCoercion:
    def test_bare_list(self):
        s = Session([_chat_trace(0), _chat_trace(1)])
        assert len(s) == 2
        assert isinstance(s.traces, TraceCollection)
        assert isinstance(s[0], Trace)
        assert s.metadata == {}

    def test_dict_with_traces_and_full_meta(self):
        s = Session(
            {
                'id': 'sess-1',
                'environment': 'production',
                'project_id': 'proj-9',
                'custom_field': 'keep-me',
                'traces': [_chat_trace(0)],
            }
        )
        assert s.id == 'sess-1'
        assert s.environment == 'production'
        assert s.project_id == 'proj-9'
        # Extra (non-core) fields are preserved.
        assert s.metadata['custom_field'] == 'keep-me'
        assert len(s) == 1

    def test_sdk_like_object(self):
        sdk = MagicMock()
        sdk.model_dump.return_value = {
            'id': 'sess-2',
            'environment': 'staging',
            'extra': 'x',
            'traces': ['should-be-ignored'],
        }
        sdk.traces = [_chat_trace(0)]
        s = Session(sdk)
        assert s.id == 'sess-2'
        assert s.environment == 'staging'
        assert s.metadata['extra'] == 'x'
        # full_traces / .traces used for the actual traces, not model_dump's list
        assert len(s) == 1
        assert 'traces' not in s.metadata

    def test_full_traces_override_coerced(self):
        sdk_stub_traces = [_pipeline_trace(0)]  # would-be stubs
        full = [_chat_trace(0), _chat_trace(1)]
        s = Session({'id': 'x', 'traces': sdk_stub_traces}, full_traces=full)
        assert len(s) == 2
        assert all(t.name == 'athena-chat' for t in s)

    def test_camelcase_meta_aliases(self):
        s = Session(
            {
                'id': 'sess-1',
                'createdAt': '2026-01-01T10:00:00Z',
                'projectId': 'proj-cc',
                'traces': [_chat_trace(0)],
            }
        )
        assert s.created_at == '2026-01-01T10:00:00Z'
        assert s.project_id == 'proj-cc'


# ---------------------------------------------------------------------------
# Chronological sorting
# ---------------------------------------------------------------------------


class TestSorting:
    def test_sorts_by_timestamp(self):
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        # Provide newest-first (as Langfuse does); expect oldest-first after sort.
        s = Session(
            [
                _chat_trace(2, ts=t2),
                _chat_trace(0, ts=t0),
                _chat_trace(1, ts=t1),
            ]
        )
        assert [t.id for t in s] == ['chat-0', 'chat-1', 'chat-2']

    def test_mixed_naive_aware_iso_and_invalid_sort_without_error(self):
        traces = [
            _chat_trace(0, ts=datetime(2026, 1, 1, 9, 0)),  # naive
            _chat_trace(1, ts=datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)),
            _chat_trace(2, ts='2026-01-01T07:00:00Z'),  # ISO string
            _chat_trace(3, ts='not-a-date'),  # invalid -> sentinel (sorts first)
        ]
        s = Session(traces)
        # Should not raise; invalid sentinel sorts earliest, then 07:00, 08:00, 09:00
        assert [t.id for t in s] == ['chat-3', 'chat-2', 'chat-1', 'chat-0']

    def test_sort_opt_out(self):
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)
        s = Session([_chat_trace(1, ts=t1), _chat_trace(0, ts=t0)], sort=False)
        assert [t.id for t in s] == ['chat-1', 'chat-0']

    def test_camelcase_timestamp_key_sorts(self):
        # Dict traces exposing camelCase `createdAt` should still sort.
        traces = [
            {
                'id': 'b',
                'input': 'q1',
                'output': 'a1',
                'createdAt': '2026-01-01T11:00:00Z',
            },
            {
                'id': 'a',
                'input': 'q0',
                'output': 'a0',
                'createdAt': '2026-01-01T10:00:00Z',
            },
        ]
        s = Session(traces)
        assert [t.id for t in s] == ['a', 'b']


# ---------------------------------------------------------------------------
# Turn detection
# ---------------------------------------------------------------------------


class TestTurnDetection:
    def test_only_chat_traces_qualify(self):
        s = Session([_chat_trace(0), _pipeline_trace(0), _chat_trace(1)])
        assert s.turn_count == 2
        assert s.turn_trace_name == 'athena-chat'
        assert s.turn_trace_names == ['athena-chat']

    def test_name_override_reliable(self):
        s = Session([_chat_trace(0), _pipeline_trace(0)])
        conv = s.conversation(name='athena')
        # Pipeline trace has dict I/O; query/output fall back to json.dumps,
        # but a turn is still built because the user side is non-empty.
        assert conv is not None
        assert len(conv.messages) == 2  # one human + one ai

    def test_is_turn_override_wins(self):
        s = Session([_chat_trace(0), _chat_trace(1), _pipeline_trace(0)])
        conv = s.conversation(is_turn=lambda t: t.id == 'chat-0')
        assert conv is not None
        # only chat-0 => 1 human + 1 ai
        assert len(conv.messages) == 2

    def test_adversarial_dict_with_input_output_keys(self):
        adversarial = FakeRawTrace(
            id='adv-0',
            name='workflow',
            input={'input': 'sneaky user text'},
            output={'output': 'sneaky response'},
        )
        s = Session([adversarial])
        # Best-effort auto-detection admits it (documented limitation).
        assert s.turn_count == 1
        # is_turn= remains the reliable escape hatch.
        assert s.conversation(is_turn=lambda t: False) is None

    def test_dominant_name_tie_break_deterministic(self):
        # Two qualifying chat names, equal counts -> first-seen wins.
        traces = [
            _chat_trace(0, name='chat-a'),
            _chat_trace(1, name='chat-b'),
        ]
        s = Session(traces, sort=False)
        assert s.turn_trace_name == 'chat-a'
        assert set(s.turn_trace_names) == {'chat-a', 'chat-b'}

    def test_dominant_name_highest_count(self):
        traces = [
            _chat_trace(0, name='chat-b'),
            _chat_trace(1, name='chat-a'),
            _chat_trace(2, name='chat-a'),
        ]
        s = Session(traces, sort=False)
        assert s.turn_trace_name == 'chat-a'

    def test_default_turn_name_applies_to_conversation(self):
        # Two chat names present; pin one as the session default.
        traces = [
            _chat_trace(0, name='athena-chat'),
            _chat_trace(1, name='athena-chat'),
            _chat_trace(2, name='other-chat'),
        ]
        s = Session(traces, sort=False, turn_name='athena-chat')
        conv = s.conversation()  # no per-call name
        assert conv is not None
        assert len(conv.messages) == 4  # only the 2 athena-chat turns
        assert s.turn_count == 2

    def test_per_call_name_overrides_default(self):
        traces = [
            _chat_trace(0, name='athena-chat'),
            _chat_trace(1, name='other-chat'),
        ]
        s = Session(traces, sort=False, turn_name='athena-chat')
        conv = s.conversation(name='other-chat')  # override
        assert len(conv.messages) == 2

    def test_default_turn_predicate(self):
        traces = [_chat_trace(0), _chat_trace(1), _chat_trace(2)]
        s = Session(traces, sort=False, turn_predicate=lambda t: t.id == 'chat-1')
        conv = s.conversation()
        assert len(conv.messages) == 2
        assert s.turn_count == 1


# ---------------------------------------------------------------------------
# Conversation building
# ---------------------------------------------------------------------------


class TestConversation:
    def test_basic_conversation(self):
        s = Session([_chat_trace(0), _chat_trace(1)])
        conv = s.conversation()
        assert conv is not None
        assert len(conv.messages) == 4  # 2 turns x (human + ai)
        assert conv.messages[0].role == 'user'
        assert conv.messages[0].content == 'question 0'
        assert conv.messages[1].role == 'assistant'
        assert conv.messages[1].content == 'answer 0'

    def test_empty_session_returns_none(self):
        s = Session([])
        assert s.conversation() is None
        assert s.turn_count == 0

    def test_no_qualifying_turns_returns_none(self):
        s = Session([_pipeline_trace(0), _pipeline_trace(1)])
        assert s.conversation() is None

    def test_metadata_attached_to_conversation(self):
        s = Session(
            {
                'id': 'sess-1',
                'environment': 'production',
                'custom_field': 'keep-me',
                'traces': [_chat_trace(0)],
            }
        )
        conv = s.conversation()
        assert conv is not None
        assert conv.metadata['id'] == 'sess-1'
        assert conv.metadata['custom_field'] == 'keep-me'


# ---------------------------------------------------------------------------
# Tool reconstruction (include_tools)
# ---------------------------------------------------------------------------


class TestIncludeTools:
    def test_builds_tool_calls_and_messages(self):
        tool_obs = FakeObservation(
            name='search_db',
            type='TOOL',
            input={'q': 'rate'},
            output='found 3 rows',
        )
        trace = _chat_trace(0, observations=[tool_obs])
        s = Session([trace])
        conv = s.conversation(include_tools=True)
        assert conv is not None
        roles = [m.role for m in conv.messages]
        assert roles == ['user', 'assistant', 'tool']
        ai = conv.messages[1]
        assert ai.tool_calls is not None
        assert len(ai.tool_calls) == 1
        call = ai.tool_calls[0]
        assert call.name == 'search_db'
        assert call.args == {'q': 'rate'}
        # Shared tool_call_id between the call and its paired message.
        assert conv.messages[2].tool_call_id == call.id
        assert conv.messages[2].content == 'found 3 rows'

    def test_skips_tool_obs_missing_name(self):
        good = FakeObservation(name='search_db', type='TOOL', output='ok')
        bad = FakeObservation(name='', type='TOOL', output='ignored')
        trace = _chat_trace(0, observations=[bad, good])
        s = Session([trace])
        conv = s.conversation(include_tools=True)
        ai = conv.messages[1]
        assert len(ai.tool_calls) == 1
        assert ai.tool_calls[0].name == 'search_db'

    def test_tools_off_by_default(self):
        tool_obs = FakeObservation(name='search_db', type='TOOL', output='ok')
        s = Session([_chat_trace(0, observations=[tool_obs])])
        conv = s.conversation()
        assert all(m.role != 'tool' for m in conv.messages)
        assert conv.messages[1].tool_calls is None

    def test_json_string_tool_args_parsed(self):
        # Langfuse may store tool inputs as a JSON string, not a dict.
        tool_obs = FakeObservation(
            name='search_db', type='TOOL', input='{"q": "rate"}', output='ok'
        )
        s = Session([_chat_trace(0, observations=[tool_obs])])
        conv = s.conversation(include_tools=True)
        call = conv.messages[1].tool_calls[0]
        assert call.args == {'q': 'rate'}

    def test_non_dict_tool_args_default_empty(self):
        tool_obs = FakeObservation(
            name='search_db', type='TOOL', input='not-json', output='ok'
        )
        s = Session([_chat_trace(0, observations=[tool_obs])])
        conv = s.conversation(include_tools=True)
        assert conv.messages[1].tool_calls[0].args == {}

    def test_dict_tool_output_serialized(self):
        tool_obs = FakeObservation(
            name='lookup', type='TOOL', output={'rows': [1, 2, 3]}
        )
        s = Session([_chat_trace(0, observations=[tool_obs])])
        conv = s.conversation(include_tools=True)
        assert '"rows"' in conv.messages[2].content


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_by_type_across_traces(self):
        t0 = _chat_trace(0, observations=[FakeObservation(name='a', type='TOOL')])
        t1 = _chat_trace(
            1,
            observations=[
                FakeObservation(name='b', type='TOOL'),
                FakeObservation(name='c', type='GENERATION'),
            ],
        )
        s = Session([t0, t1])
        assert len(s.by_type('TOOL')) == 2
        assert len(s.tools()) == 2
        assert len(s.by_type('GENERATION')) == 1

    def test_observation_types_union(self):
        t0 = _chat_trace(0, observations=[FakeObservation(type='GENERATION')])
        t1 = _chat_trace(1, observations=[FakeObservation(type='TOOL')])
        s = Session([t0, t1])
        assert s.observation_types == ['GENERATION', 'TOOL']

    def test_stub_traces_no_observations(self):
        # Stub traces (trace-level I/O but no observations) still yield a
        # conversation; aggregation simply returns [].
        s = Session([_chat_trace(0), _chat_trace(1)])
        assert s.conversation() is not None
        assert s.by_type('TOOL') == []
        assert s.tools() == []
        assert s.observation_types == []

    def test_find_all_across_traces(self):
        t0 = _chat_trace(0, observations=[FakeObservation(name='dup', type='TOOL')])
        t1 = _chat_trace(1, observations=[FakeObservation(name='dup', type='TOOL')])
        s = Session([t0, t1])
        assert len(s.find_all(name='dup')) == 2


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------


class TestToDataset:
    def test_default_one_multi_turn_item(self):
        s = Session([_chat_trace(0), _chat_trace(1)])
        ds = s.to_dataset()
        assert len(ds.items) == 1
        item = ds.items[0]
        assert item.multi_turn_conversation is not None
        assert len(item.multi_turn_conversation.messages) == 4

    def test_empty_session_no_items(self):
        s = Session([_pipeline_trace(0)])
        ds = s.to_dataset()
        assert len(ds.items) == 0

    def test_transform_returning_list_flattens(self):
        s = Session([_chat_trace(0)])

        def transform(session):
            return [
                {'query': 'q1', 'actual_output': 'a1'},
                {'query': 'q2', 'actual_output': 'a2'},
            ]

        ds = s.to_dataset(transform=transform)
        assert len(ds.items) == 2

    def test_transform_returning_none_skipped(self):
        s = Session([_chat_trace(0)])
        ds = s.to_dataset(transform=lambda session: None)
        assert len(ds.items) == 0


# ---------------------------------------------------------------------------
# Serialization / round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_includes_meta_and_traces(self):
        s = Session(
            {
                'id': 'sess-1',
                'environment': 'production',
                'custom_field': 'keep-me',
                'traces': [_chat_trace(0)],
            }
        )
        d = s.to_dict()
        assert d['id'] == 'sess-1'
        assert d['custom_field'] == 'keep-me'
        assert isinstance(d['traces'], list)
        assert len(d['traces']) == 1

    def test_round_trip_via_to_dict(self):
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)
        s = Session(
            {
                'id': 'sess-1',
                'environment': 'production',
                'traces': [_chat_trace(1, ts=t1), _chat_trace(0, ts=t0)],
            }
        )
        d = s.to_dict()
        # created_at becomes a string after JSON normalization in traces;
        # rebuilding must not raise and must still sort chronologically.
        s2 = Session(d)
        assert s2.id == 'sess-1'
        assert s2.environment == 'production'
        assert [t.id for t in s2] == ['chat-0', 'chat-1']

    def test_save_and_load_json(self, tmp_path):
        s = Session({'id': 'sess-1', 'traces': [_chat_trace(0), _chat_trace(1)]})
        path = tmp_path / 'session.json'
        # Session has no save_json itself; round-trip the dict form.
        import json

        path.write_text(json.dumps(s.to_dict(), default=str))
        loaded = Session(json.loads(path.read_text()))
        assert loaded.id == 'sess-1'
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# from_langfuse (mocked loader)
# ---------------------------------------------------------------------------


class TestFromLangfuse:
    def test_from_langfuse_uses_loader(self):
        loader = MagicMock()
        loader.get_session_with_traces.return_value = (
            {'id': 'sess-99', 'environment': 'production'},
            [_chat_trace(0), _chat_trace(1)],
        )
        s = Session.from_langfuse('sess-99', loader=loader)
        loader.get_session_with_traces.assert_called_once()
        assert s.id == 'sess-99'
        assert len(s) == 2

    def test_from_langfuse_missing_session(self):
        loader = MagicMock()
        loader.get_session_with_traces.return_value = (None, [])
        s = Session.from_langfuse('nope', loader=loader)
        assert s.id == 'nope'
        assert len(s) == 0

    def test_enrich_false_passed_to_loader(self):
        loader = MagicMock()
        loader.get_session_with_traces.return_value = (
            {'id': 'sess-1'},
            [_chat_trace(0)],  # stub-only traces
        )
        s = Session.from_langfuse('sess-1', loader=loader, enrich=False)
        _, kwargs = loader.get_session_with_traces.call_args
        assert kwargs['enrich'] is False
        # Conversation still reconstructs from stub trace-level I/O.
        conv = s.conversation()
        assert conv is not None
        assert len(conv.messages) == 2
        # Observation-level aggregation is empty on stubs.
        assert s.by_type('TOOL') == []

    def test_enrich_defaults_true(self):
        loader = MagicMock()
        loader.get_session_with_traces.return_value = ({'id': 's'}, [])
        Session.from_langfuse('s', loader=loader)
        _, kwargs = loader.get_session_with_traces.call_args
        assert kwargs['enrich'] is True

    def test_stub_retention_preserves_turn(self):
        # Simulate the loader returning a stub (trace-level I/O, no observations)
        # for a trace whose full fetch failed -- the conversation turn survives.
        loader = MagicMock()
        stub = _chat_trace(0)  # no observations
        loader.get_session_with_traces.return_value = (
            {'id': 'sess-1'},
            [stub],
        )
        s = Session.from_langfuse('sess-1', loader=loader)
        conv = s.conversation()
        assert conv is not None
        assert len(conv.messages) == 2
        assert s.by_type('TOOL') == []


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr():
    s = Session({'id': 'sess-1', 'traces': [_chat_trace(0)]})
    r = repr(s)
    assert 'sess-1' in r
    assert 'traces=1' in r
