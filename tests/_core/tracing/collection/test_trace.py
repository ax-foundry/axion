"""Tests for Trace and TraceStep."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from axion._core.tracing.collection.models import ObservationsView
from axion._core.tracing.collection.prompt_patterns import (
    PromptPatternsBase,
    create_extraction_pattern,
)
from axion._core.tracing.collection.trace import Trace, TraceStep

# ---------------------------------------------------------------------------
# Fake trace data helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeObservation:
    id: str = 'obs-1'
    name: str = 'step1'
    type: str = 'GENERATION'
    parent_observation_id: Optional[str] = None
    start_time: Any = None
    input: Any = ''
    output: Any = ''
    metadata: Optional[Dict] = None


@dataclass
class FakeRawTrace:
    id: str = 'trace-1'
    name: str = 'my-trace'
    input: Any = None
    output: Any = None
    observations: List[Any] = field(default_factory=list)
    latency: float = 0.5
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TraceStep
# ---------------------------------------------------------------------------


class TestTraceStep:
    def test_count(self):
        obs = [
            ObservationsView(name='s', type='GENERATION'),
            ObservationsView(name='s', type='SPAN'),
        ]
        step = TraceStep('s', obs)
        assert step.count == 2

    def test_first_and_last(self):
        obs = [
            ObservationsView(name='s', type='GENERATION'),
            ObservationsView(name='s', type='SPAN'),
        ]
        step = TraceStep('s', obs)
        assert step.first.type == 'GENERATION'
        assert step.last.type == 'SPAN'

    def test_first_last_empty(self):
        step = TraceStep('empty', [])
        assert step.first is None
        assert step.last is None

    def test_lookup_generation(self):
        obs = [
            ObservationsView(name='rec', type='GENERATION'),
            ObservationsView(name='rec', type='SPAN'),
        ]
        step = TraceStep('rec', obs)
        gen = step.generation
        assert gen.type == 'GENERATION'

    def test_lookup_span_via_context_alias(self):
        obs = [ObservationsView(name='rec', type='SPAN')]
        step = TraceStep('rec', obs)
        result = step.context
        assert result.type == 'SPAN'

    def test_lookup_missing_type_raises(self):
        obs = [ObservationsView(name='rec', type='SPAN')]
        step = TraceStep('rec', obs)
        with pytest.raises(AttributeError):
            _ = step.generation

    def test_repr(self):
        obs = [ObservationsView(name='s', type='GENERATION')]
        step = TraceStep('s', obs)
        assert 'TraceStep' in repr(step)
        assert "'s'" in repr(step)


class TestTraceStepVariables:
    """Test variable extraction with prompt patterns."""

    def _make_patterns(self):
        class TestPatterns(PromptPatternsBase):
            @classmethod
            def _patterns_recommendation(cls):
                return {
                    'assessment': create_extraction_pattern(
                        'Assessment', r'(?:Recommendation|$)'
                    ),
                }

        return TestPatterns()

    def test_extract_variables(self):
        obs = [
            ObservationsView(
                name='recommendation',
                type='GENERATION',
                input='Assessment: patient is stable Recommendation: continue',
            ),
        ]
        patterns = self._make_patterns()
        step = TraceStep('recommendation', obs, prompt_patterns=patterns)
        variables = step.extract_variables()
        assert 'assessment' in variables
        assert 'patient is stable' in variables['assessment']

    def test_extract_variables_no_patterns(self):
        obs = [
            ObservationsView(
                name='recommendation', type='GENERATION', input='some text'
            ),
        ]
        step = TraceStep('recommendation', obs, prompt_patterns=None)
        assert step.extract_variables() == {}

    def test_extract_variables_no_generation(self):
        obs = [ObservationsView(name='rec', type='SPAN')]
        patterns = self._make_patterns()
        step = TraceStep('rec', obs, prompt_patterns=patterns)
        assert step.extract_variables() == {}

    def test_normalize_prompt_text_dict(self):
        text = TraceStep._normalize_prompt_text({'content': 'hello world'})
        assert text == 'hello world'

    def test_normalize_prompt_text_list(self):
        text = TraceStep._normalize_prompt_text(
            [{'content': 'line1'}, {'content': 'line2'}]
        )
        assert 'line1' in text
        assert 'line2' in text


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TestTraceFromRawObject:
    def test_groups_observations_by_name(self):
        raw = FakeRawTrace(
            observations=[
                FakeObservation(name='step1', type='GENERATION'),
                FakeObservation(name='step1', type='SPAN'),
                FakeObservation(name='step2', type='GENERATION'),
            ]
        )
        t = Trace(raw)
        assert set(t.step_names) == {'step1', 'step2'}

    def test_step_access_by_name(self):
        raw = FakeRawTrace(
            observations=[FakeObservation(name='recommendation', type='GENERATION')]
        )
        t = Trace(raw)
        step = t.recommendation
        assert isinstance(step, TraceStep)
        assert step.name == 'recommendation'
        assert step.count == 1

    def test_trace_level_attribute(self):
        raw = FakeRawTrace(id='abc', latency=1.5)
        t = Trace(raw)
        assert t.id == 'abc'
        assert t.latency == 1.5

    def test_observations_flat_list(self):
        raw = FakeRawTrace(
            observations=[
                FakeObservation(name='a'),
                FakeObservation(name='b'),
            ]
        )
        t = Trace(raw)
        assert len(t.observations) == 2

    def test_steps_property(self):
        raw = FakeRawTrace(
            observations=[
                FakeObservation(name='a', type='GENERATION'),
                FakeObservation(name='b', type='SPAN'),
            ]
        )
        t = Trace(raw)
        steps = t.steps
        assert 'a' in steps
        assert 'b' in steps
        assert isinstance(steps['a'], TraceStep)

    def test_raw_property(self):
        raw = FakeRawTrace()
        t = Trace(raw)
        assert t.raw is raw


class TestTraceFromDict:
    def test_dict_input(self):
        data = {
            'id': 'd1',
            'name': 'dict-trace',
            'observations': [
                {'name': 'step1', 'type': 'GENERATION'},
            ],
        }
        t = Trace(data)
        assert t.step_names == ['step1']
        assert t.id == 'd1'

    def test_dict_fuzzy_access(self):
        data = {
            'id': 'd2',
            'createdAt': '2024-01-01',
            'observations': [],
        }
        t = Trace(data)
        assert t.created_at == '2024-01-01'


class TestTraceFromList:
    def test_list_of_observations(self):
        obs = [
            {'name': 'a', 'type': 'GENERATION'},
            {'name': 'b', 'type': 'SPAN'},
        ]
        t = Trace(obs)
        assert t.step_names == ['a', 'b']
        assert t.raw is None


class TestTraceFuzzyStepAccess:
    def test_fuzzy_step_name_case(self):
        raw = FakeRawTrace(
            observations=[FakeObservation(name='Recommendation', type='GENERATION')]
        )
        t = Trace(raw)
        # Access with lowercase
        step = t.recommendation
        assert isinstance(step, TraceStep)
        assert step.name == 'Recommendation'

    def test_fuzzy_step_name_underscore(self):
        raw = FakeRawTrace(
            observations=[FakeObservation(name='caseAssessment', type='GENERATION')]
        )
        t = Trace(raw)
        step = t.case_assessment
        assert isinstance(step, TraceStep)


class TestTraceRepr:
    def test_repr_with_id(self):
        raw = FakeRawTrace(id='abc')
        t = Trace(raw)
        r = repr(t)
        assert 'Trace' in r
        assert 'abc' in r

    def test_repr_without_trace_obj(self):
        t = Trace([])
        r = repr(t)
        assert 'Trace' in r

    def test_missing_attribute_raises(self):
        raw = FakeRawTrace(observations=[])
        t = Trace(raw)
        with pytest.raises(AttributeError):
            _ = t.totally_nonexistent_attribute_xyz


# ---------------------------------------------------------------------------
# Trace-level properties (name, input, output)
# ---------------------------------------------------------------------------


class TestTraceProperties:
    def test_name_from_raw_object(self):
        raw = FakeRawTrace(name='my-workflow')
        t = Trace(raw)
        assert t.name == 'my-workflow'

    def test_input_from_raw_object(self):
        raw = FakeRawTrace(input={'query': 'hello'})
        t = Trace(raw)
        assert t.input == {'query': 'hello'}

    def test_output_from_raw_object(self):
        raw = FakeRawTrace(output={'answer': 'world'})
        t = Trace(raw)
        assert t.output == {'answer': 'world'}

    def test_name_from_dict_trace(self):
        data = {'name': 'dict-workflow', 'observations': []}
        t = Trace(data)
        assert t.name == 'dict-workflow'

    def test_input_output_from_dict_trace(self):
        data = {
            'input': {'q': 'test'},
            'output': {'a': 'result'},
            'observations': [],
        }
        t = Trace(data)
        assert t.input == {'q': 'test'}
        assert t.output == {'a': 'result'}

    def test_properties_none_when_no_trace_obj(self):
        """List-of-observations input has no trace object."""
        t = Trace([])
        assert t.name is None
        assert t.input is None
        assert t.output is None

    def test_name_not_shadowed_by_step(self):
        """A step named 'name' should not shadow trace.name."""
        raw = FakeRawTrace(
            name='my-workflow',
            observations=[FakeObservation(name='name', type='SPAN')],
        )
        t = Trace(raw)
        # Property always returns trace-level name
        assert t.name == 'my-workflow'


# ---------------------------------------------------------------------------
# Trace.walk()
# ---------------------------------------------------------------------------


class TestTraceWalk:
    def _make_single_root_trace(self):
        """
        root (id=r)
        ├── child1 (id=c1)
        │   └── grandchild (id=gc)
        └── child2 (id=c2)
        """
        return FakeRawTrace(
            observations=[
                FakeObservation(
                    id='r',
                    name='root',
                    type='SPAN',
                    start_time=datetime(2025, 1, 1, 12, 0, 0),
                ),
                FakeObservation(
                    id='c1',
                    name='child1',
                    type='SPAN',
                    parent_observation_id='r',
                    start_time=datetime(2025, 1, 1, 12, 0, 1),
                ),
                FakeObservation(
                    id='gc',
                    name='grandchild',
                    type='GENERATION',
                    parent_observation_id='c1',
                    start_time=datetime(2025, 1, 1, 12, 0, 2),
                ),
                FakeObservation(
                    id='c2',
                    name='child2',
                    type='SPAN',
                    parent_observation_id='r',
                    start_time=datetime(2025, 1, 1, 12, 0, 3),
                ),
            ],
        )

    def _make_multi_root_trace(self):
        """
        rootA (id=rA)
        └── childA (id=cA)

        rootB (id=rB)
        └── childB (id=cB)
        """
        return FakeRawTrace(
            observations=[
                FakeObservation(
                    id='rA',
                    name='rootA',
                    type='SPAN',
                    start_time=datetime(2025, 1, 1, 12, 0, 0),
                ),
                FakeObservation(
                    id='cA',
                    name='childA',
                    type='GENERATION',
                    parent_observation_id='rA',
                    start_time=datetime(2025, 1, 1, 12, 0, 1),
                ),
                FakeObservation(
                    id='rB',
                    name='rootB',
                    type='SPAN',
                    start_time=datetime(2025, 1, 1, 12, 0, 2),
                ),
                FakeObservation(
                    id='cB',
                    name='childB',
                    type='GENERATION',
                    parent_observation_id='rB',
                    start_time=datetime(2025, 1, 1, 12, 0, 3),
                ),
            ],
        )

    def test_walk_single_root(self):
        t = Trace(self._make_single_root_trace())
        ids = [n.id for n in t.walk()]
        assert ids == ['r', 'c1', 'gc', 'c2']

    def test_walk_multi_root(self):
        t = Trace(self._make_multi_root_trace())
        ids = [n.id for n in t.walk()]
        assert ids == ['rA', 'cA', 'rB', 'cB']

    def test_walk_empty(self):
        t = Trace(FakeRawTrace(observations=[]))
        assert list(t.walk()) == []

    def test_walk_with_depth(self):
        """walk() works with node.depth for indented display."""
        t = Trace(self._make_single_root_trace())
        depths = [(n.name, n.depth) for n in t.walk()]
        assert depths == [
            ('root', 0),
            ('child1', 1),
            ('grandchild', 2),
            ('child2', 1),
        ]


# ---------------------------------------------------------------------------
# Trace.find()
# ---------------------------------------------------------------------------


class TestTraceFind:
    def _make_multi_root_trace(self):
        return FakeRawTrace(
            observations=[
                FakeObservation(
                    id='rA',
                    name='rootA',
                    type='SPAN',
                    start_time=datetime(2025, 1, 1, 12, 0, 0),
                ),
                FakeObservation(
                    id='cA',
                    name='childA',
                    type='GENERATION',
                    parent_observation_id='rA',
                    start_time=datetime(2025, 1, 1, 12, 0, 1),
                ),
                FakeObservation(
                    id='rB',
                    name='rootB',
                    type='SPAN',
                    start_time=datetime(2025, 1, 1, 12, 0, 2),
                ),
                FakeObservation(
                    id='cB',
                    name='childB',
                    type='GENERATION',
                    parent_observation_id='rB',
                    start_time=datetime(2025, 1, 1, 12, 0, 3),
                ),
            ],
        )

    def test_find_by_name_across_roots(self):
        t = Trace(self._make_multi_root_trace())
        node = t.find(name='childB')
        assert node is not None
        assert node.id == 'cB'

    def test_find_by_type(self):
        t = Trace(self._make_multi_root_trace())
        node = t.find(type='GENERATION')
        assert node is not None
        assert node.id == 'cA'  # first generation in walk order

    def test_find_by_name_and_type(self):
        t = Trace(self._make_multi_root_trace())
        node = t.find(name='childB', type='GENERATION')
        assert node is not None
        assert node.id == 'cB'

    def test_find_no_match(self):
        t = Trace(self._make_multi_root_trace())
        assert t.find(name='nonexistent') is None

    def test_find_empty_trace(self):
        t = Trace(FakeRawTrace(observations=[]))
        assert t.find(name='anything') is None
