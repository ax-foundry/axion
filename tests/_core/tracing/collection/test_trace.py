"""Tests for Trace and TraceStep."""

from dataclasses import dataclass, field
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
    name: str = 'step1'
    type: str = 'GENERATION'
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
