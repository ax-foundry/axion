"""Tests for models: enums, Usage, TraceView, ObservationsView."""

from dataclasses import dataclass

from axion._core.tracing.collection.models import (
    ModelUsageUnit,
    ObservationLevel,
    ObservationsView,
    TraceView,
    Usage,
)
from axion._core.tracing.collection.smart_access import SmartDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_model_usage_unit_values(self):
        assert ModelUsageUnit.TOKENS.value == 'TOKENS'
        assert ModelUsageUnit.CHARACTERS.value == 'CHARACTERS'

    def test_observation_level_values(self):
        assert ObservationLevel.DEFAULT.value == 'DEFAULT'
        assert ObservationLevel.ERROR.value == 'ERROR'


# ---------------------------------------------------------------------------
# Usage dataclass
# ---------------------------------------------------------------------------


class TestUsage:
    def test_defaults(self):
        u = Usage()
        assert u.input == 0
        assert u.output == 0
        assert u.total == 0
        assert u.unit == ModelUsageUnit.TOKENS

    def test_custom_values(self):
        u = Usage(input=100, output=50, total=150, unit=ModelUsageUnit.CHARACTERS)
        assert u.input == 100
        assert u.total == 150
        assert u.unit == ModelUsageUnit.CHARACTERS


# ---------------------------------------------------------------------------
# TraceView
# ---------------------------------------------------------------------------


class TestTraceView:
    def test_from_kwargs(self):
        tv = TraceView(id='t1', name='my-trace', latency=1.5)
        assert tv.id == 't1'
        assert tv.name == 'my-trace'
        assert tv.latency == 1.5

    def test_from_dict(self):
        tv = TraceView(data={'id': 't2', 'name': 'test', 'environment': 'prod'})
        assert tv.id == 't2'
        assert tv.environment == 'prod'

    def test_from_object(self):
        @dataclass
        class FakeTrace:
            id: str = 'obj1'
            name: str = 'from-obj'

        tv = TraceView(data=FakeTrace())
        assert tv.id == 'obj1'
        assert tv.name == 'from-obj'

    def test_observations_default_empty(self):
        tv = TraceView(id='t3')
        assert tv.observations == []

    def test_fuzzy_access(self):
        tv = TraceView(data={'createdAt': '2024-01-01'})
        assert tv.created_at == '2024-01-01'

    def test_nested_dict_wrapped(self):
        tv = TraceView(data={'metadata': {'key': 'val'}})
        result = tv.metadata
        assert isinstance(result, SmartDict)
        assert result.key == 'val'

    def test_repr(self):
        tv = TraceView(id='abc', name='test')
        r = repr(tv)
        assert 'abc' in r
        assert 'test' in r


# ---------------------------------------------------------------------------
# ObservationsView
# ---------------------------------------------------------------------------


class TestObservationsView:
    def test_from_kwargs(self):
        ov = ObservationsView(name='step1', type='GENERATION', input='hello')
        assert ov.name == 'step1'
        assert ov.type == 'GENERATION'
        assert ov.input == 'hello'

    def test_from_dict(self):
        ov = ObservationsView(data={'name': 'step2', 'type': 'SPAN', 'output': 'world'})
        assert ov.name == 'step2'
        assert ov.output == 'world'

    def test_from_object(self):
        @dataclass
        class FakeObs:
            name: str = 'obs1'
            type: str = 'EVENT'

        ov = ObservationsView(data=FakeObs())
        assert ov.name == 'obs1'
        assert ov.type == 'EVENT'

    def test_fuzzy_access(self):
        ov = ObservationsView(data={'modelId': 'gpt-4'})
        assert ov.model_id == 'gpt-4'

    def test_repr(self):
        ov = ObservationsView(name='rec', type='GENERATION')
        r = repr(ov)
        assert 'rec' in r
        assert 'GENERATION' in r
