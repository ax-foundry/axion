from __future__ import annotations

from types import SimpleNamespace

from axion._core.tracing.collection.scores import TraceScore


def _raw(data_type='NUMERIC', **extra):
    base = dict(
        name='my_metric',
        value=0.85,
        data_type=data_type,
        trace_id='trace-1',
        observation_id=None,
        string_value=None,
        comment='good',
        source='API',
        timestamp=None,
    )
    base.update(extra)
    return SimpleNamespace(**base)


def test_from_langfuse_numeric():
    raw = _raw(data_type='NUMERIC')
    score = TraceScore.from_langfuse(raw)
    assert score.name == 'my_metric'
    assert score.value == 0.85
    assert score.data_type == 'NUMERIC'
    assert score.string_value is None
    assert score.comment == 'good'
    assert score.trace_id == 'trace-1'


def test_from_langfuse_categorical():
    raw = _raw(data_type='CATEGORICAL', value=1.0, string_value='completed_scope')
    score = TraceScore.from_langfuse(raw)
    assert score.data_type == 'CATEGORICAL'
    assert score.string_value == 'completed_scope'
    assert score.value == 1.0


def test_from_langfuse_boolean():
    raw = _raw(data_type='BOOLEAN', value=1.0, string_value='True')
    score = TraceScore.from_langfuse(raw)
    assert score.data_type == 'BOOLEAN'
    assert score.string_value == 'True'


def test_from_langfuse_correction():
    raw = _raw(data_type='CORRECTION', value=0.0, string_value='needs_review')
    score = TraceScore.from_langfuse(raw)
    assert score.data_type == 'CORRECTION'
    assert score.string_value == 'needs_review'


def test_from_langfuse_missing_optional_fields():
    # Only the required fields; optional ones absent on the raw object.
    raw = SimpleNamespace(name='x', value=0.5, data_type='NUMERIC', source=None)
    score = TraceScore.from_langfuse(raw)
    assert score.trace_id is None
    assert score.observation_id is None
    assert score.string_value is None
    assert score.comment is None
    assert score.source is None


def test_trace_score_frozen():
    score = TraceScore(name='x', value=0.5, data_type='NUMERIC')
    try:
        score.value = 1.0  # type: ignore[misc]
        assert False, 'should have raised'
    except Exception:
        pass
