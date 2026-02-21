"""Tests for SmartAccess, SmartDict, and SmartObject."""

import pytest

from axion._core.tracing.collection.smart_access import (
    SmartAccess,
    SmartDict,
    SmartObject,
    _normalize_key,
)


# ---------------------------------------------------------------------------
# _normalize_key
# ---------------------------------------------------------------------------


class TestNormalizeKey:
    def test_lowercase_and_strip_underscores(self):
        assert _normalize_key('product_type') == 'producttype'

    def test_camel_case(self):
        assert _normalize_key('productType') == 'producttype'

    def test_already_normalized(self):
        assert _normalize_key('name') == 'name'

    def test_mixed_case_with_underscores(self):
        assert _normalize_key('My_Var_Name') == 'myvarname'


# ---------------------------------------------------------------------------
# SmartDict
# ---------------------------------------------------------------------------


class TestSmartDict:
    def test_exact_key_access(self):
        sd = SmartDict({'name': 'alice', 'age': 30})
        assert sd.name == 'alice'
        assert sd.age == 30

    def test_bracket_access(self):
        sd = SmartDict({'key': 'value'})
        assert sd['key'] == 'value'

    def test_fuzzy_snake_to_camel(self):
        sd = SmartDict({'productType': 'widget'})
        assert sd.product_type == 'widget'

    def test_fuzzy_camel_to_snake(self):
        sd = SmartDict({'created_at': '2024-01-01'})
        assert sd.createdAt == '2024-01-01'

    def test_missing_key_raises_attribute_error(self):
        sd = SmartDict({'a': 1})
        with pytest.raises(AttributeError, match='no attribute'):
            _ = sd.nonexistent

    def test_nested_dict_wrapping(self):
        sd = SmartDict({'inner': {'x': 10}})
        result = sd.inner
        assert isinstance(result, SmartDict)
        assert result.x == 10

    def test_list_wrapping(self):
        sd = SmartDict({'items': [{'val': 1}, {'val': 2}]})
        items = sd.items
        assert len(items) == 2
        assert items[0].val == 1
        assert items[1].val == 2

    def test_to_dict(self):
        data = {'a': 1, 'b': 2}
        sd = SmartDict(data)
        assert sd.to_dict() is data

    def test_repr(self):
        sd = SmartDict({'x': 1, 'y': 2})
        assert 'SmartDict' in repr(sd)
        assert 'x' in repr(sd)


# ---------------------------------------------------------------------------
# SmartObject
# ---------------------------------------------------------------------------


class TestSmartObject:
    def test_attribute_access(self):
        class Obj:
            name = 'test'
            value = 42

        so = SmartObject(Obj())
        assert so.name == 'test'
        assert so.value == 42

    def test_fuzzy_access(self):
        class Obj:
            productType = 'widget'

        so = SmartObject(Obj())
        assert so.product_type == 'widget'

    def test_nested_dict_wrapping(self):
        class Obj:
            data = {'key': 'val'}

        so = SmartObject(Obj())
        result = so.data
        assert isinstance(result, SmartDict)
        assert result.key == 'val'

    def test_nested_object_wrapping(self):
        class Inner:
            x = 10

        class Outer:
            inner = Inner()

        so = SmartObject(Outer())
        assert isinstance(so.inner, SmartObject)
        assert so.inner.x == 10

    def test_missing_attribute_raises(self):
        class Obj:
            pass

        so = SmartObject(Obj())
        with pytest.raises(AttributeError):
            _ = so.nonexistent

    def test_repr(self):
        class Obj:
            def __repr__(self):
                return '<MyObj>'

        so = SmartObject(Obj())
        assert repr(so) == '<MyObj>'


# ---------------------------------------------------------------------------
# SmartAccess base class
# ---------------------------------------------------------------------------


class TestSmartAccessBase:
    def test_subclass_must_implement_lookup(self):
        sa = SmartAccess()
        with pytest.raises(AttributeError):
            _ = sa.anything

    def test_wrap_primitives_unchanged(self):
        sa = SmartAccess()
        assert sa._wrap(42) == 42
        assert sa._wrap('hello') == 'hello'
        assert sa._wrap(None) is None

    def test_wrap_dict_returns_smart_dict(self):
        sa = SmartAccess()
        result = sa._wrap({'a': 1})
        assert isinstance(result, SmartDict)

    def test_wrap_list_recursive(self):
        sa = SmartAccess()
        result = sa._wrap([{'a': 1}, 'plain'])
        assert isinstance(result[0], SmartDict)
        assert result[1] == 'plain'
