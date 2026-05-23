from dataclasses import dataclass
from typing import Any, Dict, Optional

from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader


@dataclass
class FakeLangfuseItem:
    id: str = 'lf-1'
    input: Any = None
    expected_output: Any = None
    metadata: Optional[Dict[str, Any]] = None
    source_trace_id: Optional[str] = None


def _loader() -> LangfuseTraceLoader:
    # Skip __init__ — the method under test is pure and doesn't touch the
    # Langfuse client.
    return LangfuseTraceLoader.__new__(LangfuseTraceLoader)


def test_expected_output_extracted_from_bare_value() -> None:
    item = FakeLangfuseItem(
        input={'query': 'q'},
        expected_output='answer',
    )
    result = _loader()._build_dataset_item(item, field_map={})
    assert result.expected_output == 'answer'
    assert result.additional_output == {}


def test_expected_output_unwrapped_from_wrapper_dict() -> None:
    item = FakeLangfuseItem(
        input={'query': 'q'},
        expected_output={'expected_output': 'Building 1: AAL 221'},
    )
    result = _loader()._build_dataset_item(item, field_map={})
    assert result.expected_output == 'Building 1: AAL 221'
    assert result.additional_output == {}


def test_additional_output_pulled_from_explicit_key() -> None:
    item = FakeLangfuseItem(
        input={'query': 'q'},
        expected_output={
            'expected_output': 'Building 1: AAL 221',
            'additional_output': {
                'building_id': '01KQAH33321JGTA0CG9SJF09MB',
                'aal': 221,
            },
        },
    )
    result = _loader()._build_dataset_item(item, field_map={})
    assert result.expected_output == 'Building 1: AAL 221'
    assert result.additional_output == {
        'building_id': '01KQAH33321JGTA0CG9SJF09MB',
        'aal': 221,
    }


def test_additional_output_falls_back_to_leftover_keys() -> None:
    # When callers don't nest under `additional_output`, leftover keys on the
    # wrapper dict are treated as additional_output — symmetric with how
    # `additional_input` derives from leftover input keys.
    item = FakeLangfuseItem(
        input={'query': 'q'},
        expected_output={
            'expected_output': 'a',
            'building_id': 'x',
            'aal': 221,
        },
    )
    result = _loader()._build_dataset_item(item, field_map={})
    assert result.expected_output == 'a'
    assert result.additional_output == {'building_id': 'x', 'aal': 221}


def test_additional_output_empty_when_expected_is_bare_string() -> None:
    item = FakeLangfuseItem(
        input={'query': 'q'},
        expected_output='just a string',
    )
    result = _loader()._build_dataset_item(item, field_map={})
    assert result.additional_output == {}
