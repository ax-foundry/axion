"""Span attributes that make a Langfuse trace an experiment item.

Langfuse links an experiment item to its experiment through attributes on the
item trace's root span, not through a separate API call. See
https://langfuse.com/integrations/native/opentelemetry/experiments.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Dict, Mapping, Optional

EXPERIMENT_ID = 'langfuse.experiment.id'
EXPERIMENT_NAME = 'langfuse.experiment.name'
EXPERIMENT_METADATA = 'langfuse.experiment.metadata'
EXPERIMENT_DATASET_ID = 'langfuse.experiment.dataset.id'
EXPERIMENT_ITEM_ID = 'langfuse.experiment.item.id'
EXPERIMENT_ITEM_EXPECTED_OUTPUT = 'langfuse.experiment.item.expected_output'
EXPERIMENT_ITEM_METADATA = 'langfuse.experiment.item.metadata'
EXPERIMENT_ITEM_ROOT_OBSERVATION_ID = 'langfuse.experiment.item.root_observation_id'


def experiment_id_for(
    project_id: Optional[str], dataset_id: Optional[str], run_name: str
) -> Optional[str]:
    """The id Langfuse derives for a dataset experiment, or None without both ids.

    The server derives it the same way, so every write under one run name lands
    in one experiment, and the experiment's dataset run id is this id.
    """
    if not project_id or not dataset_id:
        return None
    payload = json.dumps(
        ['langfuse-experiment-v1', project_id, dataset_id, run_name],
        separators=(',', ':'),
        ensure_ascii=False,
    )
    return sha256(payload.encode('utf-8')).hexdigest()[:16]


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def experiment_item_attributes(
    *,
    experiment_id: str,
    run_name: str,
    dataset_id: Optional[str],
    item_id: str,
    expected_output: Any,
    experiment_metadata: Mapping[str, Any],
    item_metadata: Mapping[str, Any],
) -> Dict[str, str]:
    """Root-span attributes for one experiment item, all string-valued.

    The root observation id is the span's own id, so the caller adds it once
    the span exists. Metadata is written one attribute per key.
    """
    attributes = {
        EXPERIMENT_ID: experiment_id,
        EXPERIMENT_NAME: run_name,
        EXPERIMENT_ITEM_ID: item_id,
    }
    if dataset_id:
        attributes[EXPERIMENT_DATASET_ID] = dataset_id
    if expected_output is not None:
        attributes[EXPERIMENT_ITEM_EXPECTED_OUTPUT] = json.dumps(
            expected_output, default=str, ensure_ascii=False
        )
    for key, value in experiment_metadata.items():
        if value is not None:
            attributes[f'{EXPERIMENT_METADATA}.{key}'] = _text(value)
    for key, value in item_metadata.items():
        if value is not None:
            attributes[f'{EXPERIMENT_ITEM_METADATA}.{key}'] = _text(value)
    return attributes
