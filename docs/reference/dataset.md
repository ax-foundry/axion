---
icon: custom/database
---
# Dataset API Reference

Core data structures for building and managing evaluation datasets.

<div class="ref-import" markdown>

```python
from axion import Dataset, DatasetItem
```

</div>

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">D</span>
<p class="rule-card__title">Dataset</p>
<p class="rule-card__desc">Container for evaluation items. Supports JSON/CSV/DataFrame I/O, filtering, merging, and synthetic generation.</p>
</div>
<div class="rule-card">
<span class="rule-card__number">I</span>
<p class="rule-card__title">DatasetItem</p>
<p class="rule-card__desc">Individual test case with query, expected/actual output, context, metadata, and conversation history.</p>
</div>
</div>

---

## Dataset

::: axion.dataset.Dataset
    options:
      show_root_heading: true
      members:
        - create
        - add_item
        - add_items
        - get_item_by_id
        - filter
        - read_json
        - read_csv
        - read_dataframe
        - to_json
        - to_csv
        - to_dataframe
        - load_dataframe
        - get_summary
        - get_summary_table
        - execute_dataset_items_from_api
        - merge_response_into_dataset_items
        - synthetic_generate_from_directory
        - items

---

## DatasetItem

::: axion.dataset.DatasetItem
    options:
      show_root_heading: true
      members:
        - id
        - query
        - actual_output
        - expected_output
        - conversation
        - retrieved_content
        - latency
        - judgment
        - critique
        - acceptance_criteria
        - additional_input
        - additional_output
        - metadata
        - trace
        - trace_id
        - observation_id
        - actual_ranking
        - expected_ranking
        - tools_called
        - expected_tools
        - user_tags
        - conversation_extraction_strategy
        - conversation_stats
        - agent_trajectory
        - has_errors
        - to_transcript
        - extract_by_tag
        - get
        - keys
        - values
        - items
        - subset
        - evaluation_fields
        - update
        - update_runtime
        - merge_metadata
        - from_dict

---

<div class="ref-nav" markdown>

[Working with Datasets Guide :octicons-arrow-right-24:](../guides/datasets.md){ .md-button .md-button--primary }
[Running Evaluations :octicons-arrow-right-24:](../guides/evaluation.md){ .md-button }

</div>
