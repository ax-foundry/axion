# Dataset API Reference

Core data structures for evaluation datasets.

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
