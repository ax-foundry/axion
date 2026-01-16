from typing import List

from axion.dataset import DatasetItem
from axion.error import MetricValidationError


def process_retrieved_content(item: DatasetItem) -> List:
    """
    Normalize the retrieved content field of a dataset item into a list.

    This utility ensures that the `retrieved_content` attribute of a
    `DatasetItem` is consistently represented as a list, regardless of
    whether it was originally provided as a string, list, or another type.

    Args:
        item: The dataset item whose retrieved content should be processed.

    Returns:
        List: A normalized list of retrieved content strings. Returns an
        empty list if no valid content is available.
    """
    # Ensure retrieved_content is a list
    if isinstance(item.retrieved_content, str):
        retrieval_context = [item.retrieved_content]
    elif isinstance(item.retrieved_content, list):
        retrieval_context = item.retrieved_content
    else:
        retrieval_context = []
    return retrieval_context


def validate_required_metric_fields(
    item: DatasetItem,
    required_fields: List[str],
    name: str,
) -> None:
    """
    Validate that the input item contains all required fields for this metric.

    Args:
        item: The DatasetItem to validate
        required_fields: Fields required for metric
        name: Metric Name
    """
    missing_fields = []
    available_fields = []

    # Skip pydantic internal attributes that trigger deprecation warnings when accessed
    _pydantic_skip = {'model_fields', 'model_computed_fields', 'model_config', 'model_extra'}

    # Check what fields are available and which are missing
    for field_name in dir(item):
        if not field_name.startswith('_') and field_name not in _pydantic_skip:
            value = getattr(item, field_name, None)
            if value is not None:
                available_fields.append(field_name)

    for field in required_fields:
        value = getattr(item, field, None)
        if value is None or (isinstance(value, str) and value.strip() == ''):
            missing_fields.append(field)

    if missing_fields:
        # Create helpful error message
        error_msg = f"❌ Metric '{name}' cannot run due to missing required data.\n\n"

        # Show what's missing
        error_msg += f"Missing fields: {', '.join(missing_fields)}\n"

        # Show what's required
        error_msg += f"Required fields: {', '.join(required_fields)}\n\n"

        # Provide helpful suggestions
        error_msg += '💡 Suggestions:\n'
        for missing_field in missing_fields:
            if missing_field == 'query':
                error_msg += '  • Ensure your dataset includes user questions or search queries\n'
            elif missing_field == 'actual_output':
                error_msg += (
                    '  • Make sure your system responses/outputs are captured\n'
                )
            elif missing_field == 'expected_output':
                error_msg += '  • Add ground truth answers or expected responses to your dataset\n'
            elif missing_field == 'retrieved_content':
                error_msg += '  • Include the retrieved context/documents used to generate responses\n'
            else:
                error_msg += f"  • Add '{missing_field}' to your dataset\n"

        # Show example of correct data structure
        error_msg += '\n📋 Example of correct data structure:\n'
        example_data = {}
        for field in required_fields:
            if field == 'query':
                example_data[field] = 'What is machine learning?'
            elif field == 'actual_output':
                example_data[field] = 'Machine learning is a subset of AI...'
            elif field == 'expected_output':
                example_data[field] = 'Expected answer about ML definition'
            elif field == 'retrieved_content':
                example_data[field] = ['Document 1 about ML', 'Document 2 about AI']
            else:
                example_data[field] = f'<{field}_value>'

        import json

        error_msg += json.dumps(example_data, indent=2)

        raise MetricValidationError(error_msg)
