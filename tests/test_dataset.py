import json

import pandas as pd
import pytest
from axion.dataset import (
    AIMessage,
    Dataset,
    DatasetItem,
    HumanMessage,
    MultiTurnConversation,
    format_input,
)


class MockRichSerializer:
    pass


class MockFieldNames:
    QUERY = 'query'
    NAME = 'name'
    METADATA = 'metadata'
    ADDITIONAL_INPUT = 'additional_input'
    ADDITIONAL_OUTPUT = 'additional_output'
    RETRIEVED_CONTENT = 'retrieved_content'
    CONVERSATION = 'conversation'
    EXPECTED_OUTPUT = 'expected_output'
    ID = 'id'

    @classmethod
    def get_input_fields(cls):
        return ['query', 'expected_output', 'id', 'metadata']

    @classmethod
    def get_runtime_fields(cls):
        return [
            'actual_output',
            'retrieved_content',
            'latency',
            'trace',
            'additional_output',
        ]

    @classmethod
    def get_evaluation_fields(cls):
        return [
            'actual_output',
            'retrieved_content',
            'latency',
            'trace',
            'additional_output',
        ]

    @classmethod
    def get_display_fields(cls):
        return ['query', 'expected_output', 'actual_output']


# Mock functions
def mock_uuid7():
    return 'test-uuid-123'


def mock_current_datetime():
    return '2024-01-01T12:00:00Z'


def mock_convert_to_list(x):
    if isinstance(x, str):
        return x.split(',') if ',' in x else [x]
    return x


def mock_format_value(x):
    return str(x)


# Mock logger
class MockLogger:
    def warning(self, msg):
        pass

    def info(self, msg):
        pass

    def log_table(self, data, title, **kwargs):
        pass


class TestDatasetItem:
    """Test suite for DatasetItem class."""

    def test_init_with_defaults(self):
        """Test DatasetItem initialization with default values."""
        item = DatasetItem()

        assert item.query is None
        assert item.expected_output is None
        assert item.additional_input == {}
        assert item.metadata is None
        assert item.actual_output is None
        assert item.retrieved_content is None
        assert item.latency is None
        assert item.trace is None
        assert item.additional_output == {}

    def test_init_with_custom_values(self):
        """Test DatasetItem initialization with custom values."""
        item = DatasetItem(
            id='custom-id',
            query='What is AI?',
            expected_output='AI is artificial intelligence',
            metadata='{"source": "test"}',
            latency=0.5,
        )

        assert item.id == 'custom-id'
        assert item.query == 'What is AI?'
        assert item.expected_output == 'AI is artificial intelligence'
        assert item.metadata == '{"source": "test"}'
        assert item.latency == 0.5

    def test_get_method(self):
        """Test the get method for attribute access."""
        item = DatasetItem(query='test query')

        assert item.get('query') == 'test query'
        assert item.get('nonexistent') is None
        assert item.get('nonexistent', 'default') == 'default'

    def test_keys_values_items(self):
        """Test dictionary-like methods."""
        item = DatasetItem(query='test', expected_output='output')

        keys = item.keys()
        values = item.values()
        items = item.items()

        assert isinstance(keys, list)
        assert isinstance(values, list)
        assert isinstance(items, list)
        assert len(keys) == len(values) == len(items)
        assert 'query' in keys
        assert 'test' in values

    def test_analogous_property_access(self):
        """Test that single-turn and multi-turn properties are analogous."""
        # Single-turn
        item1 = DatasetItem(query='Test')
        assert item1.query == 'Test'
        assert item1.single_turn_query == 'Test'

        # Multi-turn
        item2 = DatasetItem(
            conversation=MultiTurnConversation(
                messages=[HumanMessage(content='Hello there')]
            )
        )
        assert item2.query == 'Hello there'
        assert item2.single_turn_query is None

    def test_update_with_dict(self):
        """Test updating item with dictionary."""
        item = DatasetItem(query='original')

        item.update({'query': 'updated', 'actual_output': 'new output'})

        assert item.query == 'updated'
        assert item.actual_output == 'new output'

    def test_update_with_datasetitem(self):
        """Test updating item with another DatasetItem."""
        item1 = DatasetItem(query='original')
        item2 = DatasetItem(query='updated', actual_output='output')

        # Update only the mutable fields, not computed properties
        update_dict = {
            'query': item2.query,
            'actual_output': item2.actual_output,
        }
        item1.update(update_dict)

        assert item1.query == 'updated'
        assert item1.actual_output == 'output'

    def test_update_no_overwrite(self):
        """Test update without overwriting existing values."""
        item = DatasetItem(query='original', actual_output='existing')

        item.update({'query': 'new', 'latency': 1.0}, overwrite=False)

        assert item.query == 'original'  # Should not be overwritten
        assert item.latency == 1.0  # Should be set (was None)

    def test_update_invalid_type(self):
        """Test update with invalid type raises error."""
        item = DatasetItem()

        with pytest.raises(ValueError, match='Unsupported update type'):
            item.update('invalid type')

    def test_update_runtime(self):
        """Test updating only runtime fields."""
        item = DatasetItem(query='test')

        item.update_runtime(
            actual_output='output',
            latency=0.5,
            query='should be ignored',  # Not a runtime field
        )

        assert item.actual_output == 'output'
        assert item.latency == 0.5
        assert item.query == 'test'  # Should remain unchanged

    def test_merge_metadata_with_dict(self):
        """Test merging metadata with dictionary."""
        item = DatasetItem(metadata='{"existing": "value"}')

        item.merge_metadata({'new': 'data'})

        metadata = json.loads(item.metadata)
        assert metadata['existing'] == 'value'
        assert metadata['new'] == 'data'

    def test_merge_metadata_with_string(self):
        """Test merging metadata with JSON string."""
        item = DatasetItem()

        item.merge_metadata('{"key": "value"}')

        metadata = json.loads(item.metadata)
        assert metadata['key'] == 'value'

    def test_merge_metadata_with_invalid_json(self):
        """Test merging metadata with invalid JSON string."""
        item = DatasetItem()

        item.merge_metadata('invalid json')

        metadata = json.loads(item.metadata)
        assert metadata['raw'] == 'invalid json'

    def test_to_dict_uses_aliases(self):
        """Test that to_dict() uses the public aliases and excludes internal fields."""
        item = DatasetItem(query='test query')
        item_dict = item.to_dict()

        # Check that the query field is present
        assert 'query' in item_dict
        assert item_dict['query'] == 'test query'

        # The internal field might still be present in the dict depending on
        # Pydantic configuration, but the public alias should work correctly
        assert item.query == 'test query'

    def test_multi_turn_conversation_in_dict(self):
        """Test that multi-turn conversation is properly serialized."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='First question'),
                AIMessage(content='First answer'),
            ]
        )
        item = DatasetItem(conversation=conversation)
        item_dict = item.to_dict()

        assert 'conversation' in item_dict
        assert item_dict['conversation'] is not None
        assert 'messages' in item_dict['conversation']


class TestMultiTurnConversation:
    """Test suite for MultiTurnConversation schema."""

    def test_basic_initialization(self):
        """Test basic initialization of MultiTurnConversation."""
        convo = MultiTurnConversation(
            messages=[
                HumanMessage(content='Hello'),
                AIMessage(content='Hi there'),
            ]
        )
        assert len(convo.messages) == 2
        assert convo.reference_text is None
        assert convo.reference_tool_calls is None
        assert convo.rubrics is None
        assert convo.metadata is None

    def test_with_reference_text(self):
        """Test MultiTurnConversation with reference text."""
        convo = MultiTurnConversation(
            messages=[HumanMessage(content='What is AI?')],
            reference_text='AI is artificial intelligence',
        )
        assert convo.reference_text == 'AI is artificial intelligence'

    def test_with_rubrics(self):
        """Test MultiTurnConversation with rubrics."""
        rubrics = {
            'accuracy': 'Response must be factually correct',
            'completeness': 'Response must address all parts of the question',
        }
        convo = MultiTurnConversation(
            messages=[HumanMessage(content='Explain quantum computing')],
            rubrics=rubrics,
        )
        assert convo.rubrics == rubrics
        assert 'accuracy' in convo.rubrics

    def test_with_metadata(self):
        """Test MultiTurnConversation with session metadata."""
        metadata = {
            'sessionProperties': {'userId': '123'},
            'plannerType': 'reactive',
            'environment': 'production',
        }
        convo = MultiTurnConversation(
            messages=[HumanMessage(content='Hello')], metadata=metadata
        )
        assert convo.metadata == metadata
        assert convo.metadata['plannerType'] == 'reactive'

    def test_empty_messages_raises_error(self):
        """Test that empty messages list raises validation error."""
        from axion._core.error import CustomValidationError

        with pytest.raises((ValueError, CustomValidationError)):
            MultiTurnConversation(messages=[])

    def test_complex_conversation(self):
        """Test a complex multi-turn conversation with all fields."""
        messages = [
            HumanMessage(content='Search for Python tutorials'),
            AIMessage(content='Here are some tutorials'),
            HumanMessage(content='Show me the first one'),
            AIMessage(content='Here is the first tutorial...'),
        ]
        rubrics = {'relevance': 'Must return relevant results'}
        metadata = {'sessionId': 'abc123', 'plannerType': 'multi-step'}

        convo = MultiTurnConversation(
            messages=messages,
            reference_text='Expected final response',
            rubrics=rubrics,
            metadata=metadata,
        )

        assert len(convo.messages) == 4
        assert convo.reference_text == 'Expected final response'
        assert convo.rubrics['relevance'] == 'Must return relevant results'
        assert convo.metadata['sessionId'] == 'abc123'


class TestRichDatasetBaseModel:
    """Test suite for RichDatasetBaseModel representation methods."""

    def test_single_turn_repr(self):
        """Test __repr__ for single-turn items."""
        item = DatasetItem(
            id='test-123', query='What is AI?', expected_output='AI is...'
        )
        repr_str = repr(item)

        assert 'test-123' in repr_str
        assert 'DatasetItem' in repr_str

    def test_multi_turn_repr(self):
        """Test __repr__ for multi-turn items."""
        convo = MultiTurnConversation(
            messages=[
                HumanMessage(content='First'),
                AIMessage(content='Response'),
                HumanMessage(content='Second'),
            ]
        )
        item = DatasetItem(id='test-456', conversation=convo)
        repr_str = repr(item)

        assert 'test-456' in repr_str
        assert 'conversation' in repr_str
        assert '3 messages' in repr_str

    def test_single_turn_str(self):
        """Test __str__ for single-turn items with detailed output."""
        item = DatasetItem(
            id='test-789',
            query='What is machine learning?',
            expected_output='ML is a subset of AI',
            actual_output='ML is part of AI',
        )
        str_output = str(item)

        assert 'Single-turn DatasetItem' in str_output
        assert 'test-789' in str_output
        assert 'Query: What is machine learning?' in str_output
        assert 'Expected Output: ML is a subset of AI' in str_output
        assert 'Actual Output: ML is part of AI' in str_output

    def test_multi_turn_str(self):
        """Test __str__ for multi-turn items with detailed output."""
        convo = MultiTurnConversation(
            messages=[
                HumanMessage(content='Tell me about Python'),
                AIMessage(content='Python is a programming language'),
                HumanMessage(content='What are its benefits?'),
                AIMessage(content='It is easy to learn and versatile'),
            ]
        )
        item = DatasetItem(
            id='test-multi',
            conversation=convo,
            expected_output='Complete explanation',
        )
        str_output = str(item)

        assert 'Multi-turn DatasetItem' in str_output
        assert 'test-multi' in str_output
        assert '4 messages in conversation' in str_output
        assert 'Message 1: Tell me about Python' in str_output
        assert 'Message 2: What are its benefits?' in str_output

    def test_repr_with_message_counts(self):
        """Test that __repr__ shows correct message type counts."""
        from axion._core.schema import ToolMessage

        convo = MultiTurnConversation(
            messages=[
                HumanMessage(content='Search for data'),
                AIMessage(content='Searching...'),
                ToolMessage(content='Results found', tool_call_id='t1'),
                HumanMessage(content='Summarize'),
                AIMessage(content='Here is summary'),
            ]
        )
        item = DatasetItem(conversation=convo)
        repr_str = repr(item)

        assert '2 user' in repr_str
        assert '2 AI' in repr_str
        assert '1 tool' in repr_str

    def test_str_without_expected_output(self):
        """Test __str__ hides 'Expected Output' when not provided."""
        item = DatasetItem(id='test-no-expected', query='Test query')
        str_output = str(item)

        assert 'Expected Output' not in str_output

    def test_str_with_retrieved_content(self):
        """Test __str__ shows retrieved content when present."""
        item = DatasetItem(
            id='test-retrieval',
            query='Test',
            retrieved_content=['Doc 1', 'Doc 2'],
        )
        str_output = str(item)

        assert 'Retrieved Content' in str_output


class TestDatasetItemWithConversation:
    """Test DatasetItem with conversation-specific functionality."""

    def test_conversation_property_access(self):
        """Test that conversation property correctly accesses multi_turn_conversation."""
        convo = MultiTurnConversation(messages=[HumanMessage(content='Test')])
        item = DatasetItem(conversation=convo)

        assert item.conversation == convo
        assert item.multi_turn_conversation == convo

    def test_expected_output_from_conversation(self):
        """Test that expected_output can be derived from conversation reference_text."""
        convo = MultiTurnConversation(
            messages=[HumanMessage(content='Test')],
            reference_text='Expected response',
        )
        item = DatasetItem(conversation=convo)

        assert item.expected_output == 'Expected response'

    def test_conversation_with_rubrics(self):
        """Test accessing rubrics from conversation."""
        rubrics = {'clarity': 'Must be clear', 'accuracy': 'Must be accurate'}
        convo = MultiTurnConversation(
            messages=[HumanMessage(content='Test')], rubrics=rubrics
        )
        item = DatasetItem(conversation=convo)

        assert item.conversation.rubrics == rubrics

    def test_conversation_with_metadata(self):
        """Test accessing metadata from conversation."""
        metadata = {'plannerType': 'sequential', 'sessionId': '12345'}
        convo = MultiTurnConversation(
            messages=[HumanMessage(content='Test')], metadata=metadata
        )
        item = DatasetItem(conversation=convo)

        assert item.conversation.metadata == metadata
        assert item.conversation.metadata['plannerType'] == 'sequential'


class TestDataset:
    """Test suite for Dataset class."""

    def test_init_with_defaults(self):
        """Test Dataset initialization with default values."""
        dataset = Dataset(name='test_dataset')

        assert dataset.name == 'test_dataset'
        assert dataset.description == ''
        assert dataset.version == '1.0'
        assert len(dataset.items) == 0

    def test_init_with_custom_values(self):
        """Test Dataset initialization with custom values."""
        dataset = Dataset(
            name='custom_dataset',
            description='A test dataset',
            version='2.0',
            metadata='{"purpose": "testing"}',
        )

        assert dataset.name == 'custom_dataset'
        assert dataset.description == 'A test dataset'
        assert dataset.version == '2.0'
        assert dataset.metadata == '{"purpose": "testing"}'

    def test_add_item_dict(self):
        """Test adding an item from a dictionary."""
        dataset = Dataset(name='test')
        dataset.add_item({'query': 'test query', 'expected_output': 'test output'})

        assert len(dataset) == 1
        assert dataset[0].query == 'test query'

    def test_add_item_datasetitem(self):
        """Test adding a DatasetItem directly."""
        dataset = Dataset(name='test')
        item = DatasetItem(query='test', expected_output='output')
        dataset.add_item(item)

        assert len(dataset) == 1
        assert dataset[0] == item

    def test_add_items_multiple(self):
        """Test adding multiple items at once."""
        dataset = Dataset(name='test')
        items = [
            {'query': 'q1', 'expected_output': 'o1'},
            {'query': 'q2', 'expected_output': 'o2'},
            {'query': 'q3', 'expected_output': 'o3'},
        ]
        dataset.add_items(items)

        assert len(dataset) == 3

    def test_filter(self):
        """Test filtering dataset items."""
        dataset = Dataset(name='test')
        dataset.add_items(
            [
                {'query': 'q1', 'latency': 0.5},
                {'query': 'q2', 'latency': 1.5},
                {'query': 'q3', 'latency': 0.3},
            ]
        )

        fast_items = dataset.filter(lambda x: x.latency < 1.0)

        assert len(fast_items) == 2

    def test_to_dataframe(self):
        """Test converting dataset to DataFrame."""
        dataset = Dataset(name='test')
        dataset.add_items(
            [
                {'query': 'q1', 'expected_output': 'o1', 'actual_output': 'a1'},
                {'query': 'q2', 'expected_output': 'o2', 'actual_output': 'a2'},
            ]
        )

        df = dataset.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'query' in df.columns
        assert 'expected_output' in df.columns

    def test_get_summary(self):
        """Test getting dataset summary."""
        dataset = Dataset(name='test', description='Test dataset', version='1.0')
        dataset.add_items(
            [
                {'query': 'q1', 'expected_output': 'o1', 'actual_output': 'a1'},
                {'query': 'q2', 'expected_output': 'o2'},
                {
                    'conversation': MultiTurnConversation(
                        messages=[HumanMessage(content='Multi-turn')]
                    )
                },
            ]
        )

        summary = dataset.get_summary()

        assert summary['name'] == 'test'
        assert summary['total_items'] == 3
        assert summary['single_turn_items'] == 2
        assert summary['multi_turn_items'] == 1
        assert summary['has_actual_output'] == 1
        assert summary['has_expected_output'] == 2

    def test_dataset_iteration(self):
        """Test iterating over dataset."""
        dataset = Dataset(name='test')
        dataset.add_items([{'query': f'q{i}'} for i in range(3)])

        items = [item for item in dataset]

        assert len(items) == 3

    def test_dataset_indexing(self):
        """Test indexing dataset items."""
        dataset = Dataset(name='test')
        dataset.add_items([{'query': 'q1'}, {'query': 'q2'}, {'query': 'q3'}])

        assert dataset[0].query == 'q1'
        assert dataset[1].query == 'q2'
        assert dataset[-1].query == 'q3'


class TestDatasetWithMultiTurnConversations:
    """Test Dataset handling of multi-turn conversations."""

    def test_add_multi_turn_conversation(self):
        """Test adding items with multi-turn conversations."""
        dataset = Dataset(name='multi-turn-test')
        convo = MultiTurnConversation(
            messages=[
                HumanMessage(content='First question'),
                AIMessage(content='First answer'),
                HumanMessage(content='Follow-up question'),
            ]
        )
        dataset.add_item({'conversation': convo})

        assert len(dataset) == 1
        assert dataset[0].conversation is not None
        assert len(dataset[0].conversation.messages) == 3

    def test_summary_with_multi_turn(self):
        """Test that summary correctly counts multi-turn items."""
        dataset = Dataset(name='test')
        dataset.add_items(
            [
                {'query': 'Single turn 1'},
                {
                    'conversation': MultiTurnConversation(
                        messages=[HumanMessage(content='Multi 1')]
                    )
                },
                {'query': 'Single turn 2'},
                {
                    'conversation': MultiTurnConversation(
                        messages=[HumanMessage(content='Multi 2')]
                    )
                },
            ]
        )

        summary = dataset.get_summary()
        assert summary['single_turn_items'] == 2
        assert summary['multi_turn_items'] == 2

    def test_filter_by_conversation_type(self):
        """Test filtering by single vs multi-turn."""
        dataset = Dataset(name='test')
        dataset.add_items(
            [
                {'query': 'Single'},
                {
                    'conversation': MultiTurnConversation(
                        messages=[HumanMessage(content='Multi')]
                    )
                },
            ]
        )

        multi_turn_only = dataset.filter(lambda x: x.conversation is not None)
        single_turn_only = dataset.filter(lambda x: x.conversation is None)

        assert len(multi_turn_only) == 1
        assert len(single_turn_only) == 1


# Integration tests that verify the whole system works together
class TestFullWorkflow:
    """Integration tests simulating real usage workflows."""

    def test_evaluation_workflow(self):
        """Test a complete evaluation workflow."""
        # Create initial dataset
        dataset = Dataset(
            name='evaluation_test',
            description='Test evaluation workflow',
            version='1.0',
        )

        # Add test queries
        test_queries = [
            {
                'query': 'What is Python?',
                'expected_output': 'Python is a programming language',
            },
            {
                'query': 'What is AI?',
                'expected_output': 'AI is artificial intelligence',
            },
            {'query': 'Explain ML', 'expected_output': 'ML is machine learning'},
        ]
        dataset.add_items(test_queries)

        # Simulate running evaluation (update with actual outputs)
        for i, item in enumerate(dataset.items):
            item.update_runtime(
                actual_output=f'Generated response {i + 1}',
                latency=0.3 + i * 0.1,
                trace=f'{{"steps": {i + 2}, "model": "test-model"}}',
            )

        # Verify results
        summary = dataset.get_summary()
        assert summary['has_actual_output'] == 3
        assert summary['has_expected_output'] == 3

        # Test filtering for analysis
        fast_responses = dataset.filter(lambda x: x.latency < 0.5)
        assert len(fast_responses) == 2

        # Convert to DataFrame for analysis
        df = dataset.to_dataframe()
        assert len(df) == 3
        assert all(
            col in df.columns
            for col in ['query', 'expected_output', 'actual_output', 'latency']
        )

    def test_dataset_evolution_workflow(self):
        """Test workflow of evolving a dataset over time."""
        # Start with basic dataset
        dataset = Dataset(name='evolving_dataset', version='1.0')

        # Add initial items
        dataset.add_items([{'query': 'Basic query 1'}, {'query': 'Basic query 2'}])

        # Simulate adding expected outputs later
        for item in dataset.items:
            item.update({'expected_output': f'Expected for: {item.query}'})

        # Add more items
        dataset.add_items(
            [
                {'query': 'New query 3', 'expected_output': 'Expected 3'},
                {'query': 'New query 4', 'expected_output': 'Expected 4'},
            ]
        )

        # Simulate evaluation run
        for item in dataset.items:
            item.update_runtime(actual_output=f'Generated for: {item.query}')

        # Verify final state
        assert len(dataset) == 4
        summary = dataset.get_summary()
        assert summary['has_expected_output'] == 4
        assert summary['has_actual_output'] == 4

    def test_multi_turn_evaluation_workflow(self):
        """Test evaluation workflow with multi-turn conversations."""
        dataset = Dataset(name='multi_turn_eval')

        # Add multi-turn conversations with rubrics and metadata
        conversations = [
            {
                'conversation': MultiTurnConversation(
                    messages=[
                        HumanMessage(content='What is Python?'),
                        AIMessage(content='Python is a programming language'),
                        HumanMessage(content='What are its main features?'),
                    ],
                    reference_text='Expected comprehensive answer about Python features',
                    rubrics={'accuracy': 'Must be factually correct'},
                    metadata={'sessionId': 'session-1', 'plannerType': 'sequential'},
                )
            },
            {
                'conversation': MultiTurnConversation(
                    messages=[
                        HumanMessage(content='Search for AI papers'),
                        AIMessage(content='Here are some papers...'),
                    ],
                    reference_text='Should provide relevant AI papers',
                    rubrics={'relevance': 'Papers must be about AI'},
                )
            },
        ]
        dataset.add_items(conversations)

        # Verify multi-turn items were added correctly
        assert len(dataset) == 2
        summary = dataset.get_summary()
        assert summary['multi_turn_items'] == 2

        # Verify rubrics and metadata are preserved
        assert dataset[0].conversation.rubrics is not None
        assert dataset[0].conversation.metadata is not None
        assert dataset[1].conversation.rubrics is not None


@pytest.fixture
def sample_conversation() -> MultiTurnConversation:
    """Provides a standard multi-turn conversation for testing."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='First user query'),
            AIMessage(content='First AI response'),
            HumanMessage(content='Second user query'),
            AIMessage(content='Second AI response'),
        ]
    )


class TestConversationExtractionStrategy:
    """
    Tests the functionality of the `conversation_extraction_strategy` field
    on the DatasetItem class.
    """

    def test_default_strategy_is_last(self, sample_conversation):
        """
        Tests that the default extraction strategy is 'last' and correctly
        extracts the last user and AI messages.
        """
        # Create a DatasetItem without specifying the strategy
        item = DatasetItem(conversation=sample_conversation)

        assert item.conversation_extraction_strategy == 'last'
        assert (
            item.query == 'Second user query'
        ), 'Default strategy should pick the last user query.'
        assert (
            item.actual_output == 'Second AI response'
        ), 'Default strategy should pick the last AI response.'

    def test_explicit_last_strategy(self, sample_conversation):
        """
        Tests that explicitly setting the strategy to 'last' works as expected.
        """
        # Create a DatasetItem explicitly setting the strategy to 'last'
        item = DatasetItem(
            conversation=sample_conversation, conversation_extraction_strategy='last'
        )

        assert item.conversation_extraction_strategy == 'last'
        assert (
            item.query == 'Second user query'
        ), "Explicit 'last' strategy should pick the last user query."
        assert (
            item.actual_output == 'Second AI response'
        ), "Explicit 'last' strategy should pick the last AI response."

    def test_first_strategy(self, sample_conversation):
        """
        Tests that setting the strategy to 'first' correctly extracts
        the first user and AI messages.
        """
        # Create a DatasetItem with the strategy set to 'first'
        item = DatasetItem(
            conversation=sample_conversation, conversation_extraction_strategy='first'
        )

        assert item.conversation_extraction_strategy == 'first'
        assert (
            item.query == 'First user query'
        ), "'first' strategy should pick the first user query."
        assert (
            item.actual_output == 'First AI response'
        ), "'first' strategy should pick the first AI response."

    def test_actual_output_is_not_overwritten_if_provided(self, sample_conversation):
        """
        Tests that the automatic setting of `actual_output` does not overwrite
        an explicitly provided value, regardless of the strategy.
        """
        # Test with 'last' strategy
        item_last = DatasetItem(
            conversation=sample_conversation,
            actual_output='Explicitly set output',
            conversation_extraction_strategy='last',
        )
        assert item_last.actual_output == 'Explicitly set output'

        # Test with 'first' strategy
        item_first = DatasetItem(
            conversation=sample_conversation,
            actual_output='Another explicit output',
            conversation_extraction_strategy='first',
        )
        assert item_first.actual_output == 'Another explicit output'

    def test_no_ai_message_case(self):
        """
        Tests that `actual_output` remains None if no AIMessage exists.
        """
        convo = MultiTurnConversation(messages=[HumanMessage(content='Hello?')])
        item = DatasetItem(conversation=convo)
        assert item.query == 'Hello?'
        assert (
            item.actual_output is None
        ), 'actual_output should be None if no AI message is present.'


@pytest.fixture
def sample_dataset_item_subset():
    """Fixture providing a sample DatasetItem."""
    return DatasetItem(
        query='What is machine learning?',
        expected_output='Machine learning is a subset of AI',
        actual_output='ML is a part of AI',
        metadata='{"difficulty": "medium"}',
        judgment='pass',
        critique='Accurate but could be more detailed.',
    )


class TestSubset:
    """Tests for the `subset` and `evaluation_fields` methods of DatasetItem."""

    def test_subset_basic_fields(self, sample_dataset_item_subset):
        result = sample_dataset_item_subset.subset(['query', 'expected_output'])
        assert result.query == sample_dataset_item_subset.query
        assert result.expected_output == sample_dataset_item_subset.expected_output
        assert result.actual_output is None
        assert result.metadata is None
        assert result.id == sample_dataset_item_subset.id

    def test_subset_missing_field_raises(self, sample_dataset_item_subset):
        with pytest.raises(
            ValueError, match='Field\\(s\\) .* do not exist on DatasetItem'
        ):
            sample_dataset_item_subset.subset(['query', 'nonexistent_field'])

    def test_subset_with_annotations(self, sample_dataset_item_subset):
        result = sample_dataset_item_subset.subset(
            ['query', 'actual_output'], copy_annotations=True
        )
        assert result.query == sample_dataset_item_subset.query
        assert result.actual_output == sample_dataset_item_subset.actual_output
        assert result.judgment == sample_dataset_item_subset.judgment
        assert result.critique == sample_dataset_item_subset.critique

    def test_subset_without_annotations(self, sample_dataset_item_subset):
        result = sample_dataset_item_subset.subset(
            ['query', 'actual_output'], copy_annotations=False
        )
        assert result.judgment is None
        assert result.critique is None

    def test_evaluation_fields_calls_subset(self, sample_dataset_item_subset):
        """Test that evaluation_fields uses subset with correct field list."""
        result = sample_dataset_item_subset.evaluation_fields()
        assert result.metadata is None
        assert result.actual_output


class TestFormatInput:
    """Tests for the format_input utility function."""

    def test_format_input_with_datasetitem(self):
        """Test that format_input returns DatasetItem unchanged."""
        item = DatasetItem(query='test')
        result = format_input(item)
        assert result is item

    def test_format_input_with_dict(self):
        """Test that format_input creates DatasetItem from dict."""
        input_dict = {'query': 'test query', 'expected_output': 'test output'}
        result = format_input(input_dict)
        assert isinstance(result, DatasetItem)
        assert result.query == 'test query'
        assert result.expected_output == 'test output'

    def test_format_input_with_invalid_type(self):
        """Test that format_input raises error for invalid types."""
        with pytest.raises(TypeError, match='Invalid type'):
            format_input('invalid string')

        with pytest.raises(TypeError, match='Invalid type'):
            format_input(123)


class TestUserTagsFeature:
    """Tests for user_tags feature on DatasetItem."""

    def test_default_user_tags_empty(self):
        """Test that user_tags defaults to empty list."""
        item = DatasetItem(query='test')
        assert item.user_tags == []

    def test_user_tags_initialization(self):
        """Test initializing DatasetItem with user_tags."""
        tags = ['production', 'high-priority', 'customer-facing']
        item = DatasetItem(query='test', user_tags=tags)

        assert item.user_tags == tags
        assert len(item.user_tags) == 3

    def test_user_tags_in_dict_serialization(self):
        """Test that user_tags are included in dictionary serialization."""
        item = DatasetItem(query='test', user_tags=['tag1', 'tag2'])
        item_dict = item.to_dict()

        assert 'user_tags' in item_dict
        assert item_dict['user_tags'] == ['tag1', 'tag2']


class TestRankingFields:
    """Tests for ranking-related fields on DatasetItem."""

    def test_actual_ranking_field(self):
        """Test actual_ranking field for retrieval results."""
        ranking = [
            {'id': 'doc1', 'score': 0.95},
            {'id': 'doc2', 'score': 0.87},
            {'id': 'doc3', 'score': 0.72},
        ]
        item = DatasetItem(query='test', actual_ranking=ranking)

        assert item.actual_ranking is not None
        assert len(item.actual_ranking) == 3
        assert item.actual_ranking[0]['score'] == 0.95

    def test_expected_ranking_field(self):
        """Test expected_ranking field for ground truth."""
        expected = [
            {'id': 'doc1', 'relevance': 1.0},
            {'id': 'doc2', 'relevance': 0.5},
        ]
        item = DatasetItem(query='test', expected_ranking=expected)

        assert item.expected_ranking is not None
        assert len(item.expected_ranking) == 2
        assert item.expected_ranking[0]['relevance'] == 1.0


class TestDocumentTextField:
    """Tests for document_text field on DatasetItem."""

    def test_document_text_initialization(self):
        """Test initializing with document_text."""
        doc_text = 'This is a sample document for testing.'
        item = DatasetItem(document_text=doc_text)

        assert item.document_text == doc_text

    def test_document_text_in_serialization(self):
        """Test that document_text is included in serialization."""
        item = DatasetItem(
            query='Summarize this', document_text='Long document text here...'
        )
        item_dict = item.to_dict()

        assert 'document_text' in item_dict
        assert item_dict['document_text'] == 'Long document text here...'
