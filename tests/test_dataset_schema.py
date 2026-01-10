import pytest
from axion._core.error import CustomValidationError
from axion._core.schema import AIMessage, HumanMessage, ToolMessage
from axion.dataset_schema import (
    MultiTurnConversation,
    RichDatasetBaseModel,
)


def test_valid_multiturn_conversation():
    messages = [
        HumanMessage(content='Hello'),
        AIMessage(content='Hi there!'),
        ToolMessage(
            tool_call_id='call_123',
            content='{"temperature": 65, "condition": "Cloudy"}',
        ),
    ]
    conversation = MultiTurnConversation(messages=messages)
    assert len(conversation.messages) == 3
    assert isinstance(conversation.messages[0], HumanMessage)


def test_invalid_multiturn_conversation_empty():
    with pytest.raises(CustomValidationError, match='Messages list cannot be empty'):
        MultiTurnConversation(messages=[])


def test_valid_multiturn_conversation_start_with_ai():
    messages = [AIMessage(content='I go first!')]
    MultiTurnConversation(messages=messages)


def test_rich_dataset_repr_str_single_turn():
    class DummyItem(RichDatasetBaseModel):
        id: str = 'abc123456789'
        query: str = 'What is AI?'
        expected_output: str = 'Artificial Intelligence'
        actual_output: str = 'AI stands for Artificial Intelligence.'

    item = DummyItem()
    out = repr(item)
    assert 'expected_output=' in out
    assert 'actual_output=' in out
    assert 'query=' in out


def test_rich_dataset_repr_str_multi_turn():
    class DummyItem(RichDatasetBaseModel):
        id: str = 'abc123456789'
        conversation: MultiTurnConversation
        expected_output: str = 'Final answer'
        actual_output: str = 'Generated answer'

    conversation = MultiTurnConversation(
        messages=[
            HumanMessage(content='User input'),
            AIMessage(content='Assistant response'),
        ]
    )
    item = DummyItem(conversation=conversation)
    out = repr(item)
    assert 'conversation=(2 messages' in out
    assert 'expected_output=' in out
    assert 'actual_output=' in out


def test_str_single_turn_item():
    class DummyItem(RichDatasetBaseModel):
        id: str = 'shortid'
        query: str = 'Explain gravity'
        expected_output: str = 'A force that attracts...'
        actual_output: str = 'Gravity pulls...'

    item = DummyItem()
    out = str(item)
    assert 'Single-turn DatasetItem' in out
    assert 'Query: Explain gravity' in out
    assert 'Expected Output: A force that attracts' in out
    assert 'Actual Output: Gravity pulls' in out


def test_str_multi_turn_item():
    class DummyItem(RichDatasetBaseModel):
        id: str = 'abc123456789'
        conversation: MultiTurnConversation
        expected_output: str = 'Final explanation'
        actual_output: str = 'Model response'

    conversation = MultiTurnConversation(
        messages=[
            HumanMessage(content='How does a plane fly?'),
            AIMessage(content='By generating lift...'),
        ]
    )
    item = DummyItem(conversation=conversation)
    out = str(item)
    assert 'Multi-turn DatasetItem' in out
    assert 'Query: How does a plane fly?' in out
    assert 'Expected Output: Final explanation' in out
    assert 'Actual Output: Model response' in out
