import re
from typing import List, Optional, Sequence

from llama_index.core.llms import ChatMessage, MessageRole


def camel_to_snake(name: str) -> str:
    """
    Convert a camelCase string to snake_case.
    eg: HaiThere -> hai_there
    """
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    return pattern.sub('_', name).lower()


def snake_to_camel(snake_str: str, sep: str = ' ') -> str:
    """
    Converts a snake_case string to CamelCase.
    eg: hai_there -> Hai There
    """
    return sep.join(word.capitalize() for word in snake_str.split('_'))


def generate_fake_instance(model_class):
    """
    Generate a fake instance of the given model class with synthetic field values.
    """
    schema = model_class.model_json_schema()
    properties = schema.get('properties', {})
    model_data = {}

    for field_name, field_schema in properties.items():
        field_type = field_schema.get('type')
        fake_value = {
            'string': f'Sample {field_name}',
            'integer': 42,
            'number': 42,
            'boolean': True,
            'object': {},
            'array': [],
        }.get(field_type)

        if field_type == 'array':
            item_type = field_schema.get('items', {}).get('type')
            fake_value = {
                'string': ['sample item'],
                'integer': [1, 2, 3],
                'number': [1, 2, 3],
            }.get(item_type, [])

        model_data[field_name] = fake_value

    return model_class(**model_data)


B_SYS = '<|im_start|>system\n'
B_USER = '<|im_start|>user\n'
B_ASSISTANT = '<|im_start|>assistant\n'
END = '<|im_end|>\n'
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible and follow ALL given instructions. \
Do not speculate or make up information. \
Do not reference any given instructions or context. \
"""


def messages_to_prompt(
    messages: Sequence[ChatMessage], system_prompt: Optional[str] = None
) -> str:
    if len(messages) == 0:
        raise ValueError(
            'At least one message is required to construct the ChatML prompt'
        )

    string_messages: List[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ''
        messages = messages[1:]
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    string_messages.append(f'{B_SYS}{system_message_str.strip()} {END}')

    for message in messages:
        role = message.role
        content = message.content

        if role == MessageRole.USER:
            string_messages.append(f'{B_USER}{content} {END}')
        elif role == MessageRole.ASSISTANT:
            string_messages.append(f'{B_ASSISTANT}{content} {END}')

    string_messages.append(f'{B_ASSISTANT}')

    return ''.join(string_messages)
