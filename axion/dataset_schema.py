from typing import Annotated, Any, Dict, List, Optional, Union

from axion._core.schema import (
    AIMessage,
    HumanMessage,
    RichBaseModel,
    ToolCall,
    ToolMessage,
)
from axion._core.types import FieldNames
from pydantic import Field, field_validator

Conversation = Annotated[
    Union[HumanMessage, AIMessage, ToolMessage], Field(discriminator='role')
]


class MultiTurnConversation(RichBaseModel):
    """Represents a full multi-turn evaluation conversation, containing the conversation history and ground truth."""

    messages: List[Conversation]
    reference_text: Optional[str] = Field(
        default=None, description='The final expected text response from the AI.'
    )
    reference_tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description='The expected tool calls the AI should make.'
    )
    rubrics: Optional[Dict[str, str]] = None

    # This maps to `sessionProperties`, `plannerType`, etc.
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Session-level metadata (e.g., plannerType, sessionProperties).',
    )

    @field_validator('messages')
    @classmethod
    def validate_conversation_flow(cls, messages):
        if not messages:
            raise ValueError('Messages list cannot be empty.')
        return messages


class RichDatasetBaseModel(RichBaseModel):
    """A base model that provides a cleaner string representation for dataset items."""

    def __repr__(self) -> str:
        """Overrides the default repr to provide a cleaner, user-friendly output."""
        return f"{self.__class__.__name__}({self.__repr_str__(', ')})"

    def __repr_str__(self, join_str: str) -> str:
        """Custom string representation for cleaner output."""
        is_multi_turn = hasattr(self, 'conversation') and self.conversation is not None
        return (
            self._format_multi_turn_repr(join_str)
            if is_multi_turn
            else self._format_single_turn_repr(join_str)
        )

    def _format_multi_turn_repr(self, join_str: str) -> str:
        """Format representation for multi-turn conversation items."""
        parts = [f'id={self.id}']

        # Add conversation summary
        if self.conversation and self.conversation.messages:
            messages = self.conversation.messages
            counts = {
                'user': sum(
                    1 for msg in messages if getattr(msg, 'role', None) == 'user'
                ),
                'assistant': sum(
                    1 for msg in messages if getattr(msg, 'role', None) == 'assistant'
                ),
                'tool': sum(
                    1 for msg in messages if getattr(msg, 'role', None) == 'tool'
                ),
            }

            summary_parts = [f"{counts['user']} user"]
            if counts['assistant'] > 0:
                summary_parts.append(f"{counts['assistant']} AI")
            if counts['tool'] > 0:
                summary_parts.append(f"{counts['tool']} tool")

            parts.append(
                f"conversation=({len(messages)} messages: {', '.join(summary_parts)})"
            )

        # Add outputs
        if getattr(self, 'expected_output', None):
            parts.append(f"expected_output='{self.expected_output}'")
        if getattr(self, 'actual_output', None):
            parts.append(f"actual_output='{self.actual_output}'")

        return join_str.join(parts)

    def _format_single_turn_repr(self, join_str: str) -> str:
        """Format representation for single-turn items."""
        parts = [f'id={self.id}']

        # Add query and outputs
        for attr in FieldNames.get_display_fields():
            value = getattr(self, attr, None)
            if value:
                parts.append(f"{attr}='{value}'")

        return join_str.join(parts)

    def __str__(self) -> str:
        """Provide a detailed, readable string for print() statements."""
        is_multi_turn = hasattr(self, 'conversation') and self.conversation is not None
        return (
            self._format_multi_turn_str()
            if is_multi_turn
            else self._format_single_turn_str()
        )

    def _format_multi_turn_str(self) -> str:
        """Format a detailed string for multi-turn conversation items."""
        lines = [f'Multi-turn DatasetItem ({self.id})']

        # Conversation details
        if self.conversation and self.conversation.messages:
            lines.append(
                f'  {len(self.conversation.messages)} messages in conversation'
            )

            # Show user messages
            user_messages = [
                msg
                for msg in self.conversation.messages
                if getattr(msg, 'role', None) == 'user'
            ]
            for i, user_msg in enumerate(user_messages, 1):
                label = 'Query' if len(user_messages) == 1 else f'Message {i}'
                lines.append(f'  {label}: {user_msg.content}')

        # Add outputs
        self._add_output_lines(lines)
        return '\n'.join(lines)

    def _format_single_turn_str(self) -> str:
        """Format a detailed string for single-turn items."""
        lines = [f'Single-turn DatasetItem ({self.id})']

        # Add query
        if getattr(self, 'query', None):
            lines.append(f'  Query: {self.query}')

        # Add outputs
        self._add_output_lines(lines)
        return '\n'.join(lines)

    def _add_output_lines(self, lines: list) -> None:
        """Add expected, actual, and retrieved output lines to the given list."""
        outputs = [
            (
                'Expected Output',
                getattr(self, 'expected_output', None),
                '[Not provided]',
            ),
            (
                'Actual Output',
                getattr(self, 'actual_output', None),
                '[Not generated yet]',
            ),
            ('Retrieved Content', getattr(self, 'retrieved_content', None), None),
        ]

        # Labels to hide when value is None
        hide_when_none = {'Retrieved Content', 'Expected Output'}

        for label, value, placeholder in outputs:
            if value is None and label in hide_when_none:
                continue
            lines.append(f"  {label}: {value if value else placeholder or ''}".rstrip())
