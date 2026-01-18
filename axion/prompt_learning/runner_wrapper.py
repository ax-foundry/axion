"""
Runner wrapper for dynamic prompt injection during evaluation.

This module provides utilities to wrap user task functions and inject
the current optimized prompt dynamically during each evaluation iteration.
"""

from typing import Any, Callable, Dict, Optional

from axion.dataset import DatasetItem


class DynamicPromptTask:
    """
    Wraps a user's task function to inject the optimized prompt dynamically.

    The wrapper intercepts calls to the task function and injects the current
    prompt via a specified keyword argument (default: 'system_prompt').

    Example usage:
        ```python
        async def my_task(item: DatasetItem, system_prompt: str) -> dict:
            # Use system_prompt to generate response
            return {'actual_output': response}

        wrapped_task = DynamicPromptTask(my_task)
        wrapped_task.set_prompt("You are a helpful assistant.")

        # Now when evaluation calls wrapped_task(item),
        # it will call my_task(item, system_prompt="You are a helpful assistant.")
        ```
    """

    def __init__(
        self,
        base_task: Callable,
        prompt_key: str = 'system_prompt',
    ):
        """
        Initialize the dynamic prompt wrapper.

        Args:
            base_task: The original task function that takes (item, system_prompt).
            prompt_key: The keyword argument name for the prompt (default: 'system_prompt').
        """
        self.base_task = base_task
        self.prompt_key = prompt_key
        self._current_prompt: str = ''

        # Preserve function metadata for introspection
        self.__name__ = getattr(base_task, '__name__', 'wrapped_task')
        self.__doc__ = getattr(base_task, '__doc__', None)

    def set_prompt(self, prompt: str) -> None:
        """
        Set the current prompt to be injected into task calls.

        Args:
            prompt: The prompt string to inject.
        """
        self._current_prompt = prompt

    @property
    def current_prompt(self) -> str:
        """Get the currently set prompt."""
        return self._current_prompt

    async def __call__(self, item: DatasetItem) -> Dict[str, Any]:
        """
        Execute the wrapped task with the current prompt injected.

        Args:
            item: The DatasetItem to process.

        Returns:
            The task output (typically containing 'actual_output').
        """
        # Inject the prompt via keyword argument
        kwargs = {self.prompt_key: self._current_prompt}
        return await self.base_task(item, **kwargs)


class PromptTemplateTask:
    """
    Alternative wrapper that uses a prompt template with variable substitution.

    This allows the prompt to contain placeholders like {query} that get
    filled from the DatasetItem.

    Example:
        ```python
        template = "You are helping answer questions about {topic}. Query: {query}"
        task = PromptTemplateTask(llm_call_fn, template)
        task.set_prompt(template)  # Sets the template
        # When called, {query} is replaced with item.query
        ```
    """

    def __init__(
        self,
        base_task: Callable,
        default_template: Optional[str] = None,
    ):
        """
        Initialize the template-based task wrapper.

        Args:
            base_task: The original task function that takes (item, prompt).
            default_template: Optional default prompt template.
        """
        self.base_task = base_task
        self._template: str = default_template or ''

        # Preserve function metadata
        self.__name__ = getattr(base_task, '__name__', 'template_task')
        self.__doc__ = getattr(base_task, '__doc__', None)

    def set_prompt(self, template: str) -> None:
        """
        Set the prompt template.

        Args:
            template: The prompt template (may contain {placeholders}).
        """
        self._template = template

    @property
    def current_prompt(self) -> str:
        """Get the current prompt template."""
        return self._template

    def _render_template(self, item: DatasetItem) -> str:
        """
        Render the template with values from the DatasetItem.

        Args:
            item: The DatasetItem to extract values from.

        Returns:
            The rendered prompt string.
        """
        # Build context from DatasetItem attributes
        context = {
            'query': getattr(item, 'query', ''),
            'expected_output': getattr(item, 'expected_output', ''),
            'retrieved_content': getattr(item, 'retrieved_content', ''),
            'id': getattr(item, 'id', ''),
        }

        # Add any metadata fields
        metadata = getattr(item, 'metadata', {}) or {}
        context.update(metadata)

        # Render template with safe substitution (missing keys left as-is)
        try:
            return self._template.format(**context)
        except KeyError:
            # Fallback: use string Template for partial substitution
            from string import Template

            return Template(self._template).safe_substitute(context)

    async def __call__(self, item: DatasetItem) -> Dict[str, Any]:
        """
        Execute the task with the rendered prompt.

        Args:
            item: The DatasetItem to process.

        Returns:
            The task output.
        """
        rendered_prompt = self._render_template(item)
        return await self.base_task(item, system_prompt=rendered_prompt)


def wrap_task_for_optimization(
    task: Callable,
    prompt_key: str = 'system_prompt',
) -> DynamicPromptTask:
    """
    Wrap a task function for use with prompt optimization.

    This is a convenience function that creates a DynamicPromptTask wrapper.

    Args:
        task: The task function to wrap. Should have signature:
              async def task(item: DatasetItem, system_prompt: str) -> dict
        prompt_key: The keyword argument name for the prompt.

    Returns:
        A DynamicPromptTask wrapper that can be used with EvaluationRunner.
    """
    return DynamicPromptTask(task, prompt_key=prompt_key)
