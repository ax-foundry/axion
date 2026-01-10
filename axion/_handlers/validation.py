from typing import Any

from axion._core.error import InvalidConfig
from pydantic import BaseModel


class Validation:
    @staticmethod
    def validate_required_fields(obj, required_fields: list) -> None:
        """
        Validate that required configuration fields are provided.

        Args:
            obj (obj): Class object
            required_fields (list): List of attribute names that must be defined.
        """
        missing_fields = [
            field for field in required_fields if not getattr(obj, field, None)
        ]
        if missing_fields:
            raise InvalidConfig(f"Missing required fields: {', '.join(missing_fields)}")

    @staticmethod
    def validate_llm_model(llm):
        # Validate llm model interface
        if llm is not None:
            if not any(
                callable(getattr(llm, method, None))
                for method in ('complete', 'acomplete')
            ):
                raise InvalidConfig('LLM model must have a callable "acomplete" method')

    @staticmethod
    def validate_embed_model(embed_model):
        # Validate embed model interface
        if embed_model is not None:
            if not any(
                callable(getattr(embed_model, method, None))
                for method in ('get_text_embedding', 'aget_text_embedding')
            ):
                raise InvalidConfig(
                    'Embed model must have a callable "aget_text_embedding" method'
                )

    @staticmethod
    def validate_io_models(input_model: Any, output_model: Any):
        for name, model in [
            ('input_model', input_model),
            ('output_model', output_model),
        ]:
            if model and not (isinstance(model, type) and issubclass(model, BaseModel)):
                raise InvalidConfig(f"'{name}' must be a Pydantic BaseModel class")
