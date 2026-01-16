from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from axion._core.parsers.parse.parser import AIOutputParser


class MockOutputModel(BaseModel):
    field: str


class MockLLM(BaseModel):
    pass


@pytest.fixture
def tracer():
    return MagicMock()


@pytest.fixture
def ai_parser(tracer):
    return AIOutputParser(llm=MockLLM, output_model=MockOutputModel, tracer=tracer)


@pytest.mark.asyncio
async def test_validate_output_success(ai_parser):
    json_str = '{"field": "valid"}'
    result = await ai_parser._validate_output(json_str)
    assert result.field == 'valid'


@pytest.mark.asyncio
async def test_validate_output_failure(ai_parser):
    json_str = '{"wrong_field": "value"}'

    with pytest.raises(ValidationError):
        await ai_parser._validate_output(json_str)
