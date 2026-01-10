from pydantic import BaseModel


class PromptSection(BaseModel):
    name: str
    content: str
    priority: int
