from pydantic import Field, BaseModel
from typing import Any

class Parameter(BaseModel):
    type: str

class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)

class OutputRequest(BaseModel):
    prompt: str
    name: str = Field(min_length=1)
    parameters: dict[str, Any]

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter