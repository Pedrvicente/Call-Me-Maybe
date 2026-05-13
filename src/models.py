from pydantic import Field, BaseModel
from typing import Any


class Parameter(BaseModel):
    """Represent a typed parameter or return value in a function definition."""

    type: str


class PromptRequest(BaseModel):
    """Represent a single user prompt to be processed by the pipeline."""

    prompt: str = Field(min_length=1)


class OutputRequest(BaseModel):
    """Represent the result of function-calling inference for a single prompt."""

    prompt: str
    name: str = Field(min_length=1)
    parameters: dict[str, Any]


class FunctionDefinition(BaseModel):
    """Represent a callable function with its signature and description."""

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
