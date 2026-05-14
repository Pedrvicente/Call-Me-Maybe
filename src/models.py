from pydantic import Field, BaseModel, ConfigDict
from typing import Any, Literal


class Parameter(BaseModel):
    """Represent a typed parameter or return value in a function definition."""
    model_config = ConfigDict(extra='forbid')
    type: Literal["string", "integer", "number", "boolean"]


class PromptRequest(BaseModel):
    """Represent a single user prompt to be processed by the pipeline."""
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1)


class OutputRequest(BaseModel):
    """Represent the result of function-calling inference for a single prompt."""
    model_config = ConfigDict(extra='forbid')
    prompt: str
    name: str = Field(min_length=1)
    parameters: dict[str, Any]


class FunctionDefinition(BaseModel):
    """Represent a callable function with its signature and description."""
    model_config = ConfigDict(extra='forbid')
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
