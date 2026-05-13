from .models import FunctionDefinition


def build_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    instructions: str = (
        f'Given these functions: {functions} For the request: {prompt} '
        'Reply with only the function name. Function: "'
    )
    return instructions
