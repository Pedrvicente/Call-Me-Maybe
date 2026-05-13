from .models import FunctionDefinition


def build_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build an instruction string that primes the model to output a function name.

    Args:
        prompt: The user request.
        functions: The list of candidate function definitions to include as context.

    Returns:
        A formatted instruction string ending with an open quote to prime generation.
    """
    instructions: str = (
        f'Given these functions: {functions} For the request: {prompt} '
        'Reply with only the function name. Function: "'
    )
    return instructions
