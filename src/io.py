import sys
from .models import FunctionDefinition, PromptRequest, OutputRequest
import json
from pydantic import ValidationError


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the JSON file containing an array of function definitions.

    Returns:
        A list of validated FunctionDefinition instances.

    Raises:
        SystemExit: If the file is not found, contains invalid JSON, or fails schema validation.
    """
    function_list: list[FunctionDefinition] = []
    try:
        with open(path, 'r') as f:
            functions = json.load(f)
            for i in functions:
                function_list.append(FunctionDefinition.model_validate(i))
    except FileNotFoundError:
        print(f"Error: file {path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in '{path}': {e}")
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: schema mismatch in '{path}': {e}")
        sys.exit(1)
    return function_list


def load_prompts(path: str) -> list[PromptRequest]:
    """Load and validate prompt requests from a JSON file.

    Args:
        path: Path to the JSON file containing an array of prompt objects.

    Returns:
        A list of validated PromptRequest instances.

    Raises:
        SystemExit: If the file is not found, contains invalid JSON, or fails schema validation.
    """
    prompts_list: list[PromptRequest] = []
    try:
        with open(path, 'r') as f:
            prompts = json.load(f)
            for i in prompts:
                prompts_list.append(PromptRequest.model_validate(i))
    except FileNotFoundError:
        print(f"Error: file {path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in '{path}': {e}")
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: schema mismatch in '{path}': {e}")
        sys.exit(1)
    return prompts_list


def save_outputs(path: str, outputs: list[OutputRequest]) -> None:
    """Serialize and write output requests to a JSON file.

    Args:
        path: Destination file path for the JSON output.
        outputs: List of OutputRequest instances to serialize.

    Raises:
        SystemExit: If the file path is not found or write permission is denied.
    """
    new_list = [i.model_dump() for i in outputs]
    try:
        with open(path, 'w') as f:
            json.dump(new_list, f, indent=2)
    except FileNotFoundError:
        print(f"Error: file {path} not found")
        sys.exit(1)
    except PermissionError:
        print(f"Error: permission denied writing to '{path}'")
        sys.exit(1)
