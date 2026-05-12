import sys
from .models import FunctionDefinition, PromptRequest, OutputRequest
import json
from pydantic import ValidationError

def load_functions(path: str) -> list[FunctionDefinition]:
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
