from .models import FunctionDefinition, Parameter, PromptRequest, OutputRequest
import json
from pydantic import ValidationError

def load_functions(path: str) -> list[FunctionDefinition]:
    function_list: list[FunctionDefinition] = [] 
    try:
        with open(path, 'r') as f:
            functions = json.load(f)
            for i in functions:
                function_list.append(FunctionDefinition.model_validate(i))
    except FileNotFoundError as e:
        print(e)
    except json.JSONDecodeError as e:
        print(e)
    except ValidationError as e:
        print(e)
    return function_list

def load_prompts(path: str) -> list[PromptRequest]:
    prompts_list: list[PromptRequest] = [] 
    try:
        with open(path, 'r') as f:
            prompts = json.load(f)
            for i in prompts:
                prompts_list.append(PromptRequest.model_validate(i))
    except FileNotFoundError as e:
        print(e)
    except json.JSONDecodeError as e:
        print(e)
    except ValidationError as e:
        print(e)
    return prompts_list

def save_outputs(path: str, outputs: list[OutputRequest]) -> None:
    new_list = []
    new_list = [i.model_dump() for i in outputs]
    try:
        with open(path, 'w') as f:
            json.dump(new_list, f, indent=2)
    except FileNotFoundError as e:
        print(e)
    except PermissionError as e:
        print(e)
