from .prompt import build_prompt
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition, PromptRequest, OutputRequest
import json
from typing import Any

def get_vocab(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, 'r') as f:
        # formato token -> id
        vocab = json.load(f)
    return {token_id: model.decode([token_id]) for token_id in vocab.values()} # formato id -> text

def select_function(prompt: PromptRequest, function: list[FunctionDefinition], model: Small_LLM_Model) -> str:
    id_to_token = get_vocab(model)
    function_names = [f'{f.name}"'for f in function]
    written_function = ""
    result = build_prompt(prompt, function)
    while True:
        tokens = model.encode(result)
        values = tokens[0].tolist()
        logits = model.get_logits_from_input_ids(values)
        for token_id in range(len(logits)):
            if token_id in id_to_token:
                chars = id_to_token[token_id]
                if not any(name.startswith(written_function + chars) for name in function_names):
                    logits[token_id] = float('-inf')
        next_token = max(logits)
        next_token_id = logits.index(next_token)
        if next_token_id not in id_to_token:
            break
        char = id_to_token[next_token_id]
        result += char
        written_function += char
        print(f"written: '{written_function}'")
        if written_function.endswith('"'):
            return written_function[:-1]
    return ""

def extract_numbers(prompt: str) -> list[float]:
    lst_floats = []
    for word in prompt.split():
        try:
            word = word.strip('.,!?;:')
            digit = float(word)
            lst_floats.append(digit)
        except ValueError:
            pass
    return lst_floats

def extract_str(prompt: str, param_name: str, description: str, model: Small_LLM_Model) -> str:
    new_prompt = f'Extract the {param_name} from: <input>{prompt}</input> Context: {description}\n{param_name} = "'
    id_to_token = get_vocab(model)
    result = ""
    for _ in range(80):
        tokens = model.encode(new_prompt)[0].tolist()
        logits = model.get_logits_from_input_ids(tokens)
        for token_id in range(len(logits)):
            if token_id in id_to_token:
                chars = id_to_token[token_id]
                if '"' in chars and not chars.endswith('"'):
                    logits[token_id] = float('-inf')
        next_token = max(logits)
        next_token_id = logits.index(next_token)
        if next_token_id not in id_to_token:
            break
        new_char = id_to_token[next_token_id]
        result += new_char
        new_prompt += new_char
        print(f"char: '{new_char}'")
        if '"' in new_char:
            print(repr(result))
            return result.strip('",\n')
    return result.strip('",\n')


def extract_parameters(prompt: str, function: FunctionDefinition, model: Small_LLM_Model) -> dict[str, Any]:
    result = {}
    numbers = extract_numbers(prompt)
    param_description = function.description
    print(f"prompt: {prompt}, numbers: {numbers}, description: {param_description}")
    for param_name, param in function.parameters.items():
        param_type = param.type
        print(f"param_name: {param_name}, param_type: {param_type}, numbers restantes: {numbers}")
        if param_type == 'number':
            result[param_name] = numbers.pop(0)
        if param_type == 'string':
            result[param_name] = extract_str(prompt, param_name, param_description, model)
    return result



        



