from .prompt import build_prompt
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition
import json
from typing import Any

def get_vocab(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, 'r') as f:
        # formato token -> id
        vocab = json.load(f)
    return {token_id: model.decode([token_id]) for token_id in vocab.values()} # formato id -> text

def select_function(prompt: str, function: list[FunctionDefinition], model: Small_LLM_Model) -> str:
    id_to_token = get_vocab(model)
    function_names = [f'{f.name}"'for f in function]
    written_function = ""
    result = build_prompt(prompt, function)
    for _ in range(20):
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

def extract_number(prompt: str, param_name: str, description: str, extracted: dict[str, Any], model: Small_LLM_Model) -> float:
    context = ""
    if extracted:
        pairs = ', '.join(f"{k}={v}" for k, v in extracted.items())
        context = f"Already extracted: {pairs}. Do not repeate these values"
    new_prompt = f'Extract the literal value of {param_name} from the input. Do not compute. Context: {context}. Extract the {param_name} from: <input>{prompt}</input> Context: {description}\n{param_name} = '
    id_to_token = get_vocab(model)
    result = ""
    for _ in range(80):
        tokens = model.encode(new_prompt)[0].tolist()
        logits = model.get_logits_from_input_ids(tokens)
        original_token = max(logits)
        original_token_id = logits.index(original_token)
        original_char = id_to_token[original_token_id]
        if any(c not in '0123456789.' for c in original_char):
            break
        for token_id in range(len(logits)):
            if token_id in id_to_token:
                chars = id_to_token[token_id]
                if any(c not in '0123456789.' for c in chars):
                    logits[token_id] = float('-inf')
                    continue
                if (result + chars) not in prompt:
                    logits[token_id] = float('-inf')
        next_token = max(logits)
        next_token_id = logits.index(next_token)
        if next_token_id not in id_to_token:
            break
        new_char = id_to_token[next_token_id]
        result += new_char
        new_prompt += new_char
        print(f"char: '{new_char}'")
    try:
        return float(result)
    except Exception as e:
        print(e)
        return 0.0

def extract_str(prompt: str, param_name: str, description: str, extracted: dict[str, Any], model: Small_LLM_Model) -> str:
    context = ""
    if extracted:
        pairs = ', '.join(f"{k}={v}" for k, v in extracted.items())
        context = f"Already extracted: {pairs}. Do not repeat these values"
    new_prompt = f'Extract the {param_name} from: <input>{prompt}</input> Context: {description} {context}\n{param_name} = "'
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
    param_description = function.description
    for param_name, param in function.parameters.items():
        param_type = param.type
        if param_type == 'number':
            result[param_name] = extract_number(prompt, param_name, param_description, result, model)
        elif param_type == 'string':
            result[param_name] = extract_str(prompt, param_name, param_description, result, model)
    return result
