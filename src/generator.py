from .prompt import build_prompt
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition, PromptRequest, OutputRequest
import json
from typing import Any

def select_function(prompt: PromptRequest, function: list[FunctionDefinition], model: Small_LLM_Model) -> FunctionDefinition:
    vocab_path = model.get_path_to_vocab_file()
    function_names = [f'{f.name}"'for f in function]
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}
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


def extract_parameters(prompt: str, function: FunctionDefinition) -> dict[str, Any]:
    param_names = []
    for param_name, param in function.parameters.items():
        param_names.append(param_name)
        param_type = param.type
        if param_type == 'number':
            values = extract_numbers(prompt)
    result = dict(zip(param_names, values))
    return result

if __name__ == '__main__':
    func = FunctionDefinition(name='fn_add_numbers', description='Add two numbers together and return their sum.', parameters={'a': Parameter(type='number'), 'b': Parameter(type='number')}, returns=Parameter(type='number'))
    promp = "What is the sum of 2 and 3?"
    extract_parameters(promp, func)



        



