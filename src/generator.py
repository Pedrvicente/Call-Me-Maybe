from .prompt import build_prompt
from llm_sdk import Small_LLM_Model
import json

def select_function(prompt: str, function: list[dict], model: Small_LLM_Model) -> str:
    vocab_path = model.get_path_to_vocab_file()
    function_names = [f['name'] for f in function]
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
        for i in function_names:
            if written_function == i:
                return written_function


        



