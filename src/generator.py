from .prompt import build_prompt
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition
import json
from typing import Any


def get_vocab(model: Small_LLM_Model) -> dict[int, str]:
    """Build a token-id-to-decoded-text mapping from the model's vocabulary file.

    Args:
        model: The language model whose vocabulary file will be read.

    Returns:
        A dict mapping each token ID to its decoded string representation.
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, 'r') as f:
        #  formato token -> id
        vocab = json.load(f)
    return {token_id: model.decode([token_id]) for token_id in vocab.values()}  # formato id -> text


def select_function(
    prompt: str, function: list[FunctionDefinition], model: Small_LLM_Model
) -> str:
    """Select the best matching function name for a prompt using constrained decoding.

    Tokens that would produce a string that is not a valid prefix of any known function
    name are masked to -inf so the model is forced to generate a valid name.

    Args:
        prompt: The natural-language user request.
        function: The list of candidate function definitions.
        model: The language model used for token scoring.

    Returns:
        The selected function name, or an empty string if none could be determined.
    """
    id_to_token = get_vocab(model)
    function_names = [f'{f.name}"' for f in function]
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
        if written_function.endswith('"'):
            return written_function[:-1]
    return ""


def extract_number(
    prompt: str, param_name: str, description: str,
    extracted: dict[str, Any], model: Small_LLM_Model
) -> float:
    """Extract a numeric parameter value from a prompt using constrained decoding.

    Only tokens that form digit or period characters present in the original prompt
    are allowed, preventing the model from hallucinating values.

    Args:
        prompt: The user request.
        param_name: The name of the parameter to extract.
        description: Human-readable description providing extraction context.
        extracted: Parameters already extracted in this call (used to avoid repetition).
        model: The language model used for token scoring.

    Returns:
        The extracted float value, or 0.0 if extraction fails.
    """
    context = ""
    if extracted:
        pairs = ', '.join(f"{k}={v}" for k, v in extracted.items())
        context = f"Already extracted: {pairs}. Do not repeat these values"
    new_prompt = (
        f'Extract the literal value of {param_name} from the input. Do not compute. '
        f'Context: {context}. Extract the {param_name} from: <input>{prompt}</input> '
        f'Context: {description}\n{param_name} = '
    )
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
    try:
        return float(result)
    except Exception as e:
        print(e)
        return 0.0


def extract_str(
    prompt: str, param_name: str, description: str,
    extracted: dict[str, Any], model: Small_LLM_Model
) -> str:
    """Extract a string parameter value from a prompt using constrained decoding.

    Generation stops when a closing double-quote token is produced. Tokens that
    contain an embedded quote (but don't end with one) are masked to prevent
    unterminated strings.

    Args:
        prompt: The user request.
        param_name: The name of the parameter to extract.
        description: Human-readable description providing extraction context.
        extracted: Parameters already extracted in this call (used to avoid repetition).
        model: The language model used for token scoring.

    Returns:
        The extracted string with surrounding quotes and whitespace stripped.
    """
    context = ""
    if extracted:
        pairs = ', '.join(f"{k}={v}" for k, v in extracted.items())
        context = f"Already extracted: {pairs}. Do not repeat these values"
    new_prompt = (
        f'Extract the {param_name} from: <input>{prompt}</input> '
        f'Context: {description} {context}\n{param_name} = "'
    )
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
        if '"' in new_char:
            return result.strip('",\n')
    return result.strip('",\n')


def extract_parameters(
    prompt: str, function: FunctionDefinition, model: Small_LLM_Model
) -> dict[str, Any]:
    """Extract all parameters defined by a function from a natural-language prompt.

    Dispatches to extract_number or extract_str based on each parameter's type.
    Previously extracted values are passed along so they are not duplicated.

    Args:
        prompt: The user request.
        function: The function definition whose parameters should be extracted.
        model: The language model used for token scoring.

    Returns:
        A dict mapping each parameter name to its extracted value.
    """
    result: dict[str, Any] = {}
    param_description = function.description
    for param_name, param in function.parameters.items():
        param_type = param.type
        if param_type == 'number':
            result[param_name] = extract_number(
                prompt, param_name, param_description, result, model
            )
        elif param_type == 'string':
            result[param_name] = extract_str(prompt, param_name, param_description, result, model)
    return result
