from .prompt import build_prompt
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition
from .visualizer import log_step, log_int_step
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
    prompt: str,
    functions: list[FunctionDefinition],
    id_to_token: dict[int, str],
    model: Small_LLM_Model,
    verbose: bool = False
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
    function_names = [f'{f.name}"' for f in functions]
    written_function = ""
    result = build_prompt(prompt, functions)
    for step in range(20):
        tokens = model.encode(result)
        values = tokens[0].tolist()
        logits = model.get_logits_from_input_ids(values)
        original_logits = list(logits)
        valid_indexes = []
        for token_id in range(len(logits)):
            if token_id in id_to_token:
                chars = id_to_token[token_id]
                if not any(name.startswith(written_function + chars) for name in function_names):
                    logits[token_id] = float('-inf')
                else:
                    valid_indexes.append(token_id)
        next_token = max(logits)
        next_token_id = logits.index(next_token)
        if next_token_id not in id_to_token:
            break
        char = id_to_token[next_token_id]
        result += char
        written_function += char
        if verbose:
            log_step(original_logits, logits, id_to_token, valid_indexes, step, written_function)
        if written_function.endswith('"'):
            return written_function[:-1]
    return ""


def extract_number(
    prompt: str,
    param_type: str,
    param_name: str,
    description: str,
    extracted: dict[str, Any],
    id_to_token: dict[int, str],
    model: Small_LLM_Model,
    verbose: bool = False
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
    result = ""
    for step in range(80):
        valid_after_digit_check: list[int] = []
        valid_after_substring_check: list[int] = []
        tokens = model.encode(new_prompt)[0].tolist()
        logits = model.get_logits_from_input_ids(tokens)
        original_logits = list(logits)
        original_token = max(logits)
        original_token_id = logits.index(original_token)
        if original_token_id not in id_to_token:
            break
        original_char = id_to_token[original_token_id]
        if any(c not in '0123456789.' for c in original_char):
            break
        for token_id in range(len(logits)):
            if token_id in id_to_token:
                chars = id_to_token[token_id]
                if any(c not in '0123456789.' for c in chars):
                    logits[token_id] = float('-inf')
                    continue
                valid_after_digit_check.append(token_id)
                if (result + chars) not in prompt:
                    logits[token_id] = float('-inf')
                else:
                    valid_after_substring_check.append(token_id)
        next_token = max(logits)
        next_token_id = logits.index(next_token)
        if next_token_id not in id_to_token:
            break
        new_char = id_to_token[next_token_id]
        result += new_char
        if verbose:
            log_int_step(original_logits,
                         logits,
                         id_to_token,
                         valid_after_digit_check,
                         valid_after_substring_check,
                         step,
                         result)
        new_prompt += new_char
    try:
        if param_type == 'integer':
            return int(result)
        if param_type == 'number':
            return float(result)
        return 0.0
    except (ValueError, TypeError) as e:
        print(f"Warning: could not convert '{result}' to {param_type}: {e}")
        return 0.0


def extract_str(
    function_name: str,
    prompt: str,
    param_name: str,
    description: str,
    extracted: dict[str, Any],
    id_to_token: dict[int, str],
    model: Small_LLM_Model
) -> str:
    context = ""
    if extracted:
        pairs = ', '.join(f"{k}={v}" for k, v in extracted.items())
        context = f"Already extracted: {pairs}. Do not repeat these values"
    new_prompt = (
            f'Function: {function_name}\n'
            f'Function purpose: {description}\n'
            f'Input: {prompt}\n'
            f'Already extracted parameters: {context if context else "none"}\n'
            f'Now extracting parameter "{param_name}". '
            f'Its value should be a substring of the input that has not been extracted yet.\n'
            f'{param_name} = "'
            )
    result = ""
    for step in range(80):
        tokens = model.encode(new_prompt)[0].tolist()
        logits = model.get_logits_from_input_ids(tokens)
        for token_id in range(len(logits)):
            if token_id not in id_to_token:
                continue
            chars = id_to_token[token_id]
            # bloquear aspas a meio (mantém o terminador estável)
            if '"' in chars and not chars.endswith('"'):
                logits[token_id] = float('-inf')
            # bloquear < e > (impede regurgitação de </input>)
            if any(c in '<>' for c in chars):
                logits[token_id] = float('-inf')
        next_token = max(logits)
        next_token_id = logits.index(next_token)
        if next_token_id not in id_to_token:
            break
        new_char = id_to_token[next_token_id]
        result += new_char
        new_prompt += new_char
        if new_char.endswith('"'):
            return result.rstrip('"')
    return result


def extract_parameters(
    prompt: str,
    function: FunctionDefinition,
    id_to_token: dict[int, str],
    model: Small_LLM_Model,
    verbose: bool = False
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
        if param_type == 'number' or param_type == 'integer':
            result[param_name] = extract_number(
                prompt,
                param_type,
                param_name,
                param_description,
                result,
                id_to_token,
                model,
                verbose
            )
        elif param_type == 'string':
            result[param_name] = extract_str(
                function.name, prompt, param_name, param_description, result, id_to_token, model)
    return result
