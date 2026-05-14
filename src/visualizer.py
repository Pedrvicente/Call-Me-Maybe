import time
import numpy as np
from typing import Callable


def define_status_for_function(tid: int,
                               id_to_token: dict[int, str],
                               valid_indices: list[int]) -> str:
    if tid not in id_to_token:
        return "⚠️ SPECIAL"
    if tid not in valid_indices:
        return "❌ MASKED"
    return "✅ VALID"


def define_status_for_int_param(tid: int,
                                id_to_token: dict[int, str],
                                valid_after_digit_check: list[int],
                                valid_after_substring_check: list[int]
                                ) -> str:
    if tid not in id_to_token:
        return "⚠️ SPECIAL"
    if tid not in valid_after_digit_check:
        return "❌ MASKED (Not a Digit)"
    if tid not in valid_after_substring_check:
        return "❌ MASKED (Not a Substring)"
    return "✅ VALID"


def print_top_tokens(
                     logits: list[float],
                     id_to_token: dict[int, str],
                     get_status: Callable[[int], str],
                     n: int = 3) -> None:
    logits_np = np.array(logits)
    top_3_ids = np.argsort(logits_np)[-n:][::-1]
    for tid in top_3_ids:
        token_text = id_to_token.get(tid, '???')
        status = get_status(tid)
        score = logits_np[tid]
        print(f"Token: '{token_text}' | Score: {score:.2f} | {status}")
        time.sleep(0.8)


def log_step(
    original_logits: list[float],
    masked_logits: list[float],
    id_to_token: dict[int, str],
    valid_indices: list[int],
    step_number: int,
    written_function: str
) -> None:
    print(f"\n--- STEP {step_number} ---")
    print("\nMy Original Top 3 Tokens are:")
    print_top_tokens(original_logits, id_to_token,
                     lambda tid: define_status_for_function(
                         tid,
                         id_to_token,
                         valid_indices
                         ))
    print("\nMy Top 3 Tokens after Masking are:")
    print_top_tokens(
        masked_logits,
        id_to_token,
        lambda tid: define_status_for_function(
            tid,
            id_to_token,
            valid_indices
        )
    )
    logits_np = np.array(masked_logits)
    best_option = int(np.argmax(logits_np))
    print(f"\nChoose: '{id_to_token.get(best_option, '???')}'")
    time.sleep(0.8)
    if '"' in written_function:
        print(f"\nFinished Function: {written_function}")
    else:
        print(f"\nWritten So Far: {written_function}")


def log_int_step(
    original_logits: list[float],
    masked_logits: list[float],
    id_to_token: dict[int, str],
    valid_after_digit_check: list[int],
    valid_after_substring_check: list[int],
    step_number: int,
    written_number: str
) -> None:
    print(f"\n--- STEP {step_number} ---")
    print("\nMy Original Top 3 Tokens are:")
    print_top_tokens(
        original_logits,
        id_to_token,
        lambda tid: define_status_for_int_param(
            tid,
            id_to_token,
            valid_after_digit_check,
            valid_after_substring_check
        )
    )
    print("\nMy Top 3 Tokens after Masking are:")
    print_top_tokens(
        masked_logits,
        id_to_token,
        lambda tid: define_status_for_int_param(
            tid,
            id_to_token,
            valid_after_digit_check,
            valid_after_substring_check
        )
    )
    logits_np = np.array(masked_logits)
    best_option = int(np.argmax(logits_np))
    print(f"\nChoose: '{id_to_token.get(best_option, '???')}'")
    time.sleep(0.8)
    print(f"\nWritten Number So Far: {written_number}")
