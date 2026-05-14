import time
import numpy as np
from typing import Callable


def define_status_for_function(tid: int,
                               id_to_token: dict[int, str],
                               valid_indices: list[int]) -> str:
    """Return a display status string for a token during function-name decoding.

    Args:
        tid: Token ID to classify.
        id_to_token: Mapping from token ID to decoded text.
        valid_indices: Token IDs that passed the function-name prefix check.

    Returns:
        A status emoji string: SPECIAL for out-of-vocab tokens, MASKED for
        filtered tokens, or VALID for tokens that remain in the beam.
    """
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
    """Return a display status string for a token during numeric-parameter decoding.

    Args:
        tid: Token ID to classify.
        id_to_token: Mapping from token ID to decoded text.
        valid_after_digit_check: Token IDs that contain only digit/period characters.
        valid_after_substring_check: Token IDs whose text is a substring of the original prompt.

    Returns:
        A status emoji string indicating whether the token is SPECIAL, masked for
        not being a digit, masked for not being a prompt substring, or VALID.
    """
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
    """Print the top-n tokens by logit score with their decoded text and status.

    Args:
        logits: Full logit vector over the vocabulary.
        id_to_token: Mapping from token ID to decoded text.
        get_status: Callable that maps a token ID to a display status string.
        n: Number of top tokens to display (default 3).
    """
    logits_np = np.array(logits)
    top_3_ids = np.argsort(logits_np)[-n:][::-1]
    for tid in top_3_ids:
        token_text = id_to_token.get(tid, '???')
        status = get_status(tid)
        score = logits_np[tid]
        print(f"Token: '{token_text}' | Score: {score:.2f} | {status}")
        time.sleep(0.6)


def log_step(
    original_logits: list[float],
    masked_logits: list[float],
    id_to_token: dict[int, str],
    valid_indices: list[int],
    step_number: int,
    written_function: str
) -> None:
    """Print a verbose decoding step for function-name selection.

    Shows the top tokens before and after masking, the chosen token, and
    the function name written so far (or the finished name once complete).

    Args:
        original_logits: Logits before any masking was applied.
        masked_logits: Logits after invalid function-prefix tokens were masked.
        id_to_token: Mapping from token ID to decoded text.
        valid_indices: Token IDs that passed the function-name prefix check.
        step_number: Current generation step index (used for display only).
        written_function: The portion of the function name generated so far.
    """
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
    time.sleep(0.6)
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
    """Print a verbose decoding step for numeric-parameter extraction.

    Shows the top tokens before and after the two-stage masking (digit check,
    then substring check), the chosen token, and the number written so far.

    Args:
        original_logits: Logits before any masking was applied.
        masked_logits: Logits after both masking stages were applied.
        id_to_token: Mapping from token ID to decoded text.
        valid_after_digit_check: Token IDs that contain only digit/period characters.
        valid_after_substring_check: Token IDs whose text is a substring of the original prompt.
        step_number: Current generation step index (used for display only).
        written_number: The portion of the numeric string generated so far.
    """
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
    time.sleep(0.6)
    print(f"\nWritten Number So Far: {written_number}")
