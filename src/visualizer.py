import time
import numpy as np


def define_status(tid: int,
                  id_to_token: dict[int, str],
                  valid_indices: list[int]) -> str:
    if tid not in id_to_token:
        return "⚠️ SPECIAL"
    if tid not in valid_indices:
        return "❌ MASKED"
    return "✅ VALID"


def print_top_tokens(logits: list[float],
                     id_to_token: dict[int, str],
                     valid_indices: list[int],
                     n: int = 3) -> None:
    logits_np = np.array(logits)
    top_3_ids = np.argsort(logits_np)[-n:][::-1]
    for tid in top_3_ids:
        token_text = id_to_token.get(tid, '???')
        status = define_status(tid, id_to_token, valid_indices)
        score = logits_np[tid]
        print(f"Token: '{token_text}' | Score: {score:.2f} | {status}")
        time.sleep(0.8)


def log_step(original_logits: list[float],
             masked_logits: list[float],
             id_to_token: dict[int, str],
             valid_indices: list[int],
             step_number: int,
             written_function: str) -> None:
    print(f"\n--- STEP {step_number} ---")
    print("My Original Top 3 Tokens are:")
    print_top_tokens(original_logits, id_to_token, valid_indices)

    print("\nMy Top 3 Tokens after Masking are:")
    print_top_tokens(masked_logits, id_to_token, valid_indices)

    logits_np = np.array(masked_logits)
    best_option = int(np.argmax(logits_np))
    print(f"Choose: '{id_to_token.get(best_option)}'")
    time.sleep(0.8)

    if '"' in written_function:
        print(f"\nFinished Function: {written_function}")
    else:
        print(f"\nWritten So Far: {written_function}")
