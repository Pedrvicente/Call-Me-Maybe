import time
import numpy as np

def log_step(logits, id_to_token, valid_indices, step_number, written_function):
    # Convert logits to a numpy array for easier manipulation
    logits_np = np.array(logits)
    
    # Get indices of the top 3 highest scores
    top_3_ids = np.argsort(logits_np)[-3:][::-1]
    best_option = top_3_ids[0]
    
    print(f"\n--- STEP {step_number} ---")
    print(f"My Top 3 Tokens are:")
    for tid in top_3_ids:
        token_text = id_to_token.get(tid, "???")
        # Check if your logic would have allowed this token
        status = "✅ VALID" if tid in valid_indices else "❌ MASKED"
        score = logits_np[tid]
        print(f"Token: '{token_text}' | Score: {score:.2f} | {status}")
        time.sleep(0.8)
    print(f"Choose: '{id_to_token.get(best_option)}'")
    time.sleep(0.8)
    if '"' in written_function:
        print(f"Finished Function: {written_function}")
    else:
        print(f"Written So Far: {written_function}")


