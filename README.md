*This project has been created as part of the 42 curriculum by pedde-al*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling tool that translates natural language prompts into structured function calls using a local Large Language Model (Qwen3-0.6B by default). Given a prompt like "What is the sum of 2 and 3?", the system identifies the correct function to call and extracts its arguments, returning structured JSON output without executing the function itself.

The core challenge is reliability: small language models (0.6B parameters) are notoriously unreliable at generating valid JSON spontaneously. This project solves that using **constrained decoding**, a technique that intervenes in the model's token generation process to guarantee structurally valid output every time.

## Instructions

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- The `llm_sdk/` directory must be present at the project root (copied from the provided package)

### Installation

```bash
git clone <your-repo-url>
cd call_me_maybe
uv sync
```

### Running

Default run (reads from `data/input/`, writes to `data/output/`):

```bash
uv run python -m src
```

With custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

With verbose output (shows constrained decoding steps in real time):

```bash
uv run python -m src --verbose
```

With an alternative model:

```bash
uv run python -m src --model Qwen/Qwen3-1.7B
```

### Input format

- `data/input/functions_definition.json`: list of available functions with `name`, `description`, `parameters`, and `returns`.
- `data/input/function_calling_tests.json`: list of natural language prompts to process.

### Output format

`data/output/function_calling_results.json`: JSON array where each object contains `prompt`, `name`, and `parameters`.

## Algorithm Explanation

The pipeline runs in two phases per prompt:

**Phase 1 — Function selection via constrained decoding**

A prompt is built describing all available functions, ending with `Function: "`. The model generates the function name token by token. At each step, every token in the vocabulary is tested: if the candidate `written_so_far + token_chars` is not a prefix of any known function name (with a closing `"` appended as terminator marker), that token's logit is set to `-inf` before sampling. The model can therefore only produce a known function name, and generation terminates when the closing `"` token is produced.

**Phase 2 — Argument extraction**

Once the function is selected, each parameter is extracted according to its type:

- **`number` / `integer`**: constrained decoding allows only tokens whose characters are all in `0123456789.`, and further requires that `result + token_chars` be a continuous substring of the original prompt. This prevents the model from computing answers (for `sqrt(144)`, for example, the model is blocked from producing `12`) and from hallucinating numeric values. Generation stops when the model's natural top-1 token (before masking) is no longer a digit, signalling that the literal value is complete.

- **`string`**: the model generates inside a quoted context (`param = "`), with the same substring constraint applied to keep the output anchored to the prompt. Tokens that contain an embedded quote in the middle are masked to prevent unterminated strings. Generation stops when a token ending in `"` is produced.

Already-extracted parameters are passed forward as context to disambiguate cases where the same function has multiple parameters of the same type. Without this, `fn_add_numbers(a, b)` on the prompt "Add 2 and 3" would extract `2` for both `a` and `b`.

## Design Decisions

**Constrained decoding over prompting alone.** Prompting a small model to produce valid JSON is unreliable. Constraining the logits at each generation step achieves near-100% structural validity without relying on model capability.

**Two-pass approach.** Function selection and argument extraction are kept separate. The first pass only needs to match function name prefixes; the second pass applies per-type constraints. Keeping these concerns apart makes each constraint simpler and easier to reason about.

**Pydantic models for all data.** `FunctionDefinition`, `PromptRequest`, and `OutputRequest` are validated on load. Malformed input files produce clear error messages and exit cleanly rather than crashing mid-pipeline.

**Vocabulary inversion at startup, passed by dependency injection.** The vocabulary file (`token → id`) is loaded once and inverted to `id → token`. The resulting `id_to_token` dict is computed once in `main()` and passed down through all functions, avoiding the repeated recomputation that would otherwise occur dozens of times per prompt.

**Context for already-extracted parameters.** Rather than re-running the model with cleverly engineered prompts to avoid duplicates, the previously-extracted values are simply included as context, letting the model itself break the symmetry.

## Performance Analysis

On the provided test set (11 prompts):

- **Function selection**: ~10/11 correct. The persistent failure is a case where lexical similarity between the prompt and a wrong function name outweighs the correct function's semantic match — a limitation of the 0.6B-parameter model's reasoning, not of the constraint architecture.
- **Argument extraction**: roughly 90% of parameters extracted correctly. Remaining failures concentrate on cases of genuine semantic ambiguity, where two distinct strings in the prompt are both valid substrings of the input and the constraint alone cannot disambiguate between them.
- **Runtime**: approximately 30–60 seconds for the full test set on Apple Silicon (MPS). Larger models (via the `--model` flag) trade speed for accuracy.

The Qwen3-0.6B model achieves usable reliability for this structured task purely through constrained decoding — the same architecture with a larger model resolves most of the remaining failures without changing a line of code, as can be verified by running `--model Qwen/Qwen3-1.7B`.

**Out-of-scope prompts.** Because constrained decoding forces selection of a token that is a prefix of some function name, prompts that do not semantically match any defined function will still produce a function call (typically the closest lexical match). For example, `"What is apple?"` against a function set with no dictionary-related function will select the least implausible existing function. This is an inherent consequence of the structural guarantee constrained decoding provides. Production systems would address this with an explicit fallback function (e.g. `fn_unknown`) in the function list, a prior in-scope classification step, or a confidence threshold based on the gap between the natural top-1 logit and the constrained top-1.

## Challenges Faced

**BPE space prefix (`Ġ`).** The tokenizer prefixes tokens with `Ġ` to encode a preceding space. Early attempts to match function names failed because the model generated `Ġfn_greet` instead of `fn_greet`. Solution: end the prompt with `"` so the model is completing a quoted string, eliminating the space prefix entirely.

**String extraction stopping condition.** Greedy constrained decoding for string parameters had no natural stopping signal. Solution: prompt the model with `param = "` so generation happens inside a string, stopping when a token ending in `"` is produced.

**Distinguishing repeated values.** For functions with multiple parameters of the same type, the model would otherwise extract the same first number for every parameter slot. Solution: pass already-extracted parameters as context, biasing the model toward the next unextracted value.

**`llm_sdk` package shadowing.** The outer `llm_sdk/` folder in the project root shadows the installed package when running scripts directly. Solution: run via `uv run python -m src` (or set `PYTHONPATH=./llm_sdk` via the Makefile), which resolves the package correctly.

## Testing Strategy

Validation was done in two ways:

- **End-to-end testing**: running the full pipeline against the test set after each change and comparing actual against expected outputs.
- **Visualization (`--verbose` mode)**: the `log_step` and `log_int_step` functions print, at every generation step, the model's natural top-3 tokens (pre-mask) alongside the post-mask top-3, with each token annotated as `VALID`, `MASKED`, or `SPECIAL`. This makes it possible to see in real time exactly where the constraint is intervening on the model's choice and where the model would have produced the same token without intervention — useful both for debugging and for understanding the system's behaviour.

Edge cases verified manually include: prompts with punctuation attached to numbers (e.g. `"3?"`), decimal and large numeric values, multi-word strings, paths containing special characters, and prompts in which the same word appears multiple times.

## Bonuses Implemented

- **Multiple model support** — the `--model` flag accepts any Hugging Face model identifier compatible with the SDK. Default is `Qwen/Qwen3-0.6B`.
- **Performance optimization** — vocabulary inversion is computed once and injected into every function that needs it, eliminating repeated recomputation.
- **Visualization of the generation process** — `--verbose` mode displays the constrained decoding step by step (see Testing Strategy above).

## Example Usage

```bash
# Default paths
uv run python -m src

# Custom input/output
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --output data/output/results.json

# Verbose mode (visualize constrained decoding)
uv run python -m src --verbose

# Alternative model
uv run python -m src --model Qwen/Qwen3-1.7B
```

Example output:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

## Resources

- [Outlines library — constrained decoding (conceptual reference)](https://github.com/outlines-dev/outlines)
- [Andrej Karpathy — "Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE): This video was super helful to understand the conceps of Enconding, Decoding, and the process of Tokenization through BytePair Encoding
- [Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1050s&pp=ygUPbGV0cyBidWlsZCBjaGF00gcJCQQLAYcqIYzv): Also super very helpful to understand a lot of concepts
- [minBPE — Karpathy's reference BPE implementation](https://github.com/karpathy/minbpe)
- [Pydantic documentation](https://docs.pydantic.dev/)

**AI usage.** AI was used throughout this project tutor, asking questions to guide understanding rather than providing direct solutions. Specifically, AI was used to explain concepts (logits, BPE tokenization, constrained decoding), to debug import errors and logic issues, to review code structure, and to discuss design trade-offs. All implementation decisions and code were written by me with AI as a guide, not as a code generator.