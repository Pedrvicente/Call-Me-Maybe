import os
import argparse
from llm_sdk import Small_LLM_Model
from .generator import select_function, extract_parameters, get_vocab
from .io import load_functions, load_prompts, save_outputs
from .models import OutputRequest


def main() -> None:
    """Run the function-calling pipeline from the command line.

    Parses --input, --output, and --functions_definition arguments, loads prompts
    and function definitions, runs constrained decoding to select a function and
    extract its parameters for each prompt, then writes results to the output file.
    """
    model: Small_LLM_Model = Small_LLM_Model()
    id_to_token = get_vocab(model)

    parser = argparse.ArgumentParser(description='Arguments to specify the files')
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-f', '--functions_definition', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.input is None:
        args.input = 'data/input/function_calling_tests.json'
    prompts = load_prompts(args.input)
    if args.functions_definition is None:
        args.functions_definition = 'data/input/functions_definition.json'
    functions = load_functions(args.functions_definition)
    if args.output is None:
        os.makedirs('data/output', exist_ok=True)
        args.output = 'data/output/function_calling_results.json'
    result: list[OutputRequest] = []
    for item in prompts:
        prompt = item.prompt
        print(f"\nProcessing: {prompt}")
        function_name = select_function(prompt, functions, id_to_token, model, verbose=args.verbose)
        fn_def = next((f for f in functions if f.name == function_name), None)
        if fn_def is None:
            continue
        params = extract_parameters(prompt, fn_def, id_to_token, model)
        output = OutputRequest(prompt=prompt, name=function_name, parameters=params)
        result.append(output)
    save_outputs(args.output, result)


if __name__ == '__main__':
    main()
