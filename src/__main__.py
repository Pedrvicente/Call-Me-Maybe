import sys
sys.path.insert(0, 'llm_sdk')
import os
import json
import argparse
from llm_sdk import Small_LLM_Model
from src.prompt import build_prompt
from .generator import select_function, extract_parameters
from .io import load_functions, load_prompts, save_outputs
from .models import OutputRequest

def main() -> None:
    model: Small_LLM_Model = Small_LLM_Model()

    parser = argparse.ArgumentParser(description='Arguments to specify the files')
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-f', '--functions_definition', type=str)
    args = parser.parse_args()

    if args.input is None:
        args.input = 'data/input/function_calling_tests.json'
    prompts = load_prompts(args.input)
    print(prompts)
    if args.functions_definition is None:
        args.functions_definition = 'data/input/functions_definition.json'
    functions = load_functions(args.functions_definition)
    if args.output is None:
        os.makedirs('data/output', exist_ok=True)
        args.output = 'data/output/function_calling_results.json'
    result: OutputRequest = []
    for item in prompts:
        prompt = item.prompt
        function_name = select_function(prompt, functions, model)
        extract_parameters(prompt, function_name, model)
        output = OutputRequest(prompt=prompt, name=function_name, parameters={})
        result.append(output)
    save_outputs(args.output, result)

if __name__ == '__main__':
    main()
