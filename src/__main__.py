import sys
sys.path.insert(0, 'llm_sdk')
import os
import json
import argparse
from llm_sdk import Small_LLM_Model
from src.prompt import build_prompt

def main() -> None:
    model: Small_LLM_Model = Small_LLM_Model()

    parser = argparse.ArgumentParser(description='Arguments to specify the files')
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-f', '--functions_definition', type=str)
    args = parser.parse_args()

    if args.input is None:
        args.input = 'data/input/function_calling_tests.json'
    with open(args.input, 'r') as f:
        data = json.load(f)
    if args.functions_definition is None:
        args.functions_definition = 'data/input/functions_definition.json'
    with open(args.functions_definition, 'r') as f:
        functions = json.load(f)
    if args.output is None:
        os.makedirs('data/output', exist_ok=True)
        args.output = 'data/output/function_calling_results.json'
    with open(args.output, 'w') as f:
        output = json.dump(data, f)
    for item in data:
        prompt = item['prompt']
        result = build_prompt(prompt, functions)
        tokens = model.encode(result)
        values = tokens[0].tolist()
        logits = model.get_logits_from_input_ids(values)
        next_token = max(logits)
        index_next_token = logits.index(next_token)
        print(index_next_token)

if __name__ == '__main__':
    main()
