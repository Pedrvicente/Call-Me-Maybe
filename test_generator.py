import sys
sys.path.insert(0, 'llm_sdk')
from src.generator import extract_parameters
from src.models import FunctionDefinition, Parameter, PromptRequest

if __name__ == '__main__':
    func = FunctionDefinition(name='fn_add_numbers', description='Add two numbers together and return their sum.', parameters={'a': Parameter(type='number'), 'b': Parameter(type='number')}, returns=Parameter(type='number'))
    promp = "What is the sum of 2 and 3?"
    extract_parameters(promp, func)