import sys
sys.path.insert(0, 'llm_sdk')
from src.generator import extract_parameters, extract_str
from src.models import FunctionDefinition, Parameter, PromptRequest
from llm_sdk import Small_LLM_Model

if __name__ == '__main__':
    model = Small_LLM_Model()
    func = FunctionDefinition(
    name='fn_greet',
    description='Generate a greeting message for a person by name.',
    parameters={'name': Parameter(type='string')},
    returns=Parameter(type='string')
)
    promp = "Greet Shrek"
    result = extract_parameters(promp, func, model)
    print(result)