from llama_cpp import Llama
import copy
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + '\models'

def get_model_file_path(model_name):
    """Returns the path of the model downloaded."""
    if model_name not in os.listdir(MODEL_DIR):
        raise ValueError("Model file not found.")
    return os.path.join(MODEL_DIR, model_name)

model_filename = 'llama-2-7b-chat.ggmlv3.q2_K.bin'
file_path = get_model_file_path(model_filename)
llm = Llama(model_path=file_path)

stream = llm(
    "What are language models?",
    max_tokens=120,
    stream=True,
)

for output in stream:
    completion_fragment = copy.deepcopy(output)
    print(completion_fragment['choices'][0]['text'])