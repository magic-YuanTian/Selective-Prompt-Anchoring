''' Models & Device '''
from weighted_utils.weighted_text_utils import *

print(torch.cuda.device_count())

'''
7B	codellama/CodeLlama-7b-hf	codellama/CodeLlama-7b-Python-hf	codellama/CodeLlama-7b-Instruct-hf
13B	codellama/CodeLlama-13b-hf	codellama/CodeLlama-13b-Python-hf	codellama/CodeLlama-13b-Instruct-hf
34B	codellama/CodeLlama-34b-hf	codellama/CodeLlama-34b-Python-hf	codellama/CodeLlama-34b-Instruct-hf
'''

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory


# tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
# model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")

checkpoint = "codellama/CodeLlama-34b-hf"
# checkpoint = "WizardLM/WizardCoder-Python-34B-V1.0"
# checkpoint = "WizardLM/WizardCoder-Python-13B-V1.0"
# checkpoint = "WizardLM/WizardCoder-3B-V1.0"
checkpoint = "codellama/CodeLlama-7b-hf"
# checkpoint = "bigcode/starcoder"
# checkpoint = "meta-llama/Llama-2-7b-chat-hf"
# checkpoint = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto", 
        load_in_8bit=True, # under 20 GB
        # torch_dtype=torch.float16, # ~30 GB
        max_memory=get_gpus_max_memory("22GB")
    )

model.eval() # Set the model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

top_k = 10

# # Load the small English model
# nlp = spacy.load("en_core_web_sm")

