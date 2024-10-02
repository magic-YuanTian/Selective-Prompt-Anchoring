from weighted_utils.weighted_text_utils import *
from datasets import load_dataset

# define a function to tune the weight for SPA
# sample values through grid search
def tune_SPA(tune_dataset, model, tokenizer, weight_range, benchmark, checkpoint):
    # sample values through grid search
    for weight in weight_range:
        _, _, code = augmented_generate_anchor(prompt, prompt_mask, model=model, tokenizer=tokenizer, checkpoint_name=checkpoint, weight=weight, device=model.device, language='python', benchmark=benchmark, max_length=750, log_path=log_file)



# based on the task, get the dataset
def get_dataset(task):
    if task == "HumanEval":
        task_humaneval_plus = load_dataset("evalplus/humanevalplus")['test']
    elif task == "mbpp":
        tasks = load_dataset("mbpp/mbpp")['test']
    else:
        raise ValueError("Invalid benchmark")
    
    return tasks
    




