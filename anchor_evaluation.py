import argparse
import os
import json
import multiprocessing
from weighted_utils.weighted_text_utils import *

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Example script to demonstrate argparse usage.")

    # Add arguments
    parser.add_argument('--task_id', type=int, default=0, help="Define which GPU to use")
    # parser.add_argument('--gpu_num', type=int, default=1, help="Define the number of GPUs")
    parser.add_argument('--checkpoint', type=str, help="Declare which checkpoint to use")
    parser.add_argument('--benchmark', type=str, help="Declare the benchmark dataset to use")
    parser.add_argument('--mode', type=str, default="instruction", help="Declare which weight enhancement mode to use")
    parser.add_argument('--weight', type=float, default=0.5, help="Declare manual weight for instruction mode")
    parser.add_argument('--use_multiple_gpu', type=bool, default=False, help="Declare if using multiple GPUs")
    parser.add_argument('--approach', type=str, choices=['difference', 'ratio'], default='difference', help="Declare if add logits difference or multiple probability ratio")
    parser.add_argument('--mask_test_case', type=bool, default=False, help="For instruct mode, specify if test cases will be masked")
    parser.add_argument('--mask_entire_prompt', type=bool, default=False, help="When this is true, the entire prompt including template will be masked")
    parser.add_argument('--note', type=str, help="Addition note for the experiment")
    
    
    # get value from the command line (Parse the arguments)
    args = parser.parse_args()


    # args.task_id = 6
    # args.checkpoint = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # args.benchmark = "mbpp"

    

    # print the arguments
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Mode: {args.mode}")
    print(f"Mask test case: {args.mask_test_case}")
    print(f"Approach: {args.approach}")
    print(f"Weight: {args.weight}")
    print(f"GPU ID: {args.task_id}")
    print(f"Use multiple GPU: {args.use_multiple_gpu}")
    print(f"Note: {args.note}")
    print(f"Mask entire prompt: {args.mask_entire_prompt}")
    

    
    # run the experiment
    run_experiment_on_specific_GPU(args)
    
    

def run_experiment_on_specific_GPU(args):
    
    task_id = args.task_id
    checkpoint = args.checkpoint
    benchmark = args.benchmark
    mode = args.mode
    weight = args.weight
    use_multiple_gpu = args.use_multiple_gpu
    mask_test_case = args.mask_test_case
    note = args.note
    mask_entire_prompt = args.mask_entire_prompt
    
    # decide mask_tok
    if "deepseek" in checkpoint.lower():
        mask_tok = '<pad>'
    elif "codellama" in checkpoint.lower() or 'qwen' in checkpoint.lower():
        mask_tok = '<unk>'
    elif "codegen" in checkpoint.lower():
        mask_tok = ' '
    else:
        raise ValueError("Undefined masked tok for checkpoint")
    
    
    print(f"--------> Processing tasks on GPU {task_id}")
    # Load tasks from the JSONL file
    if benchmark == "humaneval":
        tasks = load_tasks_from_jsonl('/dataset/HumanEval.jsonl')
    elif benchmark == "mbpp":
        tasks = load_tasks_from_jsonl('/dataset/mbpp.jsonl')
    else:
        raise ValueError("Invalid benchmark")
    
    
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    # load the model
    if use_multiple_gpu:
        print('Using multiple GPUs!!!')
        model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                device_map="auto",
                load_in_8bit=True, # under 20 GB
                low_cpu_mem_usage=True, # ~20 GB
                max_memory={0: "22GB", 1: "22GB", 2: "22GB", 3: "22GB", 4: "22GB", 5: "22GB", 6: "22GB", 7: "22GB"},
            )
    else:
        # model, tokenizer, device = load_model(checkpoint, task_id)
        device = torch.device("cuda:" + str(task_id) if torch.cuda.is_available() else "cpu")
        # Set the device to the specific GPU
        torch.cuda.set_device(task_id)
        
        model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                load_in_8bit=True, # under 20 GB
                low_cpu_mem_usage=True, # ~20 GB
                # torch_dtype=torch.float16, # ~30 GB
            )
    
    
    model.eval() # Set the model to evaluation mode
    # model.to(device)

    model_name = checkpoint.split('/')[-1]
    
    if benchmark == "humaneval":
        log_dir = f'/home/yuan/research/weighted_LM/log/humaneval/{model_name}'
        generate_data_dir = f'/home/yuan/research/weighted_LM/generated_data/humaneval/{model_name}'
    elif benchmark == "mbpp":
        log_dir = f'/home/yuan/research/weighted_LM/log/mbpp/{model_name}'
        generate_data_dir = f'/home/yuan/research/weighted_LM/generated_data/mbpp/zero_shot/{model_name}'
    else:
        raise ValueError("Invalid benchmark")


    # create the directory if not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(generate_data_dir):
        os.makedirs(generate_data_dir)
    
    log_file = os.path.join(log_dir, f'{note}_{benchmark}_{mode}_{weight}w_{model_name}.txt')
    generate_file = os.path.join(generate_data_dir, f'{note}_{benchmark}_{mode}_{weight}w_{model_name}.jsonl')
    
    # clear the generate file and log file
    with open(log_file, 'a') as f:
        f.write('')
    with open(generate_file, 'a') as f:
        f.write('')

    for data in tasks:
        
        # print(data)
        
        if benchmark == "humaneval":
            prompt = data['prompt']
        elif benchmark == "mbpp":
            prompt = data['info_prompt']
        
        print('\n' + '-' * 50 + data['task_id'] + '-' * 50 + '\n')

        task_id = data['task_id']
        
        # get the number, skip the some tasks
        if benchmark == "humaneval":
            task_id_num = int(task_id.lower().replace('humaneval/', ''))
        elif benchmark == "mbpp":
            task_id_num = int(task_id.lower().replace('mbpp/', ''))
        
        
        ############## Skip tasks for recovering evaluation ##########################
        # if task_id_num >= 133:
        #     continue
        #########################################################

        with open(log_file, 'a') as f:
            f.write('\n\n' + '-' * 50 + data['task_id'] + '-' * 50 + '\n\n')
        
        

        while True:
            try:
                
                if mask_entire_prompt:
                    prompt_mask = mask_tok
                else:
                    '''instruction'''
                    if benchmark == "humaneval":
                        prompt_mask = mask_humanEval_instruction(prompt, mask_tok=mask_tok, mask_test_case=mask_test_case)  # get masked prompt
                    elif benchmark == "mbpp":
                        pure_prompt = data['prompt']
                        prompt_mask = prompt.replace(pure_prompt, mask_tok)  # mask the initial prompt
                
                if mode == "normal":
                    all_chat, completion, code = normal_generate_deepseek_instruct(prompt, model, tokenizer, device=model.device, task='python', max_length=1993, log_path=log_file)
                elif mode == "instruction":
                    if args.approach == 'difference':
                        all_chat, _, code = augmented_generate_anchor(prompt, prompt_mask, model=model, tokenizer=tokenizer, checkpoint_name=checkpoint, weight=weight, device=model.device, language='python', benchmark=benchmark, max_length=750, log_path=log_file)
                    elif args.approach == 'ratio':
                        all_chat, _, code = augmented_generate_anchor(prompt, prompt_mask, model=model, tokenizer=tokenizer, checkpoint_name=checkpoint, weight=weight, device=model.device, task='python', benchmark=benchmark, max_length=750, log_path=log_file)
                    else:
                        raise ValueError("Invalid approach")
                elif mode == "instruction_adaptive":
                    all_chat, _, code = augmented_generate_anchor(prompt, prompt_mask, model=model, tokenizer=tokenizer, checkpoint_name=checkpoint, weight=weight, device=model.device, language='python', benchmark=benchmark, max_length=750, log_path=log_file, adaptive_attention_weight=True)
                elif mode == "instruction_confidence":
                    all_chat, _, code = augmented_generate_deepseek_instruct_confidence(prompt, prompt_mask, model, tokenizer, weight=weight, device=model.device, task='python', benchmark=benchmark, max_length=1500, log_path=log_file)
                elif mode == "self_attention":
                    _, _, code = augmented_generate_deepseek_self_attention(prompt, model, tokenizer, weight=weight, device=model.device, task='python', benchmark=benchmark, max_length=1500, log_path=log_file)
                elif mode == "sample":
                    solution, completion = sample_augment_generate(prompt, model, mask_tok=mask_tok, max_length=2500, top_k=10, max_sample=500, tokenizer=tokenizer, device=device, log_path=log_file)
                elif mode == "attention_change":
                    # _, attention_change, code = augmented_generate_anchor(prompt, prompt_mask, model=model, tokenizer=tokenizer, checkpoint_name=checkpoint, weight=0, device=model.device, language='python', benchmark=benchmark, max_length=750, output_attentions=True, log_path=log_file)
                    _, attention_change, code = augmented_generate_anchor(prompt, prompt_mask, model=model, tokenizer=tokenizer, checkpoint_name=checkpoint, weight=weight, device=model.device, language='python', benchmark=benchmark, output_attentions=True, max_length=600, log_path=log_file)
                    
                    self_attention_change = attention_change[0]
                    gradient_attention_change = attention_change[1]
                    
                    # save the attention change (which is a list) into a file
                    self_attention_item = {'task_id': task_id, 'attention_change': self_attention_change}
                    gradient_attention_item = {'task_id': task_id, 'attention_change': gradient_attention_change}
                        
                    # save the attention_item into a file
                    with open(f'attention_data/{note}_{model_name}_self_attention_change.jsonl', 'a') as f:
                        f.write(json.dumps(self_attention_item) + '\n')
                    
                    # save the attention_item into a file
                    with open(f'attention_data/{note}_{model_name}_gradient_attention_change.jsonl', 'a') as f:
                        f.write(json.dumps(gradient_attention_item) + '\n')
                    
                else:
                    raise ValueError("Invalid mode")


                # post process for text completion in humaneval
                if benchmark == "humaneval":
                    if 'codgen' in checkpoint.lower() or 'codellama' in checkpoint.lower():
                        code = prompt + code
                
                
                # create the dict element to save
                # generate_data_element = {'task_id': data['task_id'], 'prompt': prompt, 'solution': code, 'canonical_solution': data['canonical_solution'], 'test': data['test'], 'entry_point': data['entry_point']}
                generate_data_element = {'task_id': data['task_id'], 'prompt': prompt, 'solution': code, 'canonical_solution': data['canonical_solution']}
                # save the new generated data onto generate_file_path
                with open(generate_file, 'a') as f:
                    f.write(json.dumps(generate_data_element) + '\n')

            except Exception as e:
                # raise e
                
                print(f"ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: {e}")
                print('Retrying...')
                continue

            break

        
    print(f"--------> Finished processing tasks on GPU {task_id} for the benchmark HumanEval.")
   

def load_tasks_from_jsonl(jsonl_file):
    tasks = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            tasks.append(json.loads(line))
    return tasks

def divide_tasks(tasks, num_groups):
    """Divide tasks into roughly equal parts"""
    k, m = divmod(len(tasks), num_groups)
    return [tasks[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_groups)]

if __name__ == "__main__":
    main()

