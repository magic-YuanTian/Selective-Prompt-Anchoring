import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # specify which GPU(s) to be used

import contextlib
# from evalplus.data import get_human_eval_plus, write_jsonl
# from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, CodeLlamaTokenizer
import torch.nn.functional as F
import torch
import json
import copy
import gc
import time
import difflib
import re
import matplotlib.pyplot as plt
import numpy as np
import random
from IPython.display import display, HTML
from IPython.display import display, HTML, clear_output
import time
import datetime
import numpy as np
from scipy.optimize import minimize
from bisect import insort
from collections import OrderedDict
import math


''' Class '''
############################################################################################################
# element: {score, left_mask_idx, right_mask_idx, has_decomposed}
class SortedDict:
    def __init__(self):
        self.dict = {}
        self.sorted_keys = []

    def insert(self, boundary_id, value_dict):
        # Remove the previous entry if updating
        if boundary_id in self.dict:
            raise ValueError('Boundary ID already exists')

        # Insert new entry
        insort(self.sorted_keys, (value_dict['score'], boundary_id))
        self.dict[boundary_id] = value_dict

    def remove(self, boundary_id):
        if boundary_id in self.dict:
            value_dict = self.dict.pop(boundary_id)
            self.sorted_keys.remove((value_dict['score'], boundary_id))

    def get_top_k(self, k):
        top_k_ids = [boundary_id for _, boundary_id in self.sorted_keys[-k:]]
        res = OrderedDict((boundary_id, self.dict[boundary_id]) for boundary_id in reversed(top_k_ids))
        # convert to list
        res = list(res.values())
        return res
    
    # get all the sorted elements
    def get_sorted_all(self):
        res = OrderedDict((boundary_id, self.dict[boundary_id]) for _, boundary_id in self.sorted_keys)
        # convert to list
        res = list(res.values())
        return res
    
    def get_element_by_id(self, boundary_id):
        # Return the element by boundary_id if it exists
        return self.dict.get(boundary_id, None)
    
    # return a list cotaining all elements with has_decomposed = False
    def get_all_not_decomposed(self):
        return [self.dict[boundary_id] for boundary_id in self.dict if not self.dict[boundary_id]['has_decomposed']]

    # get the element with highest score that satisifies the conditions
    # 1. the 'has_decomposed' is False
    # 2. the boundary is big enough to be decomposed
    def get_decomposable_highest_score_element(self):
        for score, boundary_id in reversed(self.sorted_keys):
            # get the left and right index by splitting the boundary_id
            left_idx, right_idx = boundary_id.split('#')
            if int(right_idx) - int(left_idx) < 1:  # too small to be decomposed
                continue
            
            element = self.dict[boundary_id]
            if not element['has_decomposed']:
                return boundary_id, element
        
        # if there is no such element, return None
        print('should decomposed to the end')
        return None, None

    # the funciton is used to clear the sorted boundary list
    def clear(self):
        self.dict = {}
        self.sorted_keys = []



''' Methods '''
############################################################################################################

# load model
'''
    7B	codellama/CodeLlama-7b-hf	codellama/CodeLlama-7b-Python-hf	codellama/CodeLlama-7b-Instruct-hf
    13B	codellama/CodeLlama-13b-hf	codellama/CodeLlama-13b-Python-hf	codellama/CodeLlama-13b-Instruct-hf
    34B	codellama/CodeLlama-34b-hf	codellama/CodeLlama-34b-Python-hf	codellama/CodeLlama-34b-Instruct-hf
'''
def load_model(checkpoint, gpu_id):

    # based on the GPU id, decide the device
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

    # Set the device to the specific GPU
    torch.cuda.set_device(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            load_in_8bit=True, # under 20 GB
            low_cpu_mem_usage=True, # ~20 GB
            # torch_dtype=torch.float16, # ~30 GB
        )

    # model.to(device)

    model.eval() # Set the model to evaluation mode

    return model, tokenizer, device


def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory


def find_last_python_function(code: str) -> str:
    
    # Find all occurrences of function definitions, capturing their start positions
    function_starts = [match.start() for match in re.finditer(r'\ndef\s', code)]
    
    if function_starts:
        # Identify the start of the last function
        last_function_start = function_starts[-1]
        # Extract everything from the start of the last function to the end of the string
        last_function_text = code[last_function_start:]
        
        previous_code = code[:last_function_start]
        
        return previous_code, last_function_text
    else:
        raise ValueError("No function definitions found in the provided code.")



def mask_mbpp_instruction_with_shots(prompt, mask_tok):
    # Pattern to find "Here is my problem:\n" followed by any characters
    # `re.DOTALL` makes the dot (.) match newlines as well
    pattern = r"(Here is my problem:\n).*"
    # Replace any text following "Here is my problem:\n" with `new_text`
    replaced_text = re.sub(pattern, r'\1' + mask_tok, prompt, flags=re.DOTALL)
    return replaced_text


# According to the format of humanEval data, only mask the instruction part, return the masked prompt
# reverse_mask: if True, mask the reduce non-anchored parts 
def mask_humanEval_instruction(prompt, mask_tok, mask_test_case=False, reverse_mask=False):
    
    # get previous code and the last function
    previous_code, last_function = find_last_python_function(prompt)
    
    if previous_code is None:
        raise Exception('No function found in the prompt')
    
    # process the last python function
    
    # Find the start of the docstring
    docstring_start_index = last_function.find('"""') + 3
    # Extract the part of the last_function starting from the docstring
    docstring_content = last_function[docstring_start_index:]
    # Find the end of the docstring
    docstring_end_index = docstring_content.find('"""')
    # Extract the docstring content
    docstring = docstring_content[:docstring_end_index]
    # Split the docstring into lines
    lines = docstring.split('\n')
    # Initialize an empty list to hold the lines of the instruction part
    instruction_lines = []
    # Iterate over each line
    for line in lines:
        # Check if the line marks the beginning of a test case
        if not mask_test_case:
            if line.strip().startswith('>>>'):
                break
        # If not a test case, add the line to the instruction_lines list
        instruction_lines.append(line)
    # Join the instruction lines back into a single string and return it
    instruction_part = '\n'.join(instruction_lines).strip()
    
    if reverse_mask:
        # replace the non-instruction part with mask_tok
        result = mask_tok + instruction_part + mask_tok
    else:
        # replace the instruction part with mask_tok
        result = last_function.replace(instruction_part, mask_tok)
    
    # add previous code
    result = previous_code + result
    
    return result



# According to the format of humanEval data, only mask the instruction part, return the masked prompt
def remove_humanEval_docstring(prompt):

    # get previous code and the last function
    previous_code, last_function = find_last_python_function(prompt)
    
    if previous_code is None:
        raise Exception('No function found in the prompt')
    
    # process the last python function
    
    # Find the start of the docstring
    docstring_start_index = last_function.find('"""') + 3
    # Extract the part of the prompt starting from the docstring
    docstring_content = last_function[docstring_start_index:]
    # Find the end of the docstring
    docstring_end_index = docstring_content.find('"""')
    # Extract the docstring content
    docstring = docstring_content[:docstring_end_index]
    # Split the docstring into lines
    lines = docstring.split('\n')
    # Initialize an empty list to hold the lines of the instruction part
    instruction_lines = []
    # Iterate over each line
    for line in lines:
        instruction_lines.append(line)
    # Join the instruction lines back into a single string and return it
    instruction_part = '\n'.join(instruction_lines).strip()
    result = last_function.replace(instruction_part, '')
    
    # remove the lines containing """
    result = result.replace('"""', '')
    # remove the empty lines
    lines2 = result.split('\n')
    result_lines = []
    for line in lines2:
        if line.strip() != '':
            result_lines.append(line)
    
    result = '\n'.join(result_lines)
    
    result = previous_code + result
    
    return result

def is_line_indented(line):
    # Check if the line starts with spaces or a tab
    if line.startswith(' ') or line.startswith('\t') or '\n' in line or line == '':
        return True
    
    # Alternatively, check if stripping leading whitespace changes the length
    original_length = len(line)
    stripped_length = len(line.lstrip())
    
    # If lengths differ, there was leading whitespace
    return original_length != stripped_length

# Given the string of a function to detect if the generated things are repetitive (especially comments).
def has_been_into_generation_loop(ori_prompt, cur_prompt, max_repeat_line) -> bool:
    # if the current prompt is the same as the original prompt, then it is repetitive
    lines_ori = ori_prompt.split('\n')
    lines_cur = cur_prompt.split('\n')

    # keep the lines in lines_cur that are not in lines_ori
    lines_new = [line for line in lines_cur if line not in lines_ori]
    # check if all the last number of max_repeat_line lines are comments (start with #)
    lines_new_last = lines_new[-max_repeat_line:]

    # remove the empty lines
    lines_new_last = [line for line in lines_new_last if line != '' and line != '\n']

    for line in lines_new_last:
        if not line.startswith('#'):
            return False
    return True


def get_code_snippets_deepseek(text, start_substring="```python", end_substring="```"):
    # Regular expression pattern to find text between the substrings
    pattern = f"{start_substring}(.*?){end_substring}"
    # Finding all occurrences
    matches = re.findall(pattern, text, re.DOTALL)
    
    # there should be 1 python code snippet region
    if len(matches) != 1:
        raise Exception(f"Expected 1 code snippet, but found {len(matches)}")
    
    code = matches[0].strip()
    
    return code

# Given the string of a function to detect the function is completed writting. In other words, there is more than 1 line that is not indented.
def has_complete_python_function_generation_deepseek(text) -> bool:
    """
    Checks if the given text contains a Python code snippet enclosed between ```python and ```.
    
    Parameters:
    text (str): The text to check.
    
    Returns:
    bool: True if the text contains a Python code snippet, False otherwise.
    """
    
    # implement based on regular expression
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code_snippet = match.group(1).strip()
        return code_snippet != ''
    return False


# Given the string of a function to detect the function is completed writting. In other words, there is more than 1 line that is not indented.
def has_complete_python_function_generation(ori_prompt, cur_prompt) -> bool:
    
    # determine if repetitive '\n' appears at the end of the current prompt
    if cur_prompt.endswith('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'):
        return True
    
    # heuristic: if there are consectutive 10 lines that starts with the same symbol, return true
    lines_ori = ori_prompt.split('\n')
    lines_cur = cur_prompt.split('\n')
    
    # break if the lines in the current prompt is 10 lines more than the lines in the original prompt
    max_line_diff = 50
    if len(lines_cur) > len(lines_ori) + max_line_diff:
        return True
    
    
    # # for codegen
    # if '<|endoftext|>' in cur_prompt:
    #     return True
    
    not_indented_lines = []
    
    lines_ori = ori_prompt.split('\n')
    count_ori = 0
    for line in lines_ori:

        if not is_line_indented(line):
            not_indented_lines.append(line)
            count_ori += 1
    
    lines_cur = cur_prompt.split('\n')
    count_cur = 0
    for line in lines_cur:
        if not is_line_indented(line):
            count_cur += 1
    
    # check if this is mbpp dataset (for completion, no starting code given)
    mbpp_flag = True
    # check if there is any line starts with 'def'
    for line in lines_ori:
        if line.strip().startswith('def'):
            mbpp_flag = False
    
    def_flag = False
    # check if there is any line starts with 'def'
    for line in lines_cur:
        if line.strip().startswith('def'):
            def_flag = True
    
    if mbpp_flag and def_flag:
        if count_cur >= count_ori + 2:
            return True
        else:
            return False

    else:
        if count_cur >= count_ori + 1:
            return True
        else:
            return False


## obsolete
def normal_generate_deepseek_instruct(ori_prompt, model, tokenizer, device, task, max_length=1993, log_path=None):
    
    instruction = "Complete the following python code:\n"
    
    messages=[
        { 'role': 'user', 'content': instruction + ori_prompt}
    ]
    
    temp_input_ids_ori = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    prompt_without_docstring = remove_humanEval_docstring(ori_prompt)
    
    generate_back_prompt = tokenizer.decode(temp_input_ids_ori[0], skip_special_tokens=False)
    
    # remove <｜begin▁of▁sentence｜> at the beginning
    generate_back_prompt = generate_back_prompt.replace('<｜begin▁of▁sentence｜>', '')
    
    # force the generation start with the original prompt
    generate_back_prompt += prompt_without_docstring
    
    # generate back to ids
    input_ids = tokenizer(generate_back_prompt, return_tensors="pt")['input_ids'].to(model.device)
    
    # input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    
    print(prompt, end='', flush=True)    
    if log_path is not None:
        # if the file does not exist, create it
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(prompt)
        else:
            with open(log_path, 'a') as f:
                f.write(prompt)

    cur_prompt = prompt
    pre_prompt = prompt
    
    # only the generated part
    generated_string = ''

    with torch.no_grad():
        for _ in range(max_length - len(input_ids[0])):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            
            pre_input_ids = copy.deepcopy(input_ids)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # get the string of the current prompt
            cur_prompt = tokenizer.decode(input_ids[0])
            


            if next_token == tokenizer.eos_token_id:
                break


            # get new added tokens in the current prompt compared to the previous prompt
            new_tok = cur_prompt[len(pre_prompt):]
            
            
            # update generated_string
            generated_string_pre = generated_string
            generated_string += new_tok

            if task == 'python':
                if has_complete_python_function_generation(prompt_without_docstring, prompt_without_docstring + generated_string):
                    input_ids = pre_input_ids
                    generated_string = generated_string_pre
                    break
            
            print(new_tok, end="", flush=True)

            

            # if log_path is not None, write (append to original content without new line) the new token to the log file
            if log_path is not None:
                # if the file does not exist, create it
                with open(log_path, 'a') as f:
                    f.write(new_tok)

            
            # update the previous prompt
            pre_prompt = cur_prompt
            
    # Decode the tokens to string
    output_sequence = tokenizer.decode(input_ids[0])

    # entire_code = get_code_snippets_deepseek(generated_string)
    
    entire_code = ori_prompt + generated_string
    
    return output_sequence, generated_string, entire_code






# The normal generate method for python generation
def normal_generate(prompt, model, tokenizer, device, task, max_length=1993, log_path=None):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    
    print(prompt, end='', flush=True)    
    if log_path is not None:
        # if the file does not exist, create it
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(prompt)
        else:
            with open(log_path, 'a') as f:
                f.write(prompt)

    cur_prompt = prompt
    pre_prompt = prompt
    
    # only the generated part
    generated_string = ''

    with torch.no_grad():
        for _ in range(max_length - len(input_ids[0])):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # get the string of the current prompt
            cur_prompt = tokenizer.decode(input_ids[0])
            # remove "<s> " at the beginning
            while cur_prompt.startswith('<s> '):
                cur_prompt = cur_prompt[4:]


            if next_token == tokenizer.eos_token_id:
                break

            if task == 'python':
                if has_complete_python_function_generation(prompt, cur_prompt):
                    break
            
            # if has_been_into_generation_loop(pre_prompt, cur_prompt, 10):
            #     break
            
            # get new added tokens in the current prompt compared to the previous prompt
            new_tok = cur_prompt[len(pre_prompt):]
            print(new_tok, end="", flush=True)
            print('-' * 50, end="", flush=True)
            # update generated_string
            generated_string += new_tok

            # if log_path is not None, write (append to original content without new line) the new token to the log file
            if log_path is not None:
                # if the file does not exist, create it
                with open(log_path, 'a') as f:
                    f.write(new_tok)

            
            # update the previous prompt
            pre_prompt = cur_prompt
            


    # Decode the tokens to string
    output_sequence = tokenizer.decode(input_ids[0])

    return output_sequence, generated_string


global sorted_boundary
sorted_boundary = SortedDict()

def sample_top_down_decompose_best_mask_next_idx(prompt, sampling_time, model, mask_tok, top_k, confidence, tokenizer, device, ratio=0.5, sample_beam=1):
    global sorted_boundary
    
    input_ids_ori = tokenizer.encode(prompt, return_tensors="pt").to(device)
    logits_ori = model(input_ids_ori).logits[:, -1, :]  # calculate the original logits to get the shape
    weighted_logit_accumulated = torch.zeros(logits_ori.shape[-1], device=device).unsqueeze(0)
    if sampling_time == 0:
        return torch.argmax(logits_ori, dim=-1).unsqueeze(0)
    
    # empty the sorted_boundary
    sorted_boundary.clear()
    
    # calculate a group num based on sampling time and length of the prompt
    group_num = min(6, math.ceil(len(chunking(prompt, 1)) / sampling_time))
    
    chunk_list = chunking(prompt, group_num)
    # print('chunk_list: ', chunk_list)
    
    # get the range index of full boundary
    mask_start_idx = 0  
    mask_end_idx = len(chunk_list) - 1  # the end index is included in the range
    
    # get mask ids list
    mask_chunk_list = copy.deepcopy(chunk_list)
    mask_chunk_list[mask_start_idx:mask_end_idx+1] = [mask_tok] 
    prompt_masked_init = "".join(mask_chunk_list)
    
    # the base score is the score of masking the full range (the base score should be almost similar to score_init)
    base_score = get_base_score(prompt, model, top_k, tokenizer, device, special_tokens=[mask_tok])
    
    # calculate the score of the initial masked prompt
    # score_init = token_importance(prompt, prompt_masked_init, top_k, model, tokenizer, device)

    score_init = base_score
    
    # creat the element for the first full range boundary
    # # element: {score, left_mask_idx, right_mask_idx, has_decomposed}
    element_id_init = str(mask_start_idx) + '#' + str(mask_end_idx)
    ele = {
            'score': score_init,
            'left_mask_idx': mask_start_idx,
            'right_mask_idx': mask_end_idx,
            'has_decomposed': False,
        }
    sorted_boundary.insert(element_id_init, ele)
    
    
    for i in range(sampling_time):
        element = None
        # get the highest score element that has not been decomposed
        element_id, element = sorted_boundary.get_decomposable_highest_score_element()

        # if there is no decomposable element, break
        if element is None:
            break

        status = decompose_bounary_to_sorted_dict(copy.deepcopy(chunk_list), element_id, element['left_mask_idx'],  element['right_mask_idx'], mask_tok, model, tokenizer, device, top_k, ratio, logits_ori)
    
    # remove the element_id_init from the sorted boundary list
    sorted_boundary.remove(element_id_init)
        
    # return the top sample_beam elements from the sorted boundary list
    best_boundary_ele = sorted_boundary.get_top_k(sample_beam)
    # best_boundary_ele = sorted_boundary.get_sorted_all()
    # best_boundary_ele = sorted_boundary.get_all_not_decomposed()  # get all the elements that have not been decomposed (all leaf element), and prepare adding their influence together
    
    # initialize the logit, with the same shape as the input_ids_ori, but all values are 0
    
    chunk_list_mask = copy.deepcopy(chunk_list)
    

    for ele in best_boundary_ele:
        # get the masked input_id given the boundary
        
        chunk_list_mask[ele['left_mask_idx']:ele['right_mask_idx']+1] = [mask_tok]
        
        prompt_masked = ''.join(chunk_list_mask)
        input_ids_masked = tokenizer.encode(prompt_masked, return_tensors="pt").to(device)
        
        # # decode the input_ids_masked into string and print
        # print('\nMasked prompt:')
        # print(prompt_masked)
        
        # auto weight and get the weighted logit and weight
        weighted_logit, weight = auto_weight(input_ids_ori, input_ids_masked, ele['score'], base_score, confidence, model, tokenizer, device, logits_ori)
        
        if weight > 0:
            weighted_logit_accumulated += weighted_logit * (ele['score'] - base_score)
        else:
            if len(best_boundary_ele) == 1:  # get the best, otherwise, add them up and ignore the negative weitght
                weighted_logit_accumulated += weighted_logit
            # else:
            #     weighted_logit_accumulated += weighted_logit * (ele['score'] - base_score)


    # get the rank of the next token index based on the weighted logit, as a list
    sorted_logit_next_id_list = torch.argsort(weighted_logit_accumulated, dim=-1, descending=True)[0].tolist()
    cnt = 0
    next_token_idx = sorted_logit_next_id_list[cnt]

    # if the next token idx corresponds to a special token, get the token ranks lower to this token
    while next_token_idx in tokenizer.all_special_ids:
        cnt += 1
        next_token_idx = sorted_logit_next_id_list[cnt]

    # convert the integer back to tensor
    next_token_idx = torch.tensor([[next_token_idx]]).to(device)

    return next_token_idx

        
    # # get the next token
    # next_token_idx = torch.argmax(weighted_logit_accumulated, dim=-1).unsqueeze(0)

# Given a boundary element, decompose it into two sub-boundaries (3/4 left and 3/4 right), calculate the score and update the global sorted boundary list
def decompose_bounary_to_sorted_dict(chunk_list, element_id, mask_start_idx, mask_end_idx, mask_tok, model, tokenizer, device, top_k=10, ratio=0.5, logits_ori=None):
    
    global sorted_boundary
    
    # print('\n')
    print('Decompose', element_id, '... ', end="")
    # print(sorted_boundary.dict)
    # print('decomposing', element_id, '...', sorted_boundary.sorted_keys[::-1])
    
    chunk_list = copy.deepcopy(chunk_list)
    
    # According to the element_id, find the index in the global sorted boundary list
    
    # with in the mask range, get the left and right ratio ranges (dichotomy)
    mask_start_idx_left = mask_start_idx
    mask_end_idx_left = mask_start_idx + round((mask_end_idx - mask_start_idx) * ratio)
    mask_start_idx_right = mask_end_idx - round((mask_end_idx - mask_start_idx) * ratio)
    mask_end_idx_right = mask_end_idx

    # get the left and right input ids by replacing the mask range with the ids of mask_tok
    chunk_list_left = chunk_list[:mask_start_idx_left] + [mask_tok]  + chunk_list[mask_end_idx_left+1:]
    chunk_list_right = chunk_list[:mask_start_idx_right] + [mask_tok]  + chunk_list[mask_end_idx_right+1:]
        
    # update the decomposition status for father element
    sorted_boundary.dict[element_id]['has_decomposed'] = True
    
    # Make new ids for new elements
    boundary_id_left = str(mask_start_idx_left) + '#' + str(mask_end_idx_left)
    boundary_id_right = str(mask_start_idx_right) + '#' + str(mask_end_idx_right)
    
    # print original string
    # print('-' * 50)
    # get the father string by masking using the father element id
    chunk_list_father = chunk_list[:mask_start_idx] + [mask_tok] + chunk_list[mask_end_idx+1:]
    father_masked_string = ''.join(chunk_list_father)
    # print('Father string: ', father_masked_string)
    
    ori_string = ''.join(chunk_list)
    
    
    ########################### left ###########################
    
    # check if the new elements already exist in the global sorted boundary list
    if boundary_id_left in sorted_boundary.dict:
        # print('left already exists')
        return False
    # calculate the importance scores for the left and right input ids
    left_masked_string = ''.join(chunk_list_left)
    # print('left_masked_string: ', left_masked_string)
    score_left = token_importance(ori_string, left_masked_string, top_k, model, tokenizer, device, logits_ori)

    # create new elements (value dict) for left and right
    ele_left = {
                    'score': score_left,
                    'left_mask_idx': mask_start_idx_left,
                    'right_mask_idx': mask_end_idx_left,
                    'has_decomposed': False,
                }
    
    # add the new elements to the global sorted boundary list
    sorted_boundary.insert(boundary_id_left, ele_left)
    
    ########################### right ###########################
    
    # check if the new elements already exist in the global sorted boundary list
    if boundary_id_right in sorted_boundary.dict:
        # print('right already exists')
        return False
    # calculate the importance scores for the left and right input ids
    right_masked_string = ''.join(chunk_list_right)
    # print('right_masked_string: ', right_masked_string)
    score_right = token_importance(ori_string, right_masked_string, top_k, model, tokenizer, device, logits_ori)

    ele_right = {
                    'score': score_right,
                    'left_mask_idx': mask_start_idx_right,
                    'right_mask_idx': mask_end_idx_right,
                    'has_decomposed': False,
                }
    
    # add the new elements to the global sorted boundary list
    sorted_boundary.insert(boundary_id_right, ele_right)
    
    # print('left: ', boundary_id_left)
    # print('right: ', boundary_id_right)

    return True  # return True if the decomposition is successful



# Given a index number, output its corresponding token
def get_token(index, tokenizer):
    return tokenizer.convert_ids_to_tokens(index)

# Given a token, output its corresponding index number
def get_index(token, tokenizer):
    input_ids = tokenizer.encode(token, return_tensors="pt").to('cuda')
    return input_ids[0]

# the method to chunk the prompt
def chunking(prompt, group_num = 6):
    # Pattern explanation:
    # (\s+): Matches any whitespace (space, newline, tab) and captures it
    # (\w+): Matches any word character (letter, digit, underscore) one or more times and captures it as a token
    # ([^\w\s]): Matches any character that is not a word character or whitespace and captures it as a token
    # This pattern captures whitespace separately from words and special characters.
    pattern = r'(\s+)|(\w+)|([^\w\s])'
    
    # Use re.findall to find all occurrences that match the pattern
    matches = re.findall(pattern, prompt)
    
    # Initialize an empty list to hold the processed tokens
    tokens = []
    skip_next = False
    for i, match in enumerate(matches):
        if skip_next:
            skip_next = False
            continue
        token = "".join(match)  # Join the groups to get the token
        if token.isspace() and i < len(matches) - 1:  # If the token is whitespace and not the last one
            next_token = "".join(matches[i + 1])
            tokens.append(token + next_token)  # Attach whitespace to the next token
            skip_next = True  # Skip the next match since it's already included
        elif not token.isspace():  # Only add non-whitespace tokens or last whitespace
            tokens.append(token)
    
    
    # for each token in the list, attached to the previous token (if any)
    attach_to_previous_tokens = [',', '.', '!', '?', ':', ';', ')', ']', '}']
    
    # for each token appeared in the list, attached to the previous token (if any)
    for tok in attach_to_previous_tokens:
        for i, token in enumerate(tokens):
            if token.strip().startswith(tok) and i > 0:
                tokens[i - 1] += token
                tokens[i] = ''
                
    
    # for each token in the list, attached to the next token (if any)
    attach_to_next_tokens = ['(', '[', '{', '#', 'of', 'to', 'in', 'at', 'on', 'def', 'return', 'the']
    for tok in attach_to_next_tokens:
        for i, token in enumerate(tokens):
            if token.strip().endswith(tok) and i < len(tokens) - 1:
                tokens[i] += tokens[i + 1]
                tokens[i + 1] = ''
    
    
    # remove empty tokens
    tokens = list(filter(None, tokens))
    
    # group consecutive n tokens together as a new token
    group_tokens = []
    for i in range(0, len(tokens), group_num):
        group_tokens.append("".join(tokens[i:i + group_num]))
        
        
    return group_tokens

# given a prompt, randomly replace some of the chunks with the masked token, and return the masked prompt
def random_sample(prompt, perturb_num, mask_tok):
    # divide the prompt into chunks
    chunk_list = chunking(prompt)
    mask_list = [0] * len(chunk_list)  # initialize the mask list, 0 represents not masked, 1 represents masked
    
    # randomly select elements in the mask list and set them to 1
    indices = random.sample(range(len(chunk_list)), perturb_num)
    # Set those elements to 1
    for index in indices:
        mask_list[index] = 1

    # replace corresponding chunks with the masked token when index is 1
    new_chunk_list = [mask_tok if mask_list[i] == 1 else chunk_list[i] for i in range(len(chunk_list))]

    masked_prompt = " ".join(new_chunk_list)
    # group consecutive masks
    masked_prompt = group_consecutive_masks(masked_prompt, mask_tok)
    
    return masked_prompt, indices

# given a prompt with currrent masks, and a perturb_num, randomly flip some of the masks, and return the new masked prompt
def random_flip_with_perturb_num(prompt_ori, mask_list, perturb_num, mask_tok):
    # divide the prompt into chunks
    chunk_list = chunking(prompt_ori)
    flip_list = [0] * len(chunk_list)  # initialize the mask list, 0 represents not masked, 1 represents masked
    # randomly select elements in the mask list and set them to 1
    indices = random.sample(range(len(chunk_list)), perturb_num)
    # Set those elements to 1
    for index in indices:
        flip_list[index] = 1

    new_mask_list = copy.deepcopy(mask_list)

    
    # for each 1 in the flip list, flip the corresponding element in mask list (0 -> 1, 1 -> 0)
    for i in range(len(flip_list)):
        if flip_list[i] == 1:
            new_mask_list[i] = 1 - new_mask_list[i] # flip the mask
    
    # print(new_mask_list)
    # replace corresponding chunks with the masked token when index is 1
    new_chunk_list = [mask_tok if new_mask_list[i] == 1 else chunk_list[i] for i in range(len(chunk_list))]

    masked_prompt = " ".join(new_chunk_list)
    # group consecutive masks
    masked_prompt = group_consecutive_masks(masked_prompt, mask_tok)
    
    return masked_prompt, new_mask_list

# some sepcial tokens should not be masked, check it on the index level
def not_mask_tok(tok_idx, tokenizer):
    
    special_tokens = ['<s>', '</s>']
    
    # for each token in the special tokens, convert it to the index using the tokenizer
    special_tokens_idx = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]
    
    if tok_idx in special_tokens_idx:
        return True
    else:
        return False


# given the input ids of prompt with currrent masks, for each chunk, randomly filp the mask, and return the new masked prompt
def random_flip(input_ids_ori, mask_list, mask_tok, tokenizer, prob=0.5):
    # convert the input_ids to a list of token idx
    input_list_ori = input_ids_ori[-1].tolist()
    new_mask_list = copy.deepcopy(mask_list)
    
    for i in range(len(input_list_ori)):
        # jump over the start token
        if not_mask_tok(input_list_ori[i], tokenizer):
            new_mask_list[i] = 0
            continue
        # randomly flip the mask with a probability of 0.5
        prob = 0.5  # option 1: default 0.5
        # prob = 1 - (i + 1 / len(chunk_list))  # option 2: more focus on the beginning, gradually decrease the probability
        prob = prob # option 3: Simulated Annealing: the probability depends on the sampling time, gradually decrease over time
        if random.random() > prob:
            new_mask_list[i] = 1 - new_mask_list[i]
            
    
    # replace corresponding token idx with the idx of mask_token when index is 1
    idx_mask_tok = tokenizer.convert_tokens_to_ids(mask_tok)
    new_masked_input_list = [idx_mask_tok if new_mask_list[i] == 1 else input_list_ori[i] for i in range(len(input_list_ori))]
    
    # convert the list of token idx to input_ids
    new_input_ids_mask = torch.tensor(new_masked_input_list).unsqueeze(0).to(device)
    
    return new_input_ids_mask, new_mask_list

# given a prompt with currrent masks, and a perturb_num, randomly flip some of the masks, and return the new masked prompt
def random_flip_with_perturb_num(prompt_ori, mask_list, perturb_num, mask_tok):
    # divide the prompt into chunks
    chunk_list = chunking(prompt_ori)
    flip_list = [0] * len(chunk_list)  # initialize the mask list, 0 represents not masked, 1 represents masked
    # randomly select elements in the mask list and set them to 1
    indices = random.sample(range(len(chunk_list)), perturb_num)
    # Set those elements to 1
    for index in indices:
        flip_list[index] = 1

    new_mask_list = copy.deepcopy(mask_list)

    
    # for each 1 in the flip list, flip the corresponding element in mask list (0 -> 1, 1 -> 0)
    for i in range(len(flip_list)):
        if flip_list[i] == 1:
            new_mask_list[i] = 1 - new_mask_list[i] # flip the mask
    
    # print(new_mask_list)
    # replace corresponding chunks with the masked token when index is 1
    new_chunk_list = [mask_tok if new_mask_list[i] == 1 else chunk_list[i] for i in range(len(chunk_list))]

    masked_prompt = " ".join(new_chunk_list)
    # group consecutive masks
    masked_prompt = group_consecutive_masks(masked_prompt, mask_tok)
    
    return masked_prompt, new_mask_list


# Given a the prompt, and model, return the sampling time
def get_sampling_time(confidence, max=500, min=0):   
    # the more confident, the less sampling time
    sampling_time = max - confidence * (max - min)
    
    # round result to integer
    sampling_time = round(sampling_time)
    
    print('\nconfidence: ', confidence, end="")
    print('         sampling time: ', sampling_time, end="\n\n")
    
    return sampling_time

# given a prompt, augment generate
def sample_augment_generate(prompt, model, mask_tok, max_length, top_k, max_sample, tokenizer, device, task='python', log_path=None):
    
    print(prompt, end='', flush=True)
    if log_path is not None:
        with open(log_path, 'a') as f:
            f.write(prompt)
    
    cur_prompt = prompt
    pre_prompt = prompt
    input_ids = tokenizer.encode(cur_prompt, return_tensors="pt").to(device)
    
    # only the generated part
    generated_string = ''

    cnt = 0
    with torch.no_grad():
        for _ in range(max_length - len(input_ids[0])):
            
            # get new logits
            outputs_ori = model(input_ids).logits[:, -1, :].to(device)
            # update the confidence
            confidence = get_confidence(outputs_ori)
            # calculate the sampling time
            sampling_time = get_sampling_time(confidence, max=max_sample)

            if sampling_time == 0:
                next_token_idx = torch.argmax(outputs_ori, dim=-1).unsqueeze(0).to(device)
            else:
                # get the idx of the next token
                next_token_idx = sample_accumulate_augment_decode_next_idx(cur_prompt, sampling_time, model, mask_tok, top_k, confidence, tokenizer, device).to(device)
            
            # compare and output the original token and the new next token
            print('\n\nNext token:   `', tokenizer.decode(torch.argmax(outputs_ori, dim=-1)), '`', end="")
            print('  ------>  `', tokenizer.decode(next_token_idx.squeeze(0)), '`')
            
            # make sure input_ids and next_token_idx are on the same device
            input_ids = input_ids.to(device)
            next_token_idx = next_token_idx.to(device)
            
            
            # update the input_ids
            input_ids = torch.cat([input_ids, next_token_idx], dim=-1).to(device)
            
            # print the current prompt
            cur_prompt = tokenizer.decode(input_ids[0])
            # remove "<s> " at the beginning
            while cur_prompt.startswith('<s> '):
                cur_prompt = cur_prompt[4:]
            
            # print('\n\nCurrent Prompt:')
            # print('-'  * 50)
            # print(cur_prompt)
            # print('-'  * 50, end="\n\n")
            
            if next_token_idx[0].item() == tokenizer.eos_token_id: # break if <eos> token is generated
                break
            
            if task == 'python':
                if has_complete_python_function_generation(prompt, cur_prompt):
                    break
                # if has_been_into_generation_loop(pre_prompt, cur_prompt, 10):
                #     break
            
            # get new added tokens in the current prompt compared to the previous prompt
            new_tok = cur_prompt[len(pre_prompt):]
            print(new_tok, end="", flush=True)
            # update generated_string
            generated_string += new_tok
            
            # if log_path is not None, write (append to original content without new line) the new token to the log file
            if log_path is not None:
                # if the file does not exist, create it
                if not os.path.exists(log_path):
                    with open(log_path, 'w') as f:
                        f.write(new_tok)
                else:
                    with open(log_path, 'a') as f:
                        f.write(new_tok)
        
            # update the previous prompt
            pre_prompt = cur_prompt
            
            cnt += 1
            # if cnt > 20:
            #     break

    # return complete and generated part
    return cur_prompt, generated_string

# Given a tensor, return the length of the tensor (the inner part)
def get_tensor_length(tensor):
    return len(tensor[-1].tolist())

# get confidence of the logit, i.e., the max probability of softmax of the logit
def get_confidence(logit):
    softmax = F.softmax(logit, dim=-1)
    result = torch.max(softmax, dim=-1).values
    # convert tensor to float
    result = result.item()
    
    return result

# entrance function of the sampling and augmentation
def sample_accumulate_augment_decode_next_idx(prompt, sampling_time, model, mask_tok, top_k, confidence, tokenizer, device):
    
    # result = sample_accumulate_augment_decode_next_idx_random(prompt, sampling_time, model, mask_tok, top_k, confidence, tokenizer, device)
    # result = sample_augment_decode_next_idx_using_best_masking(prompt, sampling_time, model, mask_tok, top_k, confidence, tokenizer, device)
    result = sample_top_down_decompose_best_mask_next_idx(prompt, sampling_time, model, mask_tok, top_k, confidence, tokenizer, device)
    
    # # Adjust tensor for concatenation
    # result = result.unsqueeze(0)

    return result
    
# return the index of the next token after augmentation
def sample_accumulate_augment_decode_next_idx_random(prompt, sampling_time, model, mask_tok, top_k, tokenizer, device):
    
    # intialize the accumulated rank difference
    accumulated_rank_difference = None

    # generate the input ids
    input_ids_ori = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # generte the logits of the original prompt
    outputs_ori = model(input_ids_ori)
    logits_ori = outputs_ori.logits[:, -1, :]

    # initialize the mask list
    mask_list = [0] * get_tensor_length(input_ids_ori)  # initialize the mask list, 0 represents not masked, 1 represents masked

    # calculate the base score
    base_score = get_base_score(prompt, model, top_k, tokenizer, device, special_tokens=[mask_tok])
    # initialize the current score
    cur_score = base_score

    masked_prompt = prompt


    for i in range(sampling_time):
        
        # calculate how many tokens need to be perturbed
        # simulate annealing
        perturb_num_max = len(prompt.split()) // 2 + 1
        perturb_num = perturb_num_max - (perturb_num_max * i / sampling_time)
        # round to integer
        perturb_num = int(perturb_num)

        # sample the masked prompt
        max_prob = 0.5
        # purturb_prob = max_prob * (1 - (i / sampling_time))  # the probability of flipping the mask for each token (simulated annealing)
        purturb_prob = max_prob * (i / sampling_time)  # the probability of flipping the mask for each token (simulated annealing)
        
        # masked_prompt, mask_list = random_sample(prompt, perturb_num, mask_tok)  # totally random generating masked prompt
        masked_prompt_new, mask_list_new = random_flip(prompt, mask_list, perturb_num, mask_tok, tokenizer, purturb_prob)  # randomly flip some of the masks based on the temperature (perturb_num)
        
        # print(masked_prompt_new)

        # calculate the scoring of new masked prompt, and decide whether to accept it. Update it when the score gets larger
        score = token_importance(prompt, masked_prompt_new, top_k, model, tokenizer, device, logits_ori)
        score_diff = score - base_score
        
        # if score > cur_score:
        if True:
            # print('score: ', score)
            # print('cur_score: ', cur_score)
            cur_score = score
            mask_list = mask_list_new
            masked_prompt = masked_prompt_new
            print(mask_list)
            

            # generate input ids of the masked prompt        
            input_ids_mask = tokenizer.encode(masked_prompt, return_tensors="pt").to(device)

            # generate the logits of the masked prompt
            logits_mask = model(input_ids_mask).logits[:, -1, :]

            control_factor = score_diff * i  # used to control the weight of the rank difference
            control_factor = score
            # calculate the ranking difference
            rank_diff = ranking_difference(logits_ori, logits_mask, control_factor)

            # accumulate the rank difference
            if accumulated_rank_difference is None:
                accumulated_rank_difference = rank_diff
            else:
                accumulated_rank_difference += rank_diff
            
            # choose token based on rank difference
            test_next_token_idx_rankdiff = torch.argmax(rank_diff, dim=-1).unsqueeze(0)
            print('\nrank diff:  ', tokenizer.decode(test_next_token_idx_rankdiff[0]))
            # choose token based on accumulated difference
            test_next_token_idx_accumulated = torch.argmax(accumulated_rank_difference, dim=-1).unsqueeze(0)
            print('accumulated diff:  ', tokenizer.decode(test_next_token_idx_accumulated[0]))
            print('-------------------')
        

    # get the softmax probability of the original logits
    softmax_ori = F.softmax(logits_ori, dim=-1).to(device)

    # convert to float
    accumulated_rank_difference = accumulated_rank_difference.float()
    # get the softmax probability of the accumulated rank difference
    softmax_accumulated_rank_difference = F.softmax(accumulated_rank_difference, dim=-1).to(device)

    # add them together
    # TODO: add a weight lamada to the accumulated rank difference based on confidence of original logits
    # weighted_accumulated_rank_difference = softmax_ori + softmax_accumulated_rank_difference
    weighted_accumulated_rank_difference = softmax_ori * softmax_accumulated_rank_difference
    
    # get the next token
    next_token_idx = torch.argmax(weighted_accumulated_rank_difference, dim=-1).unsqueeze(0)
    
    return next_token_idx
    
    

# return the index of the next token after augmentation
def sample_augment_decode_next_idx_using_best_masking(prompt, sampling_time, model, mask_tok, top_k, confidence, tokenizer, device):

    # generate the input ids
    # all the following operations are based on the input_ids, not at the prompt level
    input_ids_ori = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # generte the logits of the original prompt
    outputs_ori = model(input_ids_ori)
    logits_ori = outputs_ori.logits[:, -1, :].to(device)

    # initialize the mask list
    input_len = get_tensor_length(input_ids_ori)
    mask_list = [0] * input_len  # initialize the mask list, 0 represents not masked, 1 represents masked

    # calculate the base score
    base_score = get_base_score(prompt, model, top_k, tokenizer, device, special_tokens=[mask_tok])
    cur_max_score = base_score

    score = -1  # intialize score 
    
    # initialize the input_ids_mask
    input_ids_mask = copy.deepcopy(input_ids_ori).to(device)  
    
    maksed_prompt = prompt
    
    for i in range(sampling_time):
        
        # # calculate how many tokens need to be perturbed
        # # simulate annealing
        # perturb_num_max = input_len // 2 + 1
        # perturb_num = perturb_num_max - (perturb_num_max * i / sampling_time)
        # # round to integer
        # perturb_num = int(perturb_num)

        # # sample the masked prompt
        # # masked_prompt, mask_list = random_sample(prompt, perturb_num, mask_tok)  # totally random generating masked prompt
        
        
        input_ids_mask_new, mask_list_new = random_flip(input_ids_ori, mask_list, mask_tok, tokenizer)  # randomly flip some of the masks based on the temperature (perturb_num)
        
        # get the masked prompt based on the input_ids_mask
        masked_prompt_new = tokenizer.decode(input_ids_mask_new[0])

        # calculate the scoring of new masked prompt, and decide whether to accept it. Update it when the score gets larger
        score = token_importance(prompt, masked_prompt_new, top_k, model, tokenizer, device)
        
        # Update when the score increases
        if score > cur_max_score:
            cur_max_score = score
            input_ids_mask = input_ids_mask_new
            maksed_prompt = masked_prompt_new
            mask_list = mask_list_new
            print(mask_list)
    
    # generate weighted logits
    # weighted_logit, optimal_a, confidence = auto_weighted_logit(prompt, masked_prompt)
    print('\nMaksed Prompt')
    print('=' * 50)
    print(maksed_prompt)
    print('=' * 50)
    
    if score == -1:
        print('too confident, no need to augment')
        weighted_logit = logits_ori
    else:
        weighted_logit, weight = auto_weight(input_ids_ori, input_ids_mask, cur_max_score, base_score, confidence, model, tokenizer, device, logits_ori)
    
    # get the next token
    next_token_idx = torch.argmax(weighted_logit, dim=-1).unsqueeze(0)
    
    return next_token_idx
    
    
    

# given 2 logits, calculate the weighted ranking difference (multiply by the softmax probability of the second logits)
def ranking_difference(logits_ori, logits_mask, control_factor, device):
    # Ensure the tensors are of the same shape
    if logits_ori.shape != logits_mask.shape:
        raise ValueError("Tensors must be of the same shape")

    # Flatten the tensors
    flat_logits_ori = logits_ori.flatten().to(device)
    flat_logits_mask = logits_mask.flatten().to(device)

    # Calculate ranks (argsort argsort gives ranks)
    ranks_logits_ori = get_logit_ranking(flat_logits_ori).to(device)
    ranks_logits_mask = get_logit_ranking(flat_logits_mask).to(device)

    # calculate the difference, showing how much the ranking has been increased
    rank_difference = (ranks_logits_mask - ranks_logits_ori).to(device)

    # weight this difference by the softmax probability of the ori (only top tokens are considered, reduce nosie)
    softmax_ori = F.softmax(logits_ori, dim=-1).to(device)

    # # calculate the softmax of the rank difference
    # rank_difference = rank_difference.float().to(device)
    # softmax_rank_difference = F.softmax(rank_difference, dim=-1)

    ranking_difference = (rank_difference * softmax_ori).to(device)
    # print('rank diff:  ', ranking_difference)
    # ranking_difference = softmax_rank_difference * softmax_ori
    
    # ranking_difference = rank_difference

    # # change the tensor type to long
    # ranking_difference = ranking_difference.long()

    # rank_difference = flat_logits_ori - flat_logits_mask

    return ranking_difference * control_factor


# given a logit tensor, get the ranking based on the logit
def get_logit_ranking(logits):
    _, indices = torch.sort(logits, descending=True)
    return indices


# Given the original prompt and the current prompt (with new generated tokens), mask the part of the original prompt with <unk>, and return corresponding markers
def only_mask_input(prompt_ori, prompt_cur):
    
    num_ori = len(prompt_ori.split())
    num_cur = len(prompt_cur.split())
    
    # split the prompt into a list of tokens
    markers = ['no'] * num_cur
    
    # replace markers corresponding to original parts with 'yes'
    for i in range(num_ori):
        markers[i] = 'yes'
    
    
    masked_prompt = prompt_cur.replace(prompt_ori, "<unk>")

    return markers, masked_prompt

def auto_augmented_generate_old(prompt, model, tokenizer, device, max_length, top_k=10):
    
    ori_promtp = prompt
    
    '''
    Store the record in a record list, each record is a dictionary with the following keys:

        1. score_diff: score -score_base
        2. weight: the weight used in the current iteration
        3. prompt: the current input string
        4. markers: list of masking
        5. next_token_ori: without augmentation using masking, the original predicted token
        6. next_token_augment: with augmentation, the next predicted token
        7. top_k_tokens_ori: the top K tokens in the original predication
        8. top_k_tokens_augment: the top K tokens in the augmented prediction
        9. top_k_logits_ori: the top K probabilities in the original predication (corresponding to top_k_tokens_ori)
        10. top_k_logits_augment: the top K probabilities in the augmented prediction (corresponding to top_k_tokens_augment)
        11. html_prompt: HTML string of prompt
    '''
    records = []
    
    # # get the masked prompt
    _, markers, score, masked_prompt = find_best_masked_prompt(prompt, model, top_k, tokenizer, device)

    # calculate the base score
    base_score = get_base_score(prompt, model, top_k, tokenizer, device)
    
    # score = 1
    # base_score = 0
    # markers, masked_prompt = only_mask_input(ori_promtp, prompt)
    
    # get the weight
    weight = auto_weight(prompt, masked_prompt, score, base_score, model, tokenizer, device)    
    
    input_ids_ori = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids_mask = tokenizer.encode(masked_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length - len(input_ids_ori[0])):
            
            outputs_ori = model(input_ids_ori)
            outputs_mask = model(input_ids_mask)

            # get the logits of original and masked prompt
            next_token_logits_ori = outputs_ori.logits[:, -1, :]
            next_token_logits_mask = outputs_mask.logits[:, -1, :]
            
            logit_deviation = next_token_logits_ori - next_token_logits_mask
            next_token_logits_augmented = next_token_logits_ori + logit_deviation * weight
            
            # get the next token
            next_token_idx = torch.argmax(next_token_logits_augmented, dim=-1).unsqueeze(0)
            
            if next_token_idx[0].item() == tokenizer.eos_token_id: # break if <eos> token is generated
                break
            
            # prepare for the next iteration
            
            # update the input_ids_ori by concatenating the next token
            input_ids_ori = torch.cat([input_ids_ori, next_token_idx], dim=-1)
            
            pre_prompt = prompt # store the previous prompt
            
            # get back the prompt
            prompt = tokenizer.decode(input_ids_ori[0])
            # remove the <s> at the beginning
            while prompt.startswith("<s>"):
                prompt = prompt[4:]
            
            # automatically calculate the optimal masked prompt
            _, markers, score, masked_prompt = find_best_masked_prompt(prompt, model, top_k, tokenizer, device)
            
            # calculate the base score
            base_score = get_base_score(prompt, model, top_k, tokenizer, device)
            
            # score = 1
            # base_score = 0
            # markers, masked_prompt = only_mask_input(ori_promtp, prompt)
            
            # get the weight
            weight = auto_weight(prompt, masked_prompt, score, base_score, model, tokenizer, device)    
            
            # update the input_ids            
            input_ids_ori = tokenizer.encode(prompt, return_tensors="pt").to(device)
            input_ids_mask = tokenizer.encode(masked_prompt, return_tensors="pt").to(device)
            
            
            top_k_tokens_logits_ori = get_top_k_tokens(next_token_logits_ori, tokenizer, top_k)
            top_k_tokens_logits_augmented = get_top_k_tokens(next_token_logits_augmented, tokenizer, top_k)
            # divide tokens and logits
            top_k_tokens_ori = [token for token, _ in top_k_tokens_logits_ori]
            top_k_tokens_augmented = [token for token, _ in top_k_tokens_logits_augmented]
            top_k_logits_ori = [prob for _, prob in top_k_tokens_logits_ori]
            top_k_logits_augmented = [prob for _, prob in top_k_tokens_logits_augmented]
            
            # store the record
            record = {
                "score_diff": score - base_score,
                "weight": weight,
                "prompt": pre_prompt,
                "markers": markers,
                "next_token_ori": tokenizer.decode(torch.argmax(next_token_logits_ori, dim=-1)),
                "next_token_augment": tokenizer.decode(torch.argmax(next_token_logits_augmented, dim=-1)),
                "top_k_tokens_ori": top_k_tokens_ori,
                "top_k_tokens_augment": top_k_tokens_augmented,
                "top_k_logits_ori": top_k_logits_ori,
                "top_k_logits_augment": top_k_logits_augmented
            }
            records.append(record)
            
            print(record['next_token_augment'], end="")
        
    # Decode the tokens to string
    output_sequence = tokenizer.decode(input_ids_ori[0])
    
    
    # add html prompt in records
    for i in range(len(records)):
        element = records[i]
        if element['score_diff'] > 0:
            html_prompt = highlight_prompt_in_html(element['prompt'], element['markers'])
        else:
            html_prompt = element['prompt']

        # add arrow and next token
        if element['next_token_ori'] != element['next_token_augment']:
            html_prompt += "&nbsp;&nbsp;&nbsp; <span> &#8594; </span>" + '<span style="background-color: #cb824d;">' + element['next_token_ori'] + '</span>' + '&nbsp; <span style="background-color: #61e596;"><b>' + element['next_token_augment'] + '</b></span>'
        else:
            html_prompt += "&nbsp;&nbsp;&nbsp; <span> &#8594; </span>" + '<span style="background-color: #61e596;"><b>' + element['next_token_ori'] + '</b></span>'  
        
        # add this key to the element
        records[i]['html_prompt'] = html_prompt
        
        # delete markers
        del records[i]['markers']
    
    return output_sequence, records

def auto_weight(input_ids_ori, input_id_mask, score, base_score, confidence, model, tokenizer, device, logits_ori=None, max_weight=2):
    
    weight = (score - base_score) / base_score
    weight *= (1-confidence)  # the more confident of original logit, the less weight of new added logit
    
    weight_control_factor = 1.6
    weight *= weight_control_factor # used to magnify the singal
    
    # the weight cannot exceed the max_weight
    if weight > max_weight:
        weight = max_weight
    
    print('\nscore: ', score)
    print('base_score: ', base_score)
    print('weight: ', weight)
    print('')
    
    # generate the logits using model
    if logits_ori is None:
        next_token_logits_ori = model(input_ids_ori).logits[:, -1, :].to(device)
    else:
        next_token_logits_ori = copy.deepcopy(logits_ori).to(device)
    # print('input_ids_ori: ', input_ids_ori)
    # print('next_token_logits_ori: ', next_token_logits_ori)
    
    # generate the logits using model
    next_token_logits_mask = model(input_id_mask).logits[:, -1, :].to(device)
    
    # calculate their deviation
    logit_deviation = next_token_logits_ori - next_token_logits_mask
    
    # only work when weight is greater than 0 (because we can assume the masking is not optimal)
    if weight > 0:
        weighted_logit = next_token_logits_ori + weight * logit_deviation
    else:
        weighted_logit = next_token_logits_ori
    
    return weighted_logit, weight



def auto_weighted_logit(prompt, masked_prompt):
    def softmax(z):
        """Compute softmax values for each set of scores in z."""
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def objective_function(a, X, Y):
        """Objective function to maximize the top-1 value of softmax of (1-a)X + aY."""
        Z = a * X + (1 - a) * Y
        
        Z = z_score(Z)  # normalize the logits using Z-score
        
        # get the next token
        next_token_idx = torch.argmax(Z, dim=-1).unsqueeze(0)
        # print('next token: ', tokenizer.decode(next_token_idx[0]))
        
        confidence = F.softmax(Z, dim=-1).max().item()
        return confidence
    
    

    
    # generate the logits using model
    next_token_logits_ori = model(tokenizer.encode(prompt, return_tensors="pt").to(device)).logits[:, -1, :].to(device)
    # convert it to np.array
    # logits_array = logits.cpu().detach().numpy().flatten()
    
    # generate the logits using model
    next_token_logits_mask = model(tokenizer.encode(masked_prompt, return_tensors="pt").to(device)).logits[:, -1, :].to(device)
    # print(masked_prompt)
    # print('-----------------')
    
    # calculate their deviation
    logit_deviation = next_token_logits_ori - next_token_logits_mask
    
    # convert it to np.array
    # logits_deviation_array = logit_deviation.cpu().detach().numpy().flatten()
    
    precision = 0.001
    
    # in the range of [0, 1], using the precision, sample the value and search for the one with highest obective function
    optimal_a = 1
    max_confidence = 0  # initialize the max confidence
    for i in range(int(1 / precision) + 1):
        a = i * precision
        temp_confidence = objective_function(a, next_token_logits_ori, logit_deviation)
        # print('temp a: ', a)
        # print('confidence: ', temp_confidence)
        # print('max confidence: ', max_confidence)
        
        if temp_confidence > max_confidence:
            max_confidence = temp_confidence
            optimal_a = a

    
    weighted_logit = optimal_a * next_token_logits_ori + (1 - optimal_a) * logit_deviation
    
    # # convert numpy array to torch tensor
    # weighted_logit = torch.tensor(weighted_logit).to(device)
    
    # get the original confidence and the new confidence
    confidence_ori = objective_function(1, next_token_logits_ori, next_token_logits_ori)
    
    print('\nconfidence: ', confidence_ori, end="")
    print('   --->    ', max_confidence)
    
    return weighted_logit, optimal_a, max_confidence

# Given the original prompt, the masked prompt, calculate the best proportion of the two logits such that the top-1 has the highest probability
def auto_weighted_logit_library(prompt, masked_prompt, score, base_score):
    def softmax(z):
        """Compute softmax values for each set of scores in z."""
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def objective_function(a, X, Y):
        """Objective function to maximize the top-1 value of softmax of (1-a)X + aY."""
        Z = a * X + (1 - a) * Y
        softmax_Z = softmax(Z)
        # Negate the max value as we are using minimize function
        return np.max(softmax_Z)
    
    # generate the logits using model
    logits = model(tokenizer.encode(prompt, return_tensors="pt").to(device)).logits[:, -1, :]
    # convert it to np.array
    logits_array = logits.cpu().detach().numpy().flatten()
    
    # generate the logits using model
    logits_mask = model(tokenizer.encode(masked_prompt, return_tensors="pt").to(device)).logits[:, -1, :]
    print(masked_prompt)
    print('-------')
    
    # calculate their deviation
    logit_deviation = logits - logits_mask
    
    # convert it to np.array
    logits_deviation_array = logit_deviation.cpu().detach().numpy().flatten()
    
    # Initial guess for a
    initial_a = 0.5

    # Bounds for a, ensuring it stays within [0, 1]
    bounds = [(0, 1)]

    # Use scipy.optimize.maximize with bounds
    result = minimize(fun=lambda a: -objective_function(a, logits_array, logits_deviation_array), x0=[initial_a], bounds=bounds, method='L-BFGS-B')
    # result = maximize(fun=objective_function, x0=[initial_a], args=(X, Y), bounds=bounds, method='SLSQP')

    # Optimal value of a within [0, 1]
    optimal_a = result.x[0]
    confidence = -result.fun  # Negate because we minimized the negative of the objective
    optimal_a = 0.1
    weighted_logit = optimal_a * logits_array + (1 - optimal_a) * logits_deviation_array
    
    # convert numpy array to torch tensor
    weighted_logit = torch.tensor(weighted_logit).to(device)
    
    # get the original confidence and the new confidence
    confidence_ori = get_confidence(logits)
    
    print('\nconfidence_ori: ', confidence_ori)
    print('confidence: ', confidence)
    
    return weighted_logit, optimal_a, confidence
    
    
    

def highlight_prompt_in_html(prompt, markers):
    # Initialize an empty list to collect HTML parts
    parts = []
    last_index = 0
    
    tokens = prompt.split()
    
    for token, marker in zip(tokens, markers):
        # Find the start index of the token in the prompt
        start_index = prompt.find(token, last_index)
        
        # Add the text before the token (if any) as non-highlighted
        if start_index > last_index:
            parts.append(prompt[last_index:start_index])
        
        # Add the token, highlighted if marker is 'yes'
        if marker == 'yes':
            parts.append(f"<mark>{token}</mark>")
        else:
            parts.append(token)
        
        # Update the last_index to the end of the current token
        last_index = start_index + len(token)
    
    # Add any remaining text after the last token
    if last_index < len(prompt):
        parts.append(prompt[last_index:])
    
    # Join all parts into a single HTML string
    html_output = ''.join(parts)
    
    # # Display the result as HTML
    # display(HTML(html_output))
    
    return html_output


# Given a prompt, if there are multiple consecutive <unk> in the prompt, group them together
def group_consecutive_masks(prompt, mask_tok):
    
    # replace the consecutive <unk>s with a single <unk>
    new_prompt = re.sub(r"({} )+".format(mask_tok), "{} ".format(mask_tok), prompt)
    
    return new_prompt

# Get the base score of token importance
def get_base_score(prompt, model, top_k, tokenizer, device, special_tokens):

    if '' not in special_tokens:
        special_tokens.append('')
    
    total_score = 0
    
    for token in special_tokens:
        temp_score = token_importance(prompt, token, top_k, model, tokenizer, device)
        total_score += temp_score
    
    result = total_score / len(special_tokens)
    return result


# Given a prompt, enumerate all the possible masked prompts by grouping nearby tokens as long as the score increases
def find_best_masked_prompt(prompt, model, top_k, tokenizer, device):

    '''Operation 1: when there is no "go" marker and there is empty marker "". find the new masked token with the highest importance.'''
    # update the tokens and markers by find_next_single_masked_token()
    def operation1(tokens, markers, current_score, model, top_k, tokenizer, device):
        
        # # if all there is no further space, or score improvement, return false
        # def find_next_single_masked_token(prompt, current_score, model, top_k, tokenizer, device):

        prompt_token_list = tokens

        max_score = current_score
        
        result = False
        
        # starting from single token, then expand if the score increases

        for i in range(len(prompt_token_list)):
            if prompt_token_list[i] == "<unk>":
                assert markers[i] != "", "The marker should not be empty when the token is masked"
                continue
            
            masked_prompt_list = copy.deepcopy(prompt_token_list)            
            masked_prompt_list[i] = "<unk>"
            
            # merge consecutive <unk>s
            merge_masked_prompt = group_consecutive_masks(" ".join(masked_prompt_list))
            
            # calculate the score of the masked prompt
            score = token_importance(prompt, merge_masked_prompt, top_k, model, tokenizer, device)
            
            
            # if the score is larger than the base score, add it to the list
            if score > max_score:
                max_score = score
                temp_markers = copy.deepcopy(markers)
                temp_markers[i] = "go"
                # update the current best masked prompt and the index
                result = (masked_prompt_list, temp_markers, score)

        # if there is no improvement, replace all other markers with "no"
        if result == False:
            temp_markers = copy.deepcopy(markers)
            for i in range(len(temp_markers)):
                if temp_markers[i] == "":
                    temp_markers[i] = "no"
                    
            result = (tokens, temp_markers, current_score)
        
         
        return result
        
    
    '''Operation 2: when there is "go" marker in markers, expand the existing "go" marker to the left and right, find the combination of nearby tokens with the highest importance.'''
    def operation2(tokens, markers, current_score, model, top_k, tokenizer, device): 
        
        # find the first "go" marker (if there are multiple "go" markers, only the first one will be expanded)
        go_index = markers.index("go")
        
        # expand the "go" marker to the left and right
        # judge if the left and right are occupied (no space) or there is no left and right token (for the first and last token)
        # left
        if go_index == 0:
            left_have_space = False
        else:
            left_have_space = (markers[go_index - 1] == "")
        # right
        if go_index == len(tokens) - 1:
            right_have_space = False
        else:
            right_have_space = (markers[go_index + 1] == "")
        
        # if both left and right are occupied (no space), convert 'go' to 'yes' in markers
        if not left_have_space and not right_have_space:
            markers[go_index] = "yes"
            return (tokens, markers, current_score)

        # a marker note to record the status of left and right, the key is situation, the value is the corresponding score
        marker_note = {
            'left': 0,
            'right': 0,
            'both': 0
        }
        
        # situation 1: singly expand to the left
        if left_have_space:
            # expand to the right
            left_tokens = copy.deepcopy(tokens)
            # Expand the mask and calculate the score of the new masked prompt
            left_tokens[go_index - 1] = "<unk>"
            # merge consecutive <unk>s
            new_prompt = group_consecutive_masks(" ".join(left_tokens))
            # calculate the score of the masked prompt
            score = token_importance(prompt, new_prompt, top_k, model, tokenizer, device)
            # update the marker note
            marker_note['left'] = score
        
        # situation 2: singly expand to the right
        if right_have_space:
            right_tokens = copy.deepcopy(tokens)
            # Expand the mask and calculate the score of the new masked prompt
            right_tokens[go_index + 1] = "<unk>"
            # merge consecutive <unk>s
            new_prompt = group_consecutive_masks(" ".join(right_tokens))
            # calculate the score of the masked prompt
            score = token_importance(prompt, new_prompt, top_k, model, tokenizer, device)
            # update the marker note
            marker_note['right'] = score
        
        # situation 3: expand to both left and right
        if left_have_space and right_have_space:
            # expand to the right
            both_tokens = copy.deepcopy(tokens)
            
            # Expand the mask and calculate the score of the new masked prompt
            both_tokens[go_index - 1] = "<unk>"
            both_tokens[go_index + 1] = "<unk>"
            # merge consecutive <unk>s
            new_prompt = group_consecutive_masks(" ".join(both_tokens))
            # calculate the score of the masked prompt
            score = token_importance(prompt, new_prompt, top_k, model, tokenizer, device)
            # update the marker note
            marker_note['both'] = score
        
        # find the situation with the highest score
        max_score = max(marker_note.values())
        situation = [key for key, value in marker_note.items() if value == max_score][0]
        
        # compare this score with the current score
        if max_score > current_score:
            if situation == 'left':
                tokens[go_index - 1] = "<unk>"
                markers[go_index - 1] = "go"
                # add 'no' to the right, if there is space
                if right_have_space:
                    markers[go_index + 1] = "no"
                markers[go_index] = "yes" 
            elif situation == 'right':
                tokens[go_index + 1] = "<unk>"
                markers[go_index + 1] = "go"
                # add 'no' to the left, if there is space
                if left_have_space:
                    markers[go_index - 1] = "no"
                markers[go_index] = "yes" 
            elif situation == 'both':
                tokens[go_index - 1] = "<unk>"
                tokens[go_index + 1] = "<unk>"
                markers[go_index - 1] = "go"
                markers[go_index + 1] = "go"
                markers[go_index] = "yes" 
    
        # if the score is not improved, convert 'go' to 'yes' in markers, and fill out 'no' in the left and right
        else:
            markers[go_index] = "yes"
            if left_have_space:
                markers[go_index - 1] = "no"
            if right_have_space:
                markers[go_index + 1] = "no"
            max_score = current_score
            
        return (tokens, markers, max_score)
    
    
    '''A helper function to convert 'go' to 'yes' in markers when both left and right are occupied (no space)'''
    def update_markers_by_yes(markers):
        # for each token, judge if both left and right are occupied (no space), if yes, convert 'go' to 'yes' in markers
        for i in range(len(markers)):
            if markers[i] == 'go':
                if i == 0:
                    if markers[i + 1] != "":
                        markers[i] = 'yes'
                elif i == len(markers) - 1:
                    if markers[i - 1] != "":
                        markers[i] = 'yes'
                else:
                    if markers[i - 1] != "" and markers[i + 1] != "":
                        markers[i] = 'yes'

        return markers
    
    
    # get the base score of token importance (when the mased one is null)
    
    base_score = get_base_score(prompt, model, top_k, tokenizer, device)
    current_score = base_score
    
    tokens = prompt.split()  # This simple split might need adjustment for complex tokenization.
    
    # Initialize markers and scores
    markers = [''] * len(tokens)
    
    
    while '' in markers:
        if 'go' not in markers:
            tokens, markers, current_score = operation1(tokens, markers, current_score, model, top_k, tokenizer, device)
            # print("operation1")
            # print(markers)
        else:
            tokens, markers, current_score = operation2(tokens, markers, current_score, model, top_k, tokenizer, device)
            # print("operation2")
            # print(markers)
            
        # convert 'go' to 'yes' in markers when both left and right are occupied (no space)
        markers = update_markers_by_yes(markers)
    

    # merge consecutive <unk>s
    new_prompt = group_consecutive_masks(" ".join(tokens))
    
    return tokens, markers, current_score, new_prompt

        

# for each masked prompt, calculate the importance, and sort the masked prompts by importance
def get_top_k_masked_prompts_single(prompt, top_k, model, tokenizer, device):
    # get the masked prompt list
    masked_prompt_list = get_masked_prompt_list_single_tok(prompt)
    
    # get the importance of each masked prompt
    importance_list = []
    for masked_prompt in masked_prompt_list:
        importance = token_importance(prompt, masked_prompt, top_k, model, tokenizer, device)
        importance_list.append(importance)
    
    # zip the masked prompt list and importance list
    masked_prompt_importance_list = list(zip(masked_prompt_list, importance_list))
    
    # sort the masked prompt importance list by importance
    masked_prompt_importance_list.sort(key=lambda x: x[1], reverse=True)
    
    # unzip the top k masked prompts and importance
    top_k_masked_prompt_list = []
    top_k_importance_list = []
    for masked_prompt, importance in masked_prompt_importance_list:
        top_k_masked_prompt_list.append(masked_prompt)
        top_k_importance_list.append(importance)
    
    return top_k_masked_prompt_list, top_k_importance_list

# Given a prompt, masking 1 token at a time, enumerate all the possible masked prompts
def get_masked_prompt_list_single_tok(prompt):
    # split the prompt into a list of tokens
    prompt_token_list = prompt.split()
    
    # for each token, replace it with <unk>, and get the masked prompt
    masked_prompt_list = []
    for i in range(len(prompt_token_list)):
        masked_prompt = copy.deepcopy(prompt_token_list)
        masked_prompt[i] = "<unk>"
        masked_prompt = " ".join(masked_prompt)
        masked_prompt_list.append(masked_prompt)

    return masked_prompt_list

# write a function, given 2 prompts, return the quantification of whether the masked token is important
def token_importance(prompt_ori, prompt_mask, top_k, model, tokenizer, device, logits_ori=None):
    
    # remove the <s> at the beginning
    while prompt_ori.startswith("<s>"):
        prompt_ori = prompt_ori[4:]
    while prompt_mask.startswith("<s>"):
        prompt_mask = prompt_mask[4:]

    # strip
    prompt_ori = prompt_ori.strip()
    prompt_mask = prompt_mask.strip()

    
    # get the input_ids of the original prompt
    input_ids_ori = tokenizer.encode(prompt_ori, return_tensors="pt").to(device)

    input_ids_ori = tokenizer.encode(prompt_ori, return_tensors="pt").to(device)
    # get the input_ids of the masked prompt
    input_ids_mask = tokenizer.encode(prompt_mask, return_tensors="pt").to(device)
    
    print('\n\n****** prompt_ori ****** \n', prompt_ori)
    print('****** prompt_mask ****** \n', prompt_mask)

    # print('\n\n****** input_ids_ori: ', input_ids_ori)
    # print('****** input_ids_mask: ', input_ids_mask)
    
    # caculate the importance
    importance = token_importance_ids(input_ids_ori, input_ids_mask, top_k, model, tokenizer, device, logits_ori)
    
    return importance



def token_importance_ids(input_ids_ori, input_ids_mask, top_k, model, tokenizer, device, logits_ori=None):
    
    # remove 
    
    ''' 0. Data preparation'''
    
    # get the logits of the original prompt
    if logits_ori == None:
        logits_ori_raw = model(input_ids_ori).logits[:, -1, :].to(device)
    else:
        logits_ori_raw = copy.deepcopy(logits_ori).to(device)
    
    
    # get the logits of the masked prompt
    logits_mask_raw = model(input_ids_mask).logits[:, -1, :].to(device)
    
    # normalize the logits by z-score
    logits_ori_z = z_score(logits_ori_raw)
    logits_mask_z = z_score(logits_mask_raw)
    
    # calculate the raw delta
    delta_raw = logits_ori_raw - logits_mask_raw
    # calculate the absolute value of raw delta used to amplify the difference
    delta_raw_abs = torch.abs(delta_raw)
    # flattern the delta_raw_abs
    delta_raw_abs = delta_raw_abs.flatten()

    # calculate the softmax of logit
    logits_ori_softmax = F.softmax(logits_ori_z, dim=-1)
    logits_mask_softmax = F.softmax(logits_mask_z, dim=-1)
    
    
    # calculate the difference between logits, the diff is the attention of the focused_word (also the weight unit)
    delta = logits_ori_z - logits_mask_z
    # normalize the delta
    delta_z = z_score(delta)
    
    # calculate the softmax of delta
    delta_softmax = F.softmax(delta_z, dim=-1)
    
    # print("delta_softmax: ", delta_softmax)
    
    ''' 1. Calculate the entropy to quantify randomness'''
    
    # calculate the top-k delta
    delta_softmax_top_k = torch.topk(delta_softmax, k=top_k).values[0].tolist()
    # calculate the entropy of top-k delta
    randomness = calculate_entropy(torch.tensor(delta_softmax_top_k))

    
    # TODO: make entropy starts from 0 by minus the max entropy
    
    
    ''' 2. Calculate the KL divergence of logits_ori_softmax and logits_mask_softmax to quantify changes'''
    
    # retain the common top k of logits_ori_softmax and delta_softmax
    logits_ori_softmax_O_delta_common, delta_softmax_O_delta_common, common_indx_O_delta = retain_common_top_k(logits_ori_softmax, delta_softmax, top_k)
    
    # retrieve the delta_raw_abs corresponding to the common indexes
    delta_raw_abs_common = []
    for index in common_indx_O_delta:
        delta_raw_abs_common.append(delta_raw_abs[index])
    delta_raw_abs_common = torch.tensor(delta_raw_abs_common, dtype=torch.float)  # convert to tensor
    
    # calculate the KL divergence of logits_ori_softmax and logits_mask_softmax (common)
    
    # change = kl_divergence_modified(logits_ori_softmax_O_delta_common, delta_softmax_O_delta_common, delta_raw_abs_common)
    change = kl_divergence(logits_ori_softmax_O_delta_common, delta_softmax_O_delta_common)
    
    ''' 3. Calculate the weighted rank difference to quantify the quality of probability changes'''
    # retain the common top k of logits_ori_softmax and delta_softmax
    logits_ori_z_O_delta_common, delta_z_O_delta_common, _ = retain_common_top_k(logits_ori_z, delta_z, top_k)
    
    # calculate the weighted rank difference for top k
    weighted_rank_difference_top_k = weighted_rank_difference(logits_ori_z_O_delta_common, delta_z_O_delta_common)
    
    # calculate the weighted rank difference for all
    weighted_rank_difference_all = weighted_rank_difference(logits_ori_z, delta_z)
    
    # if the top k has no more difference, it indicates total randomness, no need to calculate, just return -1 to show this is not good
    if weighted_rank_difference_top_k < weighted_rank_difference_all:
        return -1
    else:
        quality = weighted_rank_difference_top_k - weighted_rank_difference_all

    # convert tensor to float
    change = float(change)
    quality = float(quality)
    randomness = float(randomness)

    ''' 4. Finally, calculate the importance'''
    
    importance = change * quality / randomness
    importance = change * quality
    
    print("\n\nrandomness: ", randomness)
    print("change: ", change)
    print("quality: ", quality)
    print("\n-------score: ", importance)
    
    return importance


# modified version of KL divergence
def kl_divergence_modified(p, q, delta_abs):
    """ Calculate the KL Divergence between distributions p and q """
    return np.sum(np.where(p != 0, p * delta_abs * np.log(p / q), 0))

# Calculate KL Divergence between two distributions
def kl_divergence(p, q):
    """ Calculate the KL Divergence between distributions p and q """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# given 2 tensor with the same shape (shape: [1, N]), only retain the values corresponding to common top k indexes
def retain_common_top_k(tensor_1, tensor_2, top_k):
    # find common top k indexes
    top_k_index_set = find_common_top_k_index(tensor_1, tensor_2, top_k)
    
    # retain the values corresponding to the common top k indexes
    tensor_1_top_k = []
    tensor_2_top_k = []
    for index in top_k_index_set:
        tensor_1_top_k.append(tensor_1[0][index])
        tensor_2_top_k.append(tensor_2[0][index])
    
    # convert to tensors
    tensor_1_top_k = torch.tensor(tensor_1_top_k, dtype=torch.float)
    tensor_2_top_k = torch.tensor(tensor_2_top_k, dtype=torch.float)
    
    return tensor_1_top_k, tensor_2_top_k, top_k_index_set

# given 2 tensor with the same shape (shape: [1, N]), find the index of top-k values in tensor1, then find the index of top-k values in tensor2, add the indexes to the same set, then return the common set
def find_common_top_k_index(tensor_1, tensor_2, top_k):
    # get the top k index of tensor 1
    top_k_index_1 = torch.topk(tensor_1, k=top_k).indices[0].tolist()
    # get the top k index of tensor 2
    top_k_index_2 = torch.topk(tensor_2, k=top_k).indices[0].tolist()
    
    # add the 2 sets of indexes together
    top_k_index_set = set(top_k_index_1 + top_k_index_2)
    
    return list(top_k_index_set)

def weighted_rank_difference(tensor1, tensor2):
    # Ensure the tensors are of the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same shape")

    # Flatten the tensors
    flat_tensor1 = tensor1.flatten()
    flat_tensor2 = tensor2.flatten()

    # Calculate ranks (argsort argsort gives ranks)
    ranks_tensor1 = flat_tensor1.argsort().argsort()
    ranks_tensor2 = flat_tensor2.argsort().argsort()

    # Calculate absolute differences in ranks
    rank_differences = torch.abs(ranks_tensor1 - ranks_tensor2).float()
    
    # Calculate weights (average of corresponding values)
    weights = (flat_tensor1 + flat_tensor2) / 2
    
    # higher rank should have higher weight
    # compute the inverse rank, for example, there are N numbers, the rank 1 has the largest value (N), the rank N has the smallest value (1)
    inverse_ranks_1 = len(ranks_tensor1) - ranks_tensor1 + 1
    inverse_ranks_2 = len(ranks_tensor2) - ranks_tensor2 + 1
    
    # add 2 inverse ranks together, and normalized by the length
    rank_weight = (inverse_ranks_1 + inverse_ranks_2) / 2
    
    # Calculate weighted rank differences
    weighted_diff = rank_differences * weights * rank_weight
    
    # normalize the weighted diff, divided by length squared
    weighted_diff = weighted_diff / (len(ranks_tensor1) ** 2)
    
    # Return the average of the weighted rank differences
    return torch.mean(weighted_diff)
    # return weighted_diff

def divide_by_sum(tensor):
    # Sum of all elements in the tensor
    tensor_sum = torch.sum(tensor)

    # Normalize the tensor
    normalized_tensor = tensor / tensor_sum

    return normalized_tensor

def calculate_entropy(logits):
    
    # Apply softmax to convert logits to probabilities
    # probabilities = F.softmax(logits, dim=1)
    
    # make sure there is no 0 in the probabilities, if there is, replace it with a extremely small number
    logits[logits == 0] = 1e-10
    probabilities = divide_by_sum(logits)
    
    
    
    # Calculate the entropy
    log_probabilities = torch.log(probabilities)
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)

    return entropy

def my_normalize(tensor):
    # try different normalization methods
    
    # normalized_tensor = min_max_normalize(tensor)
    normalized_tensor = tensor

    return normalized_tensor

def min_max_normalize(tensor):
    # Ensure the tensor is a float type for accurate calculations
    tensor = tensor.float()

    # Calculate the minimum and maximum values
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Perform Min-Max normalization
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    return normalized_tensor

def z_score(tensor):
    # Ensure the tensor is a float type for accurate calculations
    tensor = tensor.float()

    # Calculate the mean and standard deviation
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    
    # make sure standard deviation is not 0
    if std == 0:
        std = 1e-10

    # Perform Z-score normalization
    normalized_tensor = (tensor - mean) / std

    return normalized_tensor

# Given am model, tokenizer, prompt, and masked prompt, output the logits of the masked words
def get_deviation(model, tokenizer, prompt, masked_prompt, device):
    # get the logits of the original prompt
    input_ids_ori = tokenizer.encode(prompt, return_tensors="pt").to(device)
    logits_ori = model(input_ids_ori).logits[:, -1, :].to(device)
    
    
    # get the logits of the masked prompt
    input_ids_mask = tokenizer.encode(masked_prompt, return_tensors="pt").to(device)
    logits_mask = model(input_ids_mask).logits[:, -1, :].to(device)
    
    
    # calculate the difference between logits, the diff is the attention of the focused_word (also the weight unit)
    deviation = logits_ori - logits_mask
    
    # # apply softmax to the deviation
    # deviation = F.softmax(deviation, dim=-1) 
    
    return deviation


# Given a tensor, calculate the normalized LSE of the tensor
def get_normal_LSE(tensor):
    if float(tensor.shape[1]) > 1:
        return float(torch.sum(torch.exp(tensor))) / float(tensor.shape[1])
    else:
        # throw an exception
        raise Exception("The length of the tensor is not larger than 1")


# Given a tokenizer, a logits, return the top k token ids, corresponding token, and their logits
def get_top_k_tokens(logits, tokenizer, k=10):
    # get the top k token ids, and convert tensor to a list of floats
    top_k_token_ids = torch.topk(logits, k=k).indices[0].tolist()
    # get the top k token
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_token_ids)
    # get the top k logits, and convert tensor to a list of floats
    top_k_logits = torch.topk(logits, k=k).values[0].tolist()
    # apply softmax to the logits
    # top_k_probs = F.softmax(torch.tensor(top_k_logits), dim=-1).tolist()
    
    
    # zip them in (id, token, logits, probs)
    ranked_result = list(zip(top_k_tokens, top_k_logits))
    # ranked_result = list(zip(top_k_token_ids, top_k_tokens, top_k_logits, top_k_probs))
    
    return ranked_result

def get_weighted_masked_prompt_list(prompt):
    # Pattern to find formatted tokens with the updated format <[token](weight)>
    token_pattern = re.compile(r"<\[(.*?)\]\((.*?)\)>")

    # Extract formatted tokens and their weights
    formatted_tokens = token_pattern.findall(prompt)

    results = []
    for i, (token, weight) in enumerate(formatted_tokens):
        # Create a copy of the prompt for each token to mask it individually
        masked_prompt = prompt
        for j, (other_token, _) in enumerate(formatted_tokens):
            # Replace the formatted token with '<unk>' if it's the current one, or just the token itself if it's not
            if i == j:
                masked_prompt = re.sub(r"<\[" + re.escape(other_token) + r"\]\(.*?\)>", "<unk>", masked_prompt)
            else:
                masked_prompt = re.sub(r"<\[" + re.escape(other_token) + r"\]\(.*?\)>", other_token, masked_prompt)

        # Append the modified prompt and weight to the results
        results.append((masked_prompt.strip(), float(weight)))

    return results

def get_original_prompt(formatted_prompt):
    # Pattern to find formatted tokens with the format <[token](weight)>
    token_pattern = re.compile(r"<\[(.*?)\]\((.*?)\)>")

    # Replace formatted tokens with just the token itself
    cleaned_prompt = re.sub(token_pattern, r"\1", formatted_prompt)

    return cleaned_prompt.strip()



def build_instruct_instruction(question: str, benchmark: str, language: str):
    
    if language == 'python':
        if benchmark == 'humaneval':
            return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```python
{}
```
'''.strip().format(question.strip())

        elif benchmark == 'mbpp':
            return question
        else:
            raise ValueError("The benchmark is not supported")


# given the input ids and ouptut by LLM, return a list including the attention score over the the ids
# by default, use the last layer as self-attention
def get_self_attention(input_ids, outputs, layer_num=-1):
    
    attention_weights = outputs.attentions[layer_num]
    # attention_weights shape is [batch_size, num_heads, sequence_length, sequence_length]

    # We want attention from all heads, averaged, focusing on the last input token
    # The last input token often represents the culmination of the entire input sequence
    token_attention = attention_weights[0, :, -1, :].mean(0).detach().cpu().numpy()
    
    # Normalize the attention weights
    token_attention = token_attention / token_attention.sum()
    
    return token_attention

# given the input ids and ouptut by LLM, return a list including the attention score over the the ids
# The attention is gradient-based attention
def get_gradient_attention(input_ids, model, topk = 100):
    
    # def check_grad_status(tensor, message):
    #     print(f"{message} - requires_grad: {tensor.requires_grad}, grad_fn: {tensor.grad_fn is not None}")


    # Disable gradient calculation on all parameters, and manually enable for embeddings
    model.requires_grad_(False)

    # We extract embeddings and manually set requires_grad to True
    inputs_embeds = model.get_input_embeddings()(input_ids).to(model.device)
    
    inputs_embeds.requires_grad = True

    # Forward pass using embeddings directly
    outputs = model(inputs_embeds=inputs_embeds)

    
    # Get the logits for the last token in the sequence and identify the index of the max logit
    last_token_logits = outputs.logits[:, -1, :].to(model.device)

    # Get the indices of the topk logits
    _, top_indices = torch.topk(last_token_logits, topk)

    # Create a new tensor to hold the top logits and mask others with zero
    top_logits = torch.zeros_like(last_token_logits).to(model.device)
    top_logits.scatter_(1, top_indices, last_token_logits.gather(1, top_indices))

    # Compute gradients for the top logits sum (top-10)
    top_logits.sum().backward()

    # Get the gradients of the embeddings
    grads = inputs_embeds.grad

    # Move gradients to CPU and calculate the norm of the gradients for each token to get the saliency map
    token_grads = torch.norm(grads.squeeze(), dim=1).cpu().detach().numpy()
    token_grads = token_grads / token_grads.sum()

    
    return token_grads
    
    

# Given a prompt and a model, mask it based on the top self-attention, return the masked ids
def mask_self_attention(input_ids_ori, outputs, mask_token_id, reverse=False):
    layer_num = -1  # define which layer to used as the attention layer
    
    # Fetch attention weights; focus on the last layer's attention to the last input token
    # This is assuming the attention to the next predicted token is based on the entire sequence context
    # print(len(outputs.attentions))
    attention_weights = outputs.attentions[layer_num]
    # attention_weights shape is [batch_size, num_heads, sequence_length, sequence_length]

    # We want attention from all heads, averaged, focusing on the last input token
    # The last input token often represents the culmination of the entire input sequence
    token_attention = attention_weights[0, :, -1, :].mean(0).detach().cpu().numpy()
    
    # Normalize the attention weights
    token_attention = token_attention / token_attention.sum()
    
    # based on the token_attention, decide the token index to mask
    top_k = int(0.35 * len(token_attention))
    # top_k = 10
    
    # Convert the list to a numpy array
    arr = np.array(token_attention)
    # Find the indexes of the top-k highest values
    if not reverse:
        top_k_indexes = np.argsort(arr)[-top_k:][::-1].tolist()
    else:
        top_k_indexes = np.argsort(arr)[:top_k].tolist()
        
    # replace the top k tokens with mask_token_id
    mask_inputs_ids = copy.deepcopy(input_ids_ori)
    for index in top_k_indexes:
        mask_inputs_ids[0][index] = mask_token_id

    
    return mask_inputs_ids



    

# create a funciton used to enhance the logits of the focused word during the decoding process (the generate function)
def augmented_generate_deepseek_self_attention(raw_prompt, model, tokenizer, max_length, device, task, benchmark, mask_tok='<pad>', language='python', weight=0.275, log_path=None, reverse=False):
    
    ori_messages=[
        # {"role": "system", "content": "You are a helpful assistant."},
        {'role': 'user', 'content': build_instruct_instruction(raw_prompt, benchmark, language)}
    ]
    
    input_ids_ori = tokenizer.apply_chat_template(
        ori_messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    
    # get the mask token id
    mask_tok_id = tokenizer.convert_tokens_to_ids(mask_tok)
    

    ori_prompt = tokenizer.decode(input_ids_ori[0], skip_special_tokens=False)
    cur_prompt = ori_prompt
    pre_prompt = ori_prompt
    
    # print(ori_prompt, end="", flush=True)
    if log_path is not None:
        # if the file does not exist, create it
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(ori_prompt)
        else:
            with open(log_path, 'a') as f:
                f.write(ori_prompt)
    
    # only the generated part
    generated_string = ''
    
    with torch.no_grad():
        for _ in range(max_length - len(input_ids_ori[0])):
            outputs_ori = model(input_ids_ori, output_attentions=True)
            
            # get the masked input ids based on the self-attention

            input_ids_mask = mask_self_attention(input_ids_ori, outputs_ori, mask_tok_id, reverse)

            # get the outputs of the masked prompt
            outputs_mask = model(input_ids_mask, output_attentions=True)

            # get the logits of original and masked prompt
            next_token_logits_ori = outputs_ori.logits[:, -1, :]
            next_token_logits_mask = outputs_mask.logits[:, -1, :]
            
            logit_deviation = next_token_logits_ori - next_token_logits_mask
            next_token_logits_augmented = next_token_logits_ori + logit_deviation * weight

            # Use Z-score to normalize the logits
            next_token_logits_augmented = z_score(next_token_logits_augmented)
            
            # get the next token
            next_token_idx = torch.argmax(next_token_logits_augmented, dim=-1).unsqueeze(0)
            
            # convert this logit to softmax probability, and return this highest probability (as confidence)
            confidence = F.softmax(next_token_logits_augmented, dim=-1).max().item()

            # update the input_ids
            pre_input_ids_ori = copy.deepcopy(input_ids_ori)
            input_ids_ori = torch.cat([input_ids_ori, next_token_idx], dim=-1).to(device)

            # print the current prompt
            cur_prompt = tokenizer.decode(input_ids_ori[0])

            
            if next_token_idx[0].item() == tokenizer.eos_token_id: # break if <eos> token is generated
                break
            
            # get new added tokens in the current prompt compared to the previous prompt
            new_tok = cur_prompt[len(pre_prompt):]
            
            # update generated_string
            generated_string_pre = generated_string
            generated_string += new_tok
            
            
            if task == 'python':
                # if has_complete_python_function_generation(prompt_without_docstring, prompt_without_docstring + generated_string):
                if has_complete_python_function_generation_deepseek(generated_string):
                    input_ids_ori = pre_input_ids_ori
                    generated_string = generated_string_pre
                    break
            
            print(new_tok, end="", flush=True)
            
            # if log_path is not None, write (append to original content without new line) the new token to the log file
            if log_path is not None:
                with open(log_path, 'a') as f:
                    f.write(new_tok)

            # update the previous prompt
            pre_prompt = cur_prompt
    
    # Decode the tokens to string
    output_sequence = tokenizer.decode(input_ids_ori[0])
    
    
    # get the code between ```python and ```
    try:
        entire_code = get_code_snippets_deepseek(generated_string)
    except:
        entire_code = generated_string

    return output_sequence, '', entire_code

# create a funciton used to enhance the logits of the focused word during the decoding process (the generate function)
# the weight is based on the confidence
def augmented_generate_deepseek_instruct_confidence(raw_prompt, raw_masked_prompt, model, tokenizer, max_length, device, task, benchmark, language='python', output_attentions = False, weight=0.275, log_path=None):
    
    ori_messages=[
        { 'role': 'user', 'content': build_instruct_instruction(raw_prompt, benchmark, language)}
    ]
    
    input_ids_ori = tokenizer.apply_chat_template(ori_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    masked_messages=[
        { 'role': 'user', 'content': build_instruct_instruction(raw_masked_prompt, benchmark, language)}
    ]
    input_ids_mask = tokenizer.apply_chat_template(masked_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    ori_prompt = tokenizer.decode(input_ids_ori[0], skip_special_tokens=False)
    
    cur_prompt = ori_prompt
    pre_prompt = ori_prompt
    
    # print(ori_prompt, end="", flush=True)
    if log_path is not None:
        # if the file does not exist, create it
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(ori_prompt)
        else:
            with open(log_path, 'a') as f:
                f.write(ori_prompt)
    
    # only the generated part
    generated_string = ''
    
    with torch.no_grad():
        for _ in range(max_length - len(input_ids_ori[0])):
            outputs_ori = model(input_ids_ori, output_attentions=output_attentions)
            outputs_mask = model(input_ids_mask, output_attentions=output_attentions)

            # get the logits of original and masked prompt
            next_token_logits_ori = outputs_ori.logits[:, -1, :]
            next_token_logits_mask = outputs_mask.logits[:, -1, :]
            
            # Get the confidence of the original decoding
            confidence = F.softmax(next_token_logits_ori, dim=-1).max().item()
            
            # temp_weight = weight * (1-confidence)
            temp_weight = weight * confidence # reversed confidence
            
            logit_deviation = next_token_logits_ori - next_token_logits_mask
            next_token_logits_augmented = next_token_logits_ori + logit_deviation * temp_weight
            # next_token_logits_augmented = next_token_logits_augmented / (weight + 1)
            # Use Z-score to normalize the logits
            next_token_logits_augmented = z_score(next_token_logits_augmented)
            
            # get the next token
            next_token_idx = torch.argmax(next_token_logits_augmented, dim=-1).unsqueeze(0)
      
            # update the input_ids
            pre_input_ids_ori = copy.deepcopy(input_ids_ori)
            input_ids_ori = torch.cat([input_ids_ori, next_token_idx], dim=-1).to(device)
            input_ids_mask = torch.cat([input_ids_mask, next_token_idx], dim=-1).to(device)
            
            
            # print the current prompt
            cur_prompt = tokenizer.decode(input_ids_ori[0])

            
            if next_token_idx[0].item() == tokenizer.eos_token_id: # break if <eos> token is generated
                break
            
            # get new added tokens in the current prompt compared to the previous prompt
            new_tok = cur_prompt[len(pre_prompt):]
            
            
            # update generated_string
            generated_string_pre = generated_string
            generated_string += new_tok
            
            
            if task == 'python':
                # if has_complete_python_function_generation(prompt_without_docstring, prompt_without_docstring + generated_string):
                if has_complete_python_function_generation_deepseek(generated_string):
                    input_ids_ori = pre_input_ids_ori
                    generated_string = generated_string_pre
                    break
            
            print(new_tok, end="", flush=True)
            
            # if log_path is not None, write (append to original content without new line) the new token to the log file
            if log_path is not None:
                with open(log_path, 'a') as f:
                    f.write(new_tok)

            # update the previous prompt
            pre_prompt = cur_prompt
    
    # Decode the tokens to string
    output_sequence = tokenizer.decode(input_ids_ori[0])
    
    # entire_code = get_code_snippets_deepseek(generated_string)
    
    # entire_code = raw_prompt + generated_string
    
    # get the code between ```python and ```
    try:
        entire_code = get_code_snippets_deepseek(generated_string)
    except:
        entire_code = generated_string

    return output_sequence, '', entire_code

# Given 2 list, assume multiple ranges of list 1 are replaced with a different element, find the range of the replaced element of list 1
def find_single_masked_index_range(list1, list2):
    print(list1)
    print(list2)
    
    # Find the first index where the lists differ
    first_diff_index = next((i for i, (x, y) in enumerate(zip(list1, list2)) if x != y), None)
    if first_diff_index is None:
        return None

    # Find the last index where the lists differ
    last_diff_index = next((i for i, (x, y) in enumerate(zip(reversed(list1), reversed(list2))) if x != y), None)
    if last_diff_index is None:
        return None

    # Calculate the range of the replaced element
    start_index = first_diff_index
    end_index = len(list1) - last_diff_index
    return start_index, end_index

# Given a prompt (string) and the checkpoint name, return the 
def prompt_to_inputs(prompt, tokenizer, checkpoint_name, benchmark, language, device):
    instruct_names = ['deepseek-ai/deepseek-coder-6.7b-instruct', 'deepseek-ai/deepseek-coder-1.3b-instruct', 'deepseek-ai/deepseek-coder-33b-instruct', 'Qwen/CodeQwen1.5-7B-Chat']
    base_names = ['Salesforce/codegen-350M-mono', 'codellama/CodeLlama-34b-hf', 'codellama/CodeLlama-7b-hf', 'codellama/CodeLlama-7b-Python-hf']
    
    # instruct-tuned models
    if 'Qwen' in checkpoint_name:
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            { 'role': 'user', 'content': build_instruct_instruction(prompt, benchmark, language)}
        ]
        
        template_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        model_inputs = tokenizer([template_text], return_tensors="pt").to(device)
        input_ids = model_inputs['input_ids']
    
    elif checkpoint_name in instruct_names:
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            { 'role': 'user', 'content': build_instruct_instruction(prompt, benchmark, language)}
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
        
    # base models
    elif checkpoint_name in base_names:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    else:
        raise ValueError("The checkpoint name is not supported")
    
    return input_ids

# a condition function used to determine whether the code generation process should stop
def stop_condition(generated_string, ori_prompt, language, benchmark, checkpoint_name):
    cur_prompt = ori_prompt + generated_string
    
    # determine whether there are repetitive tokens appear at the end
    # implement the function here

    if language == 'python':
        # if has_complete_python_function_generation(prompt_without_docstring, prompt_without_docstring + generated_string):
        if 'deepseek' in checkpoint_name or 'qwen' in checkpoint_name.lower(): 
            if has_complete_python_function_generation_deepseek(generated_string):
                return True

        elif 'codellama' in checkpoint_name or 'codegen' in checkpoint_name:
            if has_complete_python_function_generation(ori_prompt, cur_prompt):
                return True
            # pass
        
        
        else:
            raise Exception("The stop condition for this model is not supported yet")
    else:
        raise Exception(f"The stop method for {language} is not supported yet")
    
    return False


@contextlib.contextmanager
def conditional_no_grad(condition):
    if condition:
        yield
    else:
        with torch.no_grad():
            yield
        

def other_stop_condition(new_tok):
    if new_tok == '<｜begin▁of▁sentence｜>':
        return True
    
    return False




# temperature greedy decoding
def greedy_search_with_temperature(
    raw_prompt,
    model,
    tokenizer,
    max_length,
    device,
    num_candidates=10,
    temperature=1.2,
    top_p=0.9
):
    """
    Generate code using greedy search with temperature for pass@10 calculation.

    This function generates multiple code candidates using a greedy search approach
    with temperature scaling and nucleus (top-p) sampling. It's designed to produce
    diverse outputs for pass@10 evaluation in code generation tasks.

    Args:
        raw_prompt (str): The input prompt to guide code generation.
        model: The language model used for generation.
        tokenizer: The tokenizer corresponding to the model.
        max_length (int): Maximum length of the generated sequence.
        device: The device (CPU or GPU) to run the model on.
        num_candidates (int): Number of code candidates to generate (default is 10 for pass@10).
        temperature (float): Temperature for logits scaling. Higher values increase randomness.
        top_p (float): Cumulative probability threshold for nucleus sampling.

    Returns:
        list: A list of 'num_candidates' generated code snippets for pass@10 calculation.

    Note:
        - Higher temperature values lead to more diverse but potentially less coherent outputs.
        - The top_p parameter helps to filter out low-probability tokens, maintaining output quality.
    """
    print(f"Starting greedy search with temperature for {num_candidates} candidates")
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print(f"Maximum sequence length: {max_length}")
    
    model.eval()
    results = []
    total_start_time = time.time()

    for candidate in range(num_candidates):
        start_time = time.time()
        print(f"\nGenerating candidate {candidate + 1}/{num_candidates}")
        
        input_ids = tokenizer.encode(raw_prompt, return_tensors="pt").to(device)
        print(f"Input prompt length: {input_ids.shape[1]} tokens")
        
        with torch.no_grad():
            for step in range(max_length - input_ids.shape[1]):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append the new token to the sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Print progress every 20 steps
                if step % 20 == 0:
                    print(f"  Step {step}: Current length = {input_ids.shape[1]} tokens")
                
                # Check if EOS token is generated
                if next_token.item() == tokenizer.eos_token_id:
                    print(f"  EOS token generated at step {step}")
                    break
        
        # Decode the generated sequence
        generated_string = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Extract code snippet
        try:
            entire_code = get_code_snippets_deepseek(generated_string)
            print("  Successfully extracted code snippet")
        except Exception as e:
            entire_code = generated_string
            print(f"  Failed to extract code snippet: {str(e)}")
        
        results.append(entire_code)
        
        end_time = time.time()
        print(f"Candidate {candidate + 1} generated in {end_time - start_time:.2f} seconds")
        print(f"Generated sequence length: {len(entire_code)} characters")

    total_end_time = time.time()
    print(f"\nGeneration complete. Total time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Average time per candidate: {(total_end_time - total_start_time) / num_candidates:.2f} seconds")

    return results


# MAIN
# create a funciton used to enhance the logits of the focused word during the decoding process (the generate function)
def augmented_generate_anchor(raw_prompt, raw_masked_prompt, checkpoint_name, model, tokenizer, max_length, device, language, benchmark, weight=0.275, output_attentions=False, compute_self_attention=False, compute_gradient=False, adaptive_attention_weight=False, log_path=None, print_code_to_console=True):
    
    # preparation
    model.eval()
    ori_weight = weight  # store the original base weight
    
    # original inputs
    input_ids_ori = prompt_to_inputs(raw_prompt, tokenizer, checkpoint_name, benchmark, language, device).to(device)
    # masked inputs  
    input_ids_mask = prompt_to_inputs(raw_masked_prompt, tokenizer, checkpoint_name, benchmark, language, device).to(device)
    
    # caculate the mask length
    mask_len = len(input_ids_ori[0]) - len(input_ids_mask[0]) + 1
    
    # get the length of input_ids_ori
    init_input_ids_len = len(input_ids_ori[0])
    
    if adaptive_attention_weight:
        output_attentions = True
    
    ori_prompt = tokenizer.decode(input_ids_ori[0], skip_special_tokens=False)
    
    cur_prompt = ori_prompt
    pre_prompt = ori_prompt
    
    if print_code_to_console:
        print(ori_prompt, end="", flush=True)
    
    if log_path is not None:
        # if the file does not exist, create it
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(ori_prompt)
        else:
            with open(log_path, 'a') as f:
                f.write(ori_prompt)
    
    # only the generated part
    generated_string = ''
    
    attention_to_ori_prompt_list_self = []  # used to store the attention to the original prompt (self-attention)
    attention_to_ori_prompt_list_gradient = []  # used to store the attention to the original prompt (gradient attention)
    
    # Initialize KV cache
    past_key_values_ori = None
    past_key_values_mask = None
    
    # if output_attentions is true, with gradient, otherwise, torch.no_grad()
    with conditional_no_grad(output_attentions):
        for _ in range(max_length - len(input_ids_ori[0])):
            # Use KV cache for original prompt
            if past_key_values_ori is None:
                outputs_ori = model(input_ids_ori, output_attentions=output_attentions, use_cache=True)
                next_token_logits_ori = outputs_ori.logits[:, -1, :]
                past_key_values_ori = outputs_ori.past_key_values
            else:
                outputs_ori = model(input_ids_ori[:, -1:], output_attentions=output_attentions, use_cache=True, past_key_values=past_key_values_ori)
                next_token_logits_ori = outputs_ori.logits[:, -1, :]
                past_key_values_ori = outputs_ori.past_key_values
            
            if output_attentions:
                if compute_self_attention:
                    attention_ori_self = get_self_attention(input_ids_ori, outputs_ori)
                    attention_sum_self = sum(attention_ori_self[:init_input_ids_len])
                    if attention_sum_self > 1:
                        attention_sum_self = 1
                    attention_to_ori_prompt_list_self.append(attention_sum_self)
                
                if compute_gradient:
                    attention_ori_gradient = get_gradient_attention(input_ids_ori, model, topk = 100)
                    attention_sum_gradient = sum(attention_ori_gradient[:init_input_ids_len])
                    if attention_sum_gradient > 1:
                        attention_sum_gradient = 1
                    attention_to_ori_prompt_list_gradient.append(attention_sum_gradient)
            
            if ori_weight != 0:
                # Use KV cache for masked prompt
                if past_key_values_mask is None:
                    outputs_mask = model(input_ids_mask, output_attentions=output_attentions, use_cache=True)
                    next_token_logits_mask = outputs_mask.logits[:, -1, :]
                    past_key_values_mask = outputs_mask.past_key_values
                else:
                    outputs_mask = model(input_ids_mask[:, -1:], output_attentions=output_attentions, use_cache=True, past_key_values=past_key_values_mask)
                    next_token_logits_mask = outputs_mask.logits[:, -1, :]
                    past_key_values_mask = outputs_mask.past_key_values
                        
                logit_deviation = (next_token_logits_ori - next_token_logits_mask).detach()
                
                # decide weight based on the attention (if it is adpative) and given weight
                if adaptive_attention_weight:
                    cur_input_ids_len = len(input_ids_ori[0])
                    weight = (cur_input_ids_len - init_input_ids_len) / mask_len * ori_weight
                
                next_token_logits_augmented = (next_token_logits_ori + logit_deviation * weight).detach()
                next_token_logits_augmented = z_score(next_token_logits_augmented).detach()
            
            else:
                next_token_logits_augmented = next_token_logits_ori.detach()
            
            # get the next token
            next_token_idx = torch.argmax(next_token_logits_augmented, dim=-1).unsqueeze(0).detach()
            
            # update the input_ids
            pre_input_ids_ori = copy.deepcopy(input_ids_ori)
            input_ids_ori = torch.cat([input_ids_ori, next_token_idx], dim=-1).to(device)
            input_ids_mask = torch.cat([input_ids_mask, next_token_idx], dim=-1).to(device)
            
            input_ids_ori = input_ids_ori.detach().to(device)
            input_ids_mask = input_ids_mask.detach().to(device)
            
            # print the current prompt
            cur_prompt = tokenizer.decode(input_ids_ori[0])
            
            if next_token_idx[0].item() == tokenizer.eos_token_id:
                break
            
            # get new added tokens in the current prompt compared to the previous prompt
            new_tok = cur_prompt[len(pre_prompt):]
            
            # add special stop condition
            if other_stop_condition(new_tok):
                break
            
            # update generated_string
            generated_string_pre = generated_string
            generated_string += new_tok
            
            # determine if the code generation process should stop
            if stop_condition(generated_string, ori_prompt, language, benchmark, checkpoint_name):
                input_ids_ori = pre_input_ids_ori
                break
            
            if print_code_to_console:
                print(new_tok, end="", flush=True)
            
            if log_path is not None:
                with open(log_path, 'a') as f:
                    f.write(new_tok)

            # update the previous prompt
            pre_prompt = cur_prompt
            
            # clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
    
    # Decode the tokens to string
    output_sequence = tokenizer.decode(input_ids_ori[0])

    # get the code between ```python and ```
    try:
        entire_code = get_code_snippets_deepseek(generated_string)
    except:
        entire_code = generated_string
        
    # add prompt for pure completion models (for humaneval)
    if benchmark == "humaneval":
        if 'codegen' in checkpoint_name.lower() or 'codellama' in checkpoint_name.lower():
            entire_code = raw_prompt + entire_code

    return output_sequence, (attention_to_ori_prompt_list_self, attention_to_ori_prompt_list_gradient), entire_code



# # OLD MAIN
# # create a funciton used to enhance the logits of the focused word during the decoding process (the generate function)
# def augmented_generate_anchor(raw_prompt, raw_masked_prompt, checkpoint_name, model, tokenizer, max_length, device, language, benchmark, weight=0.275, output_attentions=False, compute_self_attention=False, compute_gradient=False, adaptive_attention_weight=False, log_path=None, print_code_to_console=True):
    
#     # preparation
#     model.eval()
#     ori_weight = weight  # store the original base weight
    
#     # original inputs
#     input_ids_ori = prompt_to_inputs(raw_prompt, tokenizer, checkpoint_name, benchmark, language, device).to(device)
#     # masked inputs
#     input_ids_mask = prompt_to_inputs(raw_masked_prompt, tokenizer, checkpoint_name, benchmark, language, device).to(device)
    
#     # caculate the mask length
#     mask_len = len(input_ids_ori[0]) - len(input_ids_mask[0]) + 1
    
#     # get the length of input_ids_ori
#     init_input_ids_len = len(input_ids_ori[0])
#     # for attention calculation of the selected part    
#     # masked_range = find_single_masked_index_range(input_ids_ori[0].tolist(), input_ids_mask[0].tolist())
#     # if masked_range is None:
#     #     raise Exception("The masked range is None")
    
#     if adaptive_attention_weight:
#         output_attentions = True
    
#     ori_prompt = tokenizer.decode(input_ids_ori[0], skip_special_tokens=False)
    
#     cur_prompt = ori_prompt
#     pre_prompt = ori_prompt
    
#     if print_code_to_console:
#         print(ori_prompt, end="", flush=True)
    
#     if log_path is not None:
#         # if the file does not exist, create it
#         if not os.path.exists(log_path):
#             with open(log_path, 'w') as f:
#                 f.write(ori_prompt)
#         else:
#             with open(log_path, 'a') as f:
#                 f.write(ori_prompt)
    
#     # only the generated part
#     generated_string = ''
    
#     attention_to_ori_prompt_list_self = []  # used to store the attention to the original prompt (self-attention)
#     attention_to_ori_prompt_list_gradient = []  # used to store the attention to the original prompt (gradient attention)
    
    
#     # if output_attentions is true, with gradient, otherwise, torch.no_grad()
#     # with torch.no_grad():
#     with conditional_no_grad(output_attentions):
#         for _ in range(max_length - len(input_ids_ori[0])):
#             outputs_ori = model(input_ids_ori, output_attentions=output_attentions)  # inference
#             next_token_logits_ori = outputs_ori.logits[:, -1, :]  # get the logits of original and masked prompt
            
#             if output_attentions:
#                 if compute_self_attention:
#                     attention_ori_self = get_self_attention(input_ids_ori, outputs_ori)
#                     attention_sum_self = sum(attention_ori_self[:init_input_ids_len])  # all the original prompt attentions
#                     # attention_sum = sum(attention_ori[masked_range[0]:masked_range[1]])  # only the attentions of the masked part 
#                     if attention_sum_self > 1:
#                         attention_sum_self = 1
#                     attention_to_ori_prompt_list_self.append(attention_sum_self)
                
#                 if compute_gradient:
#                     attention_ori_gradient = get_gradient_attention(input_ids_ori, model, topk = 100)
#                     attention_sum_gradient = sum(attention_ori_gradient[:init_input_ids_len])  # all the original prompt attentions
#                     if attention_sum_gradient > 1:
#                         attention_sum_gradient = 1
#                     attention_to_ori_prompt_list_gradient.append(attention_sum_gradient)

            
#             if ori_weight != 0:
#                 outputs_mask = model(input_ids_mask, output_attentions=output_attentions)  # inference
#                 next_token_logits_mask = outputs_mask.logits[:, -1, :]  # get the logits of original and masked prompt
                        
#                 logit_deviation = (next_token_logits_ori - next_token_logits_mask).detach()
                
#                 # decide weight based on the attention (if it is adpative) and given weight
#                 if adaptive_attention_weight:
#                     cur_input_ids_len = len(input_ids_ori[0])  # calculate the current input_ids length for adjusting the weight
#                     # weight = (cur_input_ids_len - mask_len) / mask_len * ori_weight
#                     weight = (cur_input_ids_len - init_input_ids_len) / mask_len * ori_weight
                    
#                     # print('\nmask_len: ', mask_len, flush=True)
#                     # print('\ninit_input_ids_len: ', init_input_ids_len, flush=True)
#                     # print('\ncur_input_ids_len: ', cur_input_ids_len, flush=True)
#                     # print('\nweight: ', weight, flush=True)
                
#                 next_token_logits_augmented = (next_token_logits_ori + logit_deviation * weight).detach()
#                 # next_token_logits_augmented = next_token_logits_augmented / (weight + 1)
#                 # Use Z-score to normalize the logits
#                 next_token_logits_augmented = z_score(next_token_logits_augmented).detach()
            
#             else:
#                 next_token_logits_augmented = next_token_logits_ori.detach()
            
#             # get the next token
#             next_token_idx = torch.argmax(next_token_logits_augmented, dim=-1).unsqueeze(0).detach()

            
#             # convert this logit to softmax probability, and return this highest probability (as confidence)
#             # confidence = F.softmax(next_token_logits_augmented, dim=-1).max().item().detach()
#             # print('\nnext token: ', tokenizer.decode(next_token_idx[0]), '  |  confidence: ', confidence, flush=True)
            
#             # update the input_ids
#             pre_input_ids_ori = copy.deepcopy(input_ids_ori)
#             input_ids_ori = torch.cat([input_ids_ori, next_token_idx], dim=-1).to(device)
#             input_ids_mask = torch.cat([input_ids_mask, next_token_idx], dim=-1).to(device)
            
#             input_ids_ori = input_ids_ori.detach().to(device)
#             input_ids_mask = input_ids_mask.detach().to(device)
            
#             # print the current prompt
#             cur_prompt = tokenizer.decode(input_ids_ori[0])

            
#             if next_token_idx[0].item() == tokenizer.eos_token_id: # break if <eos> token is generated
#                 break
            
#             # get new added tokens in the current prompt compared to the previous prompt
#             new_tok = cur_prompt[len(pre_prompt):]
            
#             # add special stop condition
#             if other_stop_condition(new_tok):
#                 break
            
            
#             # update generated_string
#             generated_string_pre = generated_string
#             generated_string += new_tok
            
#             # if language == 'python':
#             #     # if has_complete_python_function_generation(prompt_without_docstring, prompt_without_docstring + generated_string):
#             #     if has_complete_python_function_generation_deepseek(generated_string):
            
#             # determine if the code generation process should stop
#             if stop_condition(generated_string, ori_prompt, language, benchmark, checkpoint_name):
#                 input_ids_ori = pre_input_ids_ori
#                 # generated_string = generated_string_pre
#                 break
            
#             if print_code_to_console:
#                 print(new_tok, end="", flush=True)
            
#             # if log_path is not None, write (append to original content without new line) the new token to the log file
#             if log_path is not None:
#                 with open(log_path, 'a') as f:
#                     f.write(new_tok)

#             # update the previous prompt
#             pre_prompt = cur_prompt
            
#             # clear GPU memory
#             torch.cuda.empty_cache()
#             gc.collect()
    
#     # Decode the tokens to string
#     output_sequence = tokenizer.decode(input_ids_ori[0])

#     # get the code between ```python and ```
#     try:
#         entire_code = get_code_snippets_deepseek(generated_string)
#     except:
#         entire_code = generated_string
        
#     # add prompt for pure completion models (for humaneval)
#     if benchmark == "humaneval":
#         if 'codegen' in checkpoint_name.lower() or 'codellama' in checkpoint_name.lower():
#             entire_code = raw_prompt + entire_code

#     return output_sequence, (attention_to_ori_prompt_list_self, attention_to_ori_prompt_list_gradient), entire_code





# MAIN beam search
# keep the top k beams with the highest log probabilities
# create a funciton used to enhance the logits of the focused word during the decoding process (the generate function)
def augmented_generate_anchor_beam_search(raw_prompt, raw_masked_prompt, checkpoint_name, model, tokenizer, max_length, device, language, benchmark, weight, output_attentions=True, compute_self_attention=True, compute_gradient=True, adaptive_attention_weight=False, log_path=None, num_beams=10):
    
    final_candidates = []  # prepare the final beam candidates
    
    # preparation
    model.eval()
    ori_weight = weight  # store the original base weight
    
    if adaptive_attention_weight:
        output_attentions = True
    
    temp_input_ids_ori = prompt_to_inputs(raw_prompt, tokenizer, checkpoint_name, benchmark, language, device).to(device)
    temp_input_ids_mask = prompt_to_inputs(raw_masked_prompt, tokenizer, checkpoint_name, benchmark, language, device).to(device)
    
    # output the intial prompt
    print(tokenizer.decode(temp_input_ids_ori[0], skip_special_tokens=False), flush=True)
    
    # the beam element as a dictionary
    intial_beam_element = {
        "input_ids_ori": temp_input_ids_ori,
        "input_ids_mask": temp_input_ids_mask,
        "beam_prob": 1.0,
        "mask_len": len(temp_input_ids_ori[0]) - len(temp_input_ids_mask[0]) + 1,
        "init_input_ids_len": len(temp_input_ids_ori[0]),
        "ori_prompt": tokenizer.decode(temp_input_ids_ori[0], skip_special_tokens=False),
        "cur_prompt": tokenizer.decode(temp_input_ids_ori[0], skip_special_tokens=False),
        "pre_prompt": tokenizer.decode(temp_input_ids_ori[0], skip_special_tokens=False),
        "generated_string": '',
        "new_tok": '',
        "attention_to_ori_prompt_list_self": [],
        "attention_to_ori_prompt_list_gradient": [],
        "generation_time": 0,
        "stop_flag": False,
        "token_prob_list": [],  # used to store the confidence of the generated tokens
    }
    
    # construct beams as a list
    beams = [intial_beam_element]  # start with 1 initial beam
    candidates = []  # to buffer k*k candidates
    
    with conditional_no_grad(output_attentions):
        
        # if the final beam candidates is full, stop the generation process
        while len(final_candidates) < num_beams:
            
            # for each current beam, generate the next token corresponding to multiple candidates
            for beam_idx in range(len(beams)):
                # get states in the current beam element
                input_ids_ori = beams[beam_idx]["input_ids_ori"].to(device)
                input_ids_mask = beams[beam_idx]["input_ids_mask"].to(device)
                mask_len = beams[beam_idx]["mask_len"]
                init_input_ids_len = beams[beam_idx]["init_input_ids_len"]
                ori_prompt = beams[beam_idx]["ori_prompt"]
                cur_prompt = beams[beam_idx]["cur_prompt"]
                pre_prompt = beams[beam_idx]["pre_prompt"]
                generated_string = beams[beam_idx]["generated_string"]
                attention_to_ori_prompt_list_self = beams[beam_idx]["attention_to_ori_prompt_list_self"]
                attention_to_ori_prompt_list_gradient = beams[beam_idx]["attention_to_ori_prompt_list_gradient"]
                stop_flag = beams[beam_idx]["stop_flag"]
                generation_time = beams[beam_idx]["generation_time"]
                token_prob_list = beams[beam_idx]["token_prob_list"]
                beam_prob = beams[beam_idx]["beam_prob"]


                if stop_flag:
                    final_candidates.append(beams[beam_idx])
                    continue
                
                ## Start generating the next token ##
                outputs_ori = model(input_ids_ori.to(device), output_attentions=output_attentions)  # inference
                next_token_logits_ori = outputs_ori.logits[:, -1, :]  # get the logits of original and masked prompt
                
                # record attentions of models
                if output_attentions:
                    if compute_self_attention:
                        attention_ori_self = get_self_attention(input_ids_ori, outputs_ori)
                        attention_sum_self = sum(attention_ori_self[:init_input_ids_len])  # all the original prompt attentions
                        if attention_sum_self > 1:
                            attention_sum_self = 1
                        attention_to_ori_prompt_list_self.append(attention_sum_self)
                    
                    if compute_gradient:
                        attention_ori_gradient = get_gradient_attention(input_ids_ori, model, topk=100)
                        attention_sum_gradient = sum(attention_ori_gradient[:init_input_ids_len])  # all the original prompt attentions
                        if attention_sum_gradient > 1:
                            attention_sum_gradient = 1
                        attention_to_ori_prompt_list_gradient.append(attention_sum_gradient)
    
                
                if ori_weight != 0:
                    outputs_mask = model(input_ids_mask.to(device), output_attentions=output_attentions)  # inference
                    next_token_logits_mask = outputs_mask.logits[:, -1, :]  # get the logits of original and masked prompt
                            
                    logit_deviation = (next_token_logits_ori - next_token_logits_mask).detach()
                    
                    # Weight is decided based on the attention (if it is adaptive) and given weight
                    if adaptive_attention_weight:
                        cur_input_ids_len = len(input_ids_ori[0])  # calculate the current input_ids length for adjusting the weight
                        weight = (cur_input_ids_len - init_input_ids_len) / mask_len * ori_weight
                        
                    next_token_logits_augmented = (next_token_logits_ori + logit_deviation * weight).detach()
                    # Use Z-score to normalize the logits
                    next_token_logits_augmented = z_score(next_token_logits_augmented).detach()
                
                else:
                    next_token_logits_augmented = next_token_logits_ori.detach()
                
                

                
                # get the next TOP-k token (beam search)
                next_token_probs = F.softmax(next_token_logits_augmented, dim=-1)
                # get the top-k indice
                top_k_probs, top_k_indices = torch.topk(next_token_probs, k=num_beams, dim=-1)
                # print('\ntop_k_probs: ', top_k_probs, flush=True)
                
                
                # each current beam derivates num_beams candidate beams in the next step
                for candidate_idx in range(num_beams):
                    # get the next token
                    next_token_idx = top_k_indices[0][candidate_idx].unsqueeze(0)

                    

                    # Ensure input_ids_ori is 2D if it isn't already
                    if input_ids_ori.dim() == 1:
                        input_ids_ori = input_ids_ori.unsqueeze(0)

                    # Ensure next_token_idx is 2D and has the same batch dimension as input_ids_ori
                    if next_token_idx.dim() == 1:

                        next_token_idx = next_token_idx.unsqueeze(0)

                    elif next_token_idx.size(0) != input_ids_ori.size(0):
                        next_token_idx = next_token_idx.unsqueeze(0).expand(input_ids_ori.size(0), -1)

                    
                    


                    # update the input_ids
                    candidate_input_ids_ori = torch.cat([input_ids_ori, next_token_idx], dim=-1).detach().to(device)
                    candidate_input_ids_mask = torch.cat([input_ids_mask, torch.ones_like(next_token_idx)], dim=-1).detach().to(device)
                    

                    
                    # print the current prompt
                    candidate_cur_prompt = tokenizer.decode(candidate_input_ids_ori[0])
                    
                    pre_prompt = tokenizer.decode(input_ids_ori[0], skip_special_tokens=False)
                    candidate_cur_prompt = tokenizer.decode(candidate_input_ids_ori[0])
                    

                    
                    candiate_new_tok = candidate_cur_prompt[len(pre_prompt):]
                    
                    

                    candidate_generated_string = generated_string + candiate_new_tok

                    new_token_prob = top_k_probs[0][candidate_idx].item()
                    candidate_token_prob_list = token_prob_list + [new_token_prob]
                    
                    # Calculate the log probabilities instead of probabilities
                    log_prob_list = [math.log(prob) for prob in candidate_token_prob_list]
                    # Sum the log probabilities and convert back to normal scale by exponentiating
                    # Normalize by the generation time using exponent of the average log probability
                    # normalized_beam_prob = math.exp(sum(log_prob_list) / (len(candidate_token_prob_list) + 1))
                    # calculate the probability by multiplying all token probabilities in candidate_token_prob_list

                    # normalized beam prob is the accumulated multiplication of all token probabilities, root by the generation time
                    normalized_beam_prob = math.prod(candidate_token_prob_list) ** (1 / (generation_time + 1))
                    
                    # add candidate element to candidates
                    candidate_element = {
                        "input_ids_ori": candidate_input_ids_ori,
                        "input_ids_mask": candidate_input_ids_mask,
                        "beam_prob": normalized_beam_prob,   # this is accumulated multiplication of all token probabilities, but normalizied (increase) to prevent to overflow (equals to 0 after multiplying too many times)
                        "mask_len": mask_len,
                        "init_input_ids_len": init_input_ids_len,
                        "ori_prompt": ori_prompt,
                        "cur_prompt": candidate_cur_prompt,
                        "pre_prompt": cur_prompt,
                        "generated_string": candidate_generated_string,
                        "new_tok": candiate_new_tok,
                        "attention_to_ori_prompt_list_self": attention_to_ori_prompt_list_self.copy(),
                        "attention_to_ori_prompt_list_gradient": attention_to_ori_prompt_list_gradient.copy(),
                        "stop_flag": next_token_idx[0].item() == tokenizer.eos_token_id or stop_condition(candidate_generated_string, ori_prompt, language, benchmark, checkpoint_name) or other_stop_condition(candiate_new_tok) or max_length <= len(candidate_input_ids_ori[0]),
                        "generation_time": generation_time + 1,
                        "token_prob_list": candidate_token_prob_list,
                    }

                    candidates.append(candidate_element)

            
            # # normalize the beam_prob to prevent overflow
            # beam_prob_sum = sum([candidate["beam_prob"] for candidate in candidates])
            # for candidate in candidates:
            #     candidate["beam_prob"] = candidate["beam_prob"] / beam_prob_sum   
            
            # After processing all candidates, select the top beams
            # out of k*k candidates, select the top k candidates based on the beam_prob
            candidates.sort(key=lambda x: x["beam_prob"], reverse=True)
            beams = copy.deepcopy(candidates[:num_beams])
            
            # print the code in each beam
            # print beam information like candidate
            for i, beam in enumerate(beams):
                print(f'\nBeam {i}:', flush=True)
                print('-' * 45, flush=True)
                # print based on different models
                if 'codellama' in checkpoint_name.lower() or 'codegen' in checkpoint_name.lower():
                    print(raw_prompt + beam["generated_string"], flush=True)
                else:
                    print(beam["generated_string"], flush=True)
                
            
            print("\n\n-----==> number in final candidates: ", len(final_candidates), flush=True)

            print('\n')
            
            candidates = []
            
            # clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

    
    
    # get entire codes for each beam
    output_sequences = []
    entire_codes = []
    for final_beam in final_candidates:
        output_sequence = tokenizer.decode(final_beam["input_ids_ori"][0])
        output_sequences.append(output_sequence)
        try:
            entire_code = get_code_snippets_deepseek(final_beam["generated_string"])
        except:
            entire_code = final_beam["generated_string"]
            
            
        # add prompt for pure completion models (for humaneval)
        if benchmark == "humaneval":
            if 'codegen' in checkpoint_name.lower() or 'codellama' in checkpoint_name.lower():
                entire_code = raw_prompt + entire_code    
        
        entire_codes.append(entire_code)
    
    return output_sequences, entire_codes




# create a funciton used to enhance the logits of the focused word during the decoding process (the generate function)
def augmented_generate_deepseek_instruct_ratio(raw_prompt, raw_masked_prompt, model, tokenizer, max_length, device, task, benchmark, weight, log_path=None):
    
    
    ori_messages=[
        { 'role': 'user', 'content': build_deepseekcoder_instruction_python(raw_prompt, benchmark)}
    ]
    
    input_ids_ori = tokenizer.apply_chat_template(ori_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    masked_messages=[
        { 'role': 'user', 'content': build_deepseekcoder_instruction_python(raw_masked_prompt, benchmark)}
    ]
    input_ids_mask = tokenizer.apply_chat_template(masked_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    ori_prompt = tokenizer.decode(input_ids_ori[0], skip_special_tokens=False)
    
    cur_prompt = ori_prompt
    pre_prompt = ori_prompt
    
    # print(ori_prompt, end="", flush=True)
    if log_path is not None:
        # if the file does not exist, create it
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(ori_prompt)
        else:
            with open(log_path, 'a') as f:
                f.write(ori_prompt)
    
    # only the generated part
    generated_string = ''
    
    with torch.no_grad():
        for _ in range(max_length - len(input_ids_ori[0])):
            outputs_ori = model(input_ids_ori)
            outputs_mask = model(input_ids_mask)

            # get the logits of original and masked prompt
            next_token_logits_ori = outputs_ori.logits[:, -1, :]
            next_token_logits_mask = outputs_mask.logits[:, -1, :]
            
            # softmax the logits
            next_token_probs_ori = F.softmax(next_token_logits_ori, dim=-1)
            next_token_probs_mask = F.softmax(next_token_logits_mask, dim=-1)
            
            # print('\nnext_token_probs_ori: ', next_token_probs_ori, flush=True)
            # print('\nnext_token_probs_mask: ', next_token_probs_mask, flush=True)
            
            # calculate the ratio of tensors between original and masked logits (original / masked)
            logit_ratio = next_token_probs_ori / next_token_probs_mask
            
            # adjust the ratio by the rooting the weight
            logit_ratio = logit_ratio ** (weight)
            
            # print('\nlogit_ratio: ', logit_ratio, flush=True)
            
            # next_token_probs_augmented = next_token_probs_ori * logit_ratio * weight + next_token_probs_ori * (1 - weight)
            
            next_token_probs_augmented = next_token_probs_ori * logit_ratio 
            
            
            # softmax the augmented probs
            # next_token_probs_augmented = F.softmax(next_token_probs_augmented, dim=-1)
            
            # # next_token_logits_augmented = next_token_logits_augmented / (weight + 1)
            # # Use Z-score to normalize the logits
            # next_token_logits_augmented = z_score(next_token_logits_augmented)
            
            # get the next token
            next_token_idx = torch.argmax(next_token_probs_augmented, dim=-1).unsqueeze(0)
            
            # convert this logit to softmax probability, and return this highest probability (as confidence)
            # confidence = F.softmax(next_token_probs_augmented, dim=-1).max().item()
            # print('\nnext token: ', tokenizer.decode(next_token_idx[0]), '  |  confidence: ', confidence, flush=True)
            
            # update the input_ids
            pre_input_ids_ori = copy.deepcopy(input_ids_ori)
            input_ids_ori = torch.cat([input_ids_ori, next_token_idx], dim=-1).to(device)
            input_ids_mask = torch.cat([input_ids_mask, next_token_idx], dim=-1).to(device)
            
            
            # print the current prompt
            cur_prompt = tokenizer.decode(input_ids_ori[0])

            
            if next_token_idx[0].item() == tokenizer.eos_token_id: # break if <eos> token is generated
                break
            
            # get new added tokens in the current prompt compared to the previous prompt
            new_tok = cur_prompt[len(pre_prompt):]
            
            
            # update generated_string
            generated_string_pre = generated_string
            generated_string += new_tok
            
            
            if task == 'python':
                # if has_complete_python_function_generation(prompt_without_docstring, prompt_without_docstring + generated_string):
                if has_complete_python_function_generation_deepseek(generated_string):
                    input_ids_ori = pre_input_ids_ori
                    generated_string = generated_string_pre
                    break
            
            print(new_tok, end="", flush=True)
            
            # if log_path is not None, write (append to original content without new line) the new token to the log file
            if log_path is not None:
                with open(log_path, 'a') as f:
                    f.write(new_tok)

            # update the previous prompt
            pre_prompt = cur_prompt

    # Decode the tokens to string
    output_sequence = tokenizer.decode(input_ids_ori[0])



    # get the code between ```python and ```
    try:
        entire_code = get_code_snippets_deepseek(generated_string)
    except:
        entire_code = generated_string
    
    return output_sequence, '', entire_code


# create a funciton used to enhance the logits of the focused word during the decoding process (the generate function)
def augmented_generate_deepseek_instruct_old(raw_prompt, raw_masked_prompt, model, tokenizer, max_length, device, task, weight=0.5, log_path=None):
    
    # instruction = "Complete the following python code:\n"
    instruction = ""
    
    ori_messages=[
        { 'role': 'user', 'content': instruction + raw_prompt}
    ]
    temp_input_ids_ori = tokenizer.apply_chat_template(ori_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    masked_messages=[
        { 'role': 'user', 'content': instruction + raw_masked_prompt}
    ]
    temp_input_ids_mask = tokenizer.apply_chat_template(masked_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    
    prompt_without_docstring = remove_humanEval_docstring(raw_prompt)
    
    generate_back_prompt = tokenizer.decode(temp_input_ids_ori[0], skip_special_tokens=False)
    generate_back_prompt_mask = tokenizer.decode(temp_input_ids_mask[0], skip_special_tokens=False)
    
    # remove <｜begin▁of▁sentence｜> at the beginning
    generate_back_prompt = generate_back_prompt.replace('<｜begin▁of▁sentence｜>', '')
    generate_back_prompt_mask = generate_back_prompt_mask.replace('<｜begin▁of▁sentence｜>', '')
    
    # force the generation start with the original prompt
    generate_back_prompt += prompt_without_docstring
    generate_back_prompt_mask += prompt_without_docstring
    
    # generate_back_prompt += raw_prompt
    # generate_back_prompt_mask += raw_masked_prompt
    
    # generate back to ids
    input_ids_ori = tokenizer(generate_back_prompt, return_tensors="pt")['input_ids'].to(model.device)
    input_ids_mask = tokenizer(generate_back_prompt_mask, return_tensors="pt")['input_ids'].to(model.device)
    
    ori_prompt = tokenizer.decode(input_ids_ori[0], skip_special_tokens=False)
    
    cur_prompt = ori_prompt
    pre_prompt = ori_prompt
    
    print(ori_prompt, end="", flush=True)
    if log_path is not None:
        # if the file does not exist, create it
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(ori_prompt)
        else:
            with open(log_path, 'a') as f:
                f.write(ori_prompt)
    
    # only the generated part
    generated_string = ''
    
    with torch.no_grad():
        for _ in range(max_length - len(input_ids_ori[0])):
            outputs_ori = model(input_ids_ori)
            outputs_mask = model(input_ids_mask)

            # get the logits of original and masked prompt
            next_token_logits_ori = outputs_ori.logits[:, -1, :]
            next_token_logits_mask = outputs_mask.logits[:, -1, :]
            
            logit_deviation = next_token_logits_ori - next_token_logits_mask
            next_token_logits_augmented = next_token_logits_ori + logit_deviation * weight
            # next_token_logits_augmented = next_token_logits_augmented / (weight + 1)
            # Use Z-score to normalize the logits
            next_token_logits_augmented = z_score(next_token_logits_augmented)
            
            # get the next token
            next_token_idx = torch.argmax(next_token_logits_augmented, dim=-1).unsqueeze(0)
            
            # convert this logit to softmax probability, and return this highest probability (as confidence)
            confidence = F.softmax(next_token_logits_augmented, dim=-1).max().item()
            # print('\nnext token: ', tokenizer.decode(next_token_idx[0]), '  |  confidence: ', confidence, flush=True)
            
            # update the input_ids
            pre_input_ids_ori = copy.deepcopy(input_ids_ori)
            input_ids_ori = torch.cat([input_ids_ori, next_token_idx], dim=-1).to(device)
            input_ids_mask = torch.cat([input_ids_mask, next_token_idx], dim=-1).to(device)
            
            
            # print the current prompt
            cur_prompt = tokenizer.decode(input_ids_ori[0])

            
            if next_token_idx[0].item() == tokenizer.eos_token_id: # break if <eos> token is generated
                break
            
            # get new added tokens in the current prompt compared to the previous prompt
            new_tok = cur_prompt[len(pre_prompt):]
            
            
            # update generated_string
            generated_string_pre = generated_string
            generated_string += new_tok
            
            
            if task == 'python':
                if has_complete_python_function_generation(prompt_without_docstring, prompt_without_docstring + generated_string):
                    input_ids_ori = pre_input_ids_ori
                    generated_string = generated_string_pre
                    break
            
            print(new_tok, end="", flush=True)
            
            # if log_path is not None, write (append to original content without new line) the new token to the log file
            if log_path is not None:
                with open(log_path, 'a') as f:
                    f.write(new_tok)

            # update the previous prompt
            pre_prompt = cur_prompt
    
    # Decode the tokens to string
    output_sequence = tokenizer.decode(input_ids_ori[0])
    
    # entire_code = get_code_snippets_deepseek(generated_string)
    
    entire_code = raw_prompt + generated_string

    return output_sequence, generated_string, entire_code




# create a funciton used to enhance the logits of the focused word during the decoding process (the generate function)
def augmented_generate(ori_prompt, masked_prompt, model, tokenizer, max_length, device, task, weight=0.5, log_path=None):
    input_ids_ori = tokenizer.encode(ori_prompt, return_tensors="pt").to(device)
    input_ids_mask = tokenizer.encode(masked_prompt, return_tensors="pt").to(device)
    
    cur_prompt = ori_prompt
    pre_prompt = ori_prompt
    
    print(ori_prompt, end="", flush=True)
    if log_path is not None:
        # if the file does not exist, create it
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(ori_prompt)
        else:
            with open(log_path, 'a') as f:
                f.write(ori_prompt)
    
    # only the generated part
    generated_string = ''
    
    with torch.no_grad():
        for _ in range(max_length - len(input_ids_ori[0])):
            outputs_ori = model(input_ids_ori)
            outputs_mask = model(input_ids_mask)

            # get the logits of original and masked prompt
            next_token_logits_ori = outputs_ori.logits[:, -1, :]
            next_token_logits_mask = outputs_mask.logits[:, -1, :]
            
            logit_deviation = next_token_logits_ori - next_token_logits_mask
            next_token_logits_augmented = next_token_logits_ori + logit_deviation * weight
            # next_token_logits_augmented = next_token_logits_augmented / (weight + 1)
            # Use Z-score to normalize the logits
            next_token_logits_augmented = z_score(next_token_logits_augmented)
            
            # get the next token
            next_token_idx = torch.argmax(next_token_logits_augmented, dim=-1).unsqueeze(0)
            
            # convert this logit to softmax probability, and return this highest probability (as confidence)
            confidence = F.softmax(next_token_logits_augmented, dim=-1).max().item()
            # print('\nnext token: ', tokenizer.decode(next_token_idx[0]), '  |  confidence: ', confidence, flush=True)
            
            # update the input_ids
            pre_input_ids_ori = copy.deepcopy(input_ids_ori)
            input_ids_ori = torch.cat([input_ids_ori, next_token_idx], dim=-1).to(device)
            input_ids_mask = torch.cat([input_ids_mask, next_token_idx], dim=-1).to(device)
            
            
            # print the current prompt
            cur_prompt = tokenizer.decode(input_ids_ori[0])
            # remove "<s> " at the beginning
            while cur_prompt.startswith('<s> '):
                cur_prompt = cur_prompt[4:]
            
            
            if next_token_idx[0].item() == tokenizer.eos_token_id: # break if <eos> token is generated
                break
            
            if task == 'python':
                if has_complete_python_function_generation(ori_prompt, cur_prompt):
                    input_ids_ori = pre_input_ids_ori
                    break

            
            # get new added tokens in the current prompt compared to the previous prompt
            new_tok = cur_prompt[len(pre_prompt):]
            print(new_tok, end="", flush=True)
            # update generated_string
            generated_string += new_tok
            
            # if log_path is not None, write (append to original content without new line) the new token to the log file
            if log_path is not None:
                with open(log_path, 'a') as f:
                    f.write(new_tok)

            # update the previous prompt
            pre_prompt = cur_prompt
    
    # Decode the tokens to string
    output_sequence = tokenizer.decode(input_ids_ori[0])

    return output_sequence, generated_string




# augmented generated parallelly after the prompt is processed
def augmented_generate_parallelly(weighted_prompt, model, tokenizer, max_length, device):
    # get the original prompt from the weighted/formatted prompt
    ori_prompt = get_original_prompt(weighted_prompt)
    
    # Process the weighted prompt to a list of separately toekn-masked prompts (with <unk>), associated with weights
    weighted_masked_prompt_list = get_weighted_masked_prompt_list(weighted_prompt)
    
    # get the input ids of the original prompt
    input_ids_ori = tokenizer.encode(ori_prompt, return_tensors="pt").to(device)
    
    # get the input ids of the masked prompts for each element in the weighted_masked_prompt_list
    input_ids_mask_list = []
    for masked_prompt, weight in weighted_masked_prompt_list:
        input_ids_mask = tokenizer.encode(masked_prompt, return_tensors="pt").to(device)
        input_ids_mask_list.append((input_ids_mask, weight))
    
    
    # start decoding
    with torch.no_grad():
        for _ in range(max_length - len(input_ids_ori[0])):
            # get the logits of original and masked prompt
            outputs_ori = model(input_ids_ori)
            next_token_logits_ori = outputs_ori.logits[:, -1, :]
            
            
            logit_deviation_list = []
            for (input_ids_mask, weight) in input_ids_mask_list:
                outputs_mask = model(input_ids_mask)
                next_token_logits_mask = outputs_mask.logits[:, -1, :]
                logit_deviation = (next_token_logits_ori - next_token_logits_mask) * weight
                logit_deviation_list.append(logit_deviation)
                
            # add the logits of masked prompts to the original logits, get the accumulated logits
            
            accumulated_logits = torch.zeros_like(next_token_logits_ori)
            for next_token_logits_deviation in logit_deviation_list:
                accumulated_logits += next_token_logits_deviation
            next_token_logits_augmented = next_token_logits_ori + accumulated_logits
                
            
            # after calculating the accumulated logits, get the next token
            next_token_idx = torch.argmax(next_token_logits_augmented, dim=-1).unsqueeze(0)
            
            # convert this logit to softmax probability, and return this highest probability (as confidence)
            confidence = F.softmax(next_token_logits_augmented, dim=-1).max().item()
            print('\nnext token: ', tokenizer.decode(next_token_idx[0]), 'confidence: ', confidence, flush=True)
            
            # update the input_ids for original prompt
            input_ids_ori = torch.cat([input_ids_ori, next_token_idx], dim=-1)
            # update the input_ids of masked prompts
            for i, (input_ids_mask, weight) in enumerate(input_ids_mask_list):
                input_ids_mask_list[i] = (torch.cat([input_ids_mask, next_token_idx], dim=-1), weight)
            
            if next_token_idx[0].item() == tokenizer.eos_token_id: # break if <eos> token is generated
                break
    
    
    # Decode the tokens to string
    output_sequence = tokenizer.decode(input_ids_ori[0])
    
    return output_sequence


# given the prompt, mask_tok, and the benchmark, return the masked prompt
def get_masked_prompt(prompt, mask_tok, benchmark, test_list=''):
    if benchmark == "humaneval":
        return mask_humanEval_instruction(prompt, mask_tok=mask_tok)
    elif benchmark == "mbpp":
        return mask_tok + '\n'.join(test_list)
    else:
        raise ValueError("Undefined benchmark")


# #


def clean_token(token):
    """Clean special tokens and characters."""
    # Skip special model tokens
    if token.startswith('<|') and token.endswith('|>'):
        return None
        
    # Handle special characters
    token = (token.replace('Ġ', ' ')  # Replace Ġ with space
                 .replace('ĉ', '\n')   # Replace ĉ with newline
                 .replace('Ċ', '\n')   # Replace Ċ with newline
                 .strip())             # Remove leading/trailing whitespace
    
    return token if token else None

def process_attention_dist(attention_dist, text, current_token_idx=None):
    """
    Process and clean attention distribution with concrete token indices.
    current_token_idx: tracks position of current token in the sequence (None for first token)
    """
    cleaned_dist = []
    tokens_with_positions = []
    current_position = 0
    token_idx = 0
    
    # First pass: collect all valid tokens and their positions
    for item in attention_dist:
        cleaned_token = clean_token(item['token'])
        if cleaned_token:
            token_pos = text.find(cleaned_token, current_position)
            if token_pos != -1:
                tokens_with_positions.append({
                    'token': cleaned_token,
                    'position': token_pos,
                    'attention': float(item['attention']),
                    'idx': token_idx
                })
                current_position = token_pos + len(cleaned_token)
                token_idx += 1
    
    # For generation steps, only include tokens up to the current position
    if current_token_idx is not None:
        tokens_with_positions = tokens_with_positions[:current_token_idx+1]
    
    # Normalize attention values
    if tokens_with_positions:
        attention_values = [item['attention'] for item in tokens_with_positions]
        max_attention = max(attention_values)
        min_attention = min(attention_values)
        attention_range = max_attention - min_attention
        
        # Normalize and create final distribution
        for item in tokens_with_positions:
            norm_attention = (item['attention'] - min_attention) / attention_range if attention_range > 0 else 1.0
            cleaned_dist.append({
                'token': item['token'],
                'attention': float(norm_attention),
                'position': item['position'],
                'idx': item['idx']
            })
    
    return cleaned_dist


def map_token_to_word_attention(tokens_with_attention, text):
    """
    Convert token-level attention to word-level attention.
    Returns word-level attention with exact positions and token index ranges.
    """
    words = text.split()
    word_positions = []
    current_pos = 0
    
    # Find the position of each word in the original text
    for word in words:
        while current_pos < len(text) and text[current_pos].isspace():
            current_pos += 1
        if current_pos < len(text):
            word_positions.append({
                'word': word,
                'start': current_pos,
                'end': current_pos + len(word),
                'attention': 0.0,
                'token_indices': set()  # Store all token indices that overlap with this word
            })
            current_pos += len(word)

    # Map token attention to words and track token indices
    for token_info in tokens_with_attention:
        token = token_info['token']
        token_pos = token_info['position']
        token_attention = token_info['attention']
        token_idx = token_info['idx']
        
        # Find which words this token overlaps with
        token_end = token_pos + len(token)
        for word_info in word_positions:
            # Check if token overlaps with this word
            if (token_pos < word_info['end'] and token_end > word_info['start']):
                # Add token index to the word's set of indices
                word_info['token_indices'].add(token_idx)
                # Update attention value (take maximum of overlapping tokens)
                word_info['attention'] = max(word_info['attention'], token_attention)

    # Create final word-level attention distribution
    word_attention_dist = []
    for i, word_info in enumerate(word_positions):
        if word_info['attention'] > 0:  # Only include words with attention
            token_indices = sorted(list(word_info['token_indices']))
            word_attention_dist.append({
                'token': word_info['word'],
                'attention': float(word_info['attention']),
                'position': word_info['start'],
                'idx_range': {
                    'start': min(token_indices) if token_indices else i,
                    'end': max(token_indices) if token_indices else i
                }
            })
    
    return word_attention_dist

def SPA_generate(data, model, tokenizer, special_token, max_new_tokens=1000, use_kv_cache=True, output_attentions=False):
    
    TOP_K_CANDIDATES = 5
    MAX_INPUT_LENGTH = 2048  # Add max input length to prevent OOM
    
    submitted_text = data.get("text", "")
    text_values = data.get("anchors", [])

    messages = [
        {"role": "system", "content": "You are a helpful assistant!"},
        {"role": "user", "content": submitted_text},
    ]

    model.config.use_flash_attention = True
    
    # Truncate input if too long
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    if inputs.shape[1] > MAX_INPUT_LENGTH:
        inputs = inputs[:, -MAX_INPUT_LENGTH:]
    
    weighted_text_augments = []
    current_sequence = submitted_text  # Keep track of the full sequence for token positions
    token_count = 0                   # Keep track of number of tokens processed

    # Initialize KV cache and get attention
    with torch.no_grad():
        outputs = model(inputs, use_cache=use_kv_cache, output_attentions=output_attentions)
        past_key_values = outputs.past_key_values if use_kv_cache else None
        
        attention_dist = []
        if output_attentions:
            attention_weights = outputs.attentions
            avg_attention = torch.mean(torch.stack(attention_weights), dim=[0,2])
            last_token_attention = avg_attention[0, -1, :]
            input_tokens = tokenizer.convert_ids_to_tokens(inputs[0])
            
            # Process initial attention distribution
            attention_dist = process_attention_dist(
                [{"token": token, "attention": float(attn)} 
                 for token, attn in zip(input_tokens, last_token_attention)],
                current_sequence,
                token_count
            )
    
    # Initialize KV caches for masked inputs
    for item in text_values:
        temp_element = {}
        temp_message = [
            {"role": "system", "content": "You are a helpful assistant who should provide concise answers."},
            {"role": "user", "content": item['masked_text']},
        ]
        temp_inputs = tokenizer.apply_chat_template(
            temp_message, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        
        # Truncate masked inputs if too long
        if temp_inputs.shape[1] > MAX_INPUT_LENGTH:
            temp_inputs = temp_inputs[:, -MAX_INPUT_LENGTH:]
            
        temp_element['inputs'] = temp_inputs
        temp_element['weight'] = item['value']
        
        with torch.no_grad():
            masked_outputs = model(temp_element['inputs'], use_cache=use_kv_cache)
            temp_element['past_key_values'] = masked_outputs.past_key_values if use_kv_cache else None
            
        weighted_text_augments.append(temp_element)

    for i in range(max_new_tokens):

        with torch.no_grad():
            model_inputs = {'input_ids': inputs[:, -1:]}
            if use_kv_cache:
                model_inputs['past_key_values'] = past_key_values
                model_inputs['use_cache'] = True
            
            outputs = model(**model_inputs, output_attentions=output_attentions)
            past_key_values = outputs.past_key_values if use_kv_cache else None
            ori_logits = outputs.logits[:, -1, :]
            
            if output_attentions:
                # Get all input tokens up to current position
                attention_weights = outputs.attentions
                avg_attention = torch.mean(torch.stack(attention_weights), dim=[0,2])
                last_token_attention = avg_attention[0, -1, :]
                all_tokens = tokenizer.convert_ids_to_tokens(inputs[0])
                
                # Update attention distribution with all previous tokens
                attention_dist = process_attention_dist(
                    [{"token": token, "attention": float(attn)} 
                     for token, attn in zip(all_tokens, last_token_attention)],
                    current_sequence,
                    token_count
                )
            
            for augment_ele in weighted_text_augments:
                model_inputs = {'input_ids': augment_ele['inputs'][:, -1:]}
                if use_kv_cache:
                    model_inputs['past_key_values'] = augment_ele['past_key_values']
                    model_inputs['use_cache'] = True
                
                masked_outputs = model(**model_inputs)
                augment_ele['past_key_values'] = masked_outputs.past_key_values if use_kv_cache else None
                augment_ele['logits'] = masked_outputs.logits[:, -1, :]
                augment_ele['probs'] = torch.nn.functional.softmax(masked_outputs.logits[:, -1, :], dim=-1)
                augment_ele['logits_diff'] = (ori_logits - augment_ele['logits']) * augment_ele['weight']

        augmented_logits = ori_logits + sum([augment_ele['logits_diff'] for augment_ele in weighted_text_augments])
        next_token_probs = torch.nn.functional.softmax(augmented_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(next_token_probs, TOP_K_CANDIDATES)
        
        next_token_id = top_k_indices[0, 0].item()
        next_token = tokenizer.decode(next_token_id)

        confidence_score = next_token_probs[0, next_token_id].item()

        candidates = []
        for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
            if prob.item() > 0.01:
                candidates.append({
                    'token': tokenizer.decode(idx.item()),
                    'probability': prob.item()
                })

        if next_token_id == tokenizer.eos_token_id:
            break

        # Update the current sequence and token count
        current_sequence += next_token
        token_count += 1

        # Convert token-level attention to word-level attention if attention is enabled
        word_attention_dist = attention_dist if output_attentions else []
        
        response_data = {
            'token': next_token,
            'confidence': confidence_score,
            'candidates': candidates,
            'attention_dist': word_attention_dist  # Use word-level attention if enabled
        }
        yield response_data

        inputs = torch.cat([inputs, torch.tensor([[next_token_id]]).to(model.device)], dim=-1)
        for augment_ele in weighted_text_augments:
            augment_ele['inputs'] = torch.cat([augment_ele['inputs'], torch.tensor([[next_token_id]]).to(model.device)], dim=-1)
            augment_ele['logits'] = None
            augment_ele['probs'] = None
            augment_ele['logits_diff'] = None

    model.config.use_flash_attention = False


# get the entire generation result
def get_SPA_generation(data, model, tokenizer, special_token="<|reserved_special_token_0|>", max_new_tokens=500):
    response = ""
    for chunk in SPA_generate(data, model, tokenizer, special_token, max_new_tokens):
        if 'token' in chunk.keys():
            # print(chunk)
            response += chunk['token']
    return response

# base SPA that anchors the entire initial prompt as anchors
# Given a prompt, return the formatted data for SPA generation
def get_base_spa_input(prompt, weight, special_token):
    return {
        'text': prompt,
        'anchors': [{'text': prompt, 'value': weight, 'masked_text': special_token}]
    }




# define the SPA generation pipeline for different benchmarks
def eval_spa_pipeline(benchmark, weight, model, tokenizer, special_token, prompt, **kwargs):
    
    if benchmark == "truthfulqa":
        
        if weight == 0:
            # get input data without anchors
            data = {
                'text': prompt,
                'anchors': []
            }


        else:
            # get input data with anchors
            data = get_base_spa_input(prompt, weight, special_token)
            

            
    elif benchmark == "boolq":
        if weight == 0:
            # get input data without anchors
            data = {
                'text': prompt,
                'anchors': []
            }


        else:

            # get the context from the additional arguments
            context = kwargs['context']
            question = kwargs['question']
            # get input data with anchors
            # replace the context with masked token
            masked_prompt = prompt.replace(context, special_token).replace(question, special_token)

            data = {
                'text': prompt,
                'anchors': [{'masked_text': masked_prompt, 'value': weight}]
            }



    elif benchmark == "mmlu":
        if weight == 0:
            # get input data without anchors
            data = {
                'text': prompt,
                'anchors': []
            }

        else:
            # get the context from the additional arguments
            question = kwargs['question']
            choices = kwargs['choices']

            masked_prompt = prompt.replace(question, special_token).replace(str(choices), special_token)

            data = {
                'text': prompt,
                'anchors': [{'masked_text': masked_prompt, 'value': weight}]
            }
    
    elif benchmark == "gsm8k":
        if weight == 0:
            # get input data without anchors
            data = {
                'text': prompt,
                'anchors': []
            }
        else:
            # get the context from the additional arguments
            question = kwargs['question']
            # get input data with anchors
            masked_prompt = prompt.replace(question, special_token)
            data = {
                'text': prompt,
                'anchors': [{'masked_text': masked_prompt, 'value': weight}]
            }
            
    elif benchmark == "bird":
        if weight == 0:
            # get input data without anchors
            data = {
                'text': prompt,
                'anchors': []
            }
        else:
            question = kwargs['question']
            # get input data with anchors
            masked_prompt = prompt.replace(question, special_token)
            data = {
                'text': prompt,
                'anchors': [{'masked_text': masked_prompt, 'value': weight}]
            }
    
    elif benchmark == "humaneval":
        if weight == 0:
            # get input data without anchors
            data = {
                'text': prompt,
                'anchors': []
            }
        else:
            code_to_complete = kwargs['code_to_complete']
            # get input data with anchors
            masked_prompt = prompt.replace(code_to_complete, special_token)
            data = {
                'text': prompt,
                'anchors': [{'masked_text': masked_prompt, 'value': weight}]
            }
    
    
    # print everything
    print('-'*100, flush=True)
    print(data, flush=True)
    print('-'*100, flush=True)
    
    response = get_SPA_generation(data, model, tokenizer, special_token=special_token, max_new_tokens=len(prompt)+500)
    
    return response