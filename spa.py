import time
import re
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    GenerationConfig,
    TextIteratorStreamer,
)

# --- Helper Function for Input Preparation ---

def create_masked_attention(input_ids, target_strings, tokenizer):
    """
    Creates an attention mask where tokens corresponding to any of the target strings have 0 attention.
    """
    # Ensure input_ids is 2D
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)
    
    # Create default attention mask (all 1s)
    attention_mask = torch.ones_like(input_ids)
    
    # Convert single string to list for uniform processing
    if isinstance(target_strings, str):
        target_strings = [target_strings]
    
    # Get the input IDs as a list
    input_ids_list = input_ids[0].tolist()
    
    # Decode each token individually for comparison
    token_texts = []
    for token_id in input_ids_list:
        token_texts.append(tokenizer.decode([token_id]))
    

    
    masked_indices = []
    
    # Try tokenizing each target string to find its exact token representation
    for target_string in target_strings:
        if not target_string:
            continue
            
        # Tokenize the target string to get its expected token IDs
        target_ids = tokenizer.encode(target_string, add_special_tokens=False)
        target_tokens = [tokenizer.decode([id]) for id in target_ids]

        
        # First approach: Direct token sequence matching
        # Look for the sequence of tokens in the input
        for i in range(len(token_texts) - len(target_tokens) + 1):
            # Check if this position starts a matching sequence
            all_match = True
            for j, target_token in enumerate(target_tokens):
                if i+j >= len(token_texts) or target_token != token_texts[i+j]:
                    all_match = False
                    break
            
            if all_match:
                for j in range(len(target_tokens)):
                    attention_mask[0, i+j] = 0
                    masked_indices.append(i+j)
                    
        # Second approach: Look for individual tokens that make up the target
        for i, token_text in enumerate(token_texts):
            if token_text.strip() in target_tokens:
                attention_mask[0, i] = 0
                masked_indices.append(i)
                
        # Third approach: If the target is split between tokens, try to detect it
        # For example 'MASKTOKEN' might be split as ' MASK' and 'TOKEN'
        if len(target_tokens) == 1 and len(target_tokens[0]) > 2:  # Only for substantial single tokens
            # Look for token pairs that might contain the target
            for i in range(len(token_texts) - 1):
                pair = token_texts[i].strip() + token_texts[i+1].strip()
                if target_string in pair:
                    attention_mask[0, i] = 0
                    attention_mask[0, i+1] = 0
                    masked_indices.extend([i, i+1])
                    
                # Check for triplet if possible
                if i < len(token_texts) - 2:
                    triplet = token_texts[i].strip() + token_texts[i+1].strip() + token_texts[i+2].strip()
                    if target_string in triplet:
                        attention_mask[0, i] = 0
                        attention_mask[0, i+1] = 0
                        attention_mask[0, i+2] = 0
                        masked_indices.extend([i, i+1, i+2])
        
    
    # Print the final mask
    mask_positions = list(set(masked_indices))  # Remove duplicates
    mask_positions.sort()
    
    if mask_positions:
        masked_text = [token_texts[idx] for idx in mask_positions]
    else:
        print("WARNING: No tokens were masked!")
        # Last resort - just mask any token containing part of the target
        for target_string in target_strings:
            for i, token_text in enumerate(token_texts):
                if (target_string in token_text) or (token_text.strip() in target_string and len(token_text.strip()) > 2):
                    attention_mask[0, i] = 0
                    masked_indices.append(i)
        
        # Check again
        mask_positions = list(set(masked_indices))
        mask_positions.sort()
        
    return attention_mask


def preprocess_anchors(anchors):
    # remove duplicates in anchors
    anchors = list(set(anchors))
    # remove ""， " " in anchors
    anchors = [anchor for anchor in anchors if anchor != "" and anchor != " "]
    # sort the anchors by length
    anchors = sorted(anchors, key=len, reverse=True)
    return anchors


# Define a wrapper function to handle different cases
# The provided anchors are viewed as global anchors
def format_spa_input(input, anchors, mask_token, whole_word_only=True):
    # check if the input is a string or a list of messages
    if isinstance(input, str):
        # 1. Collect all anchors
        current_anchors = list(anchors) # Start with global anchors
        tag_anchors = []
        if re.search(r"<anchor>", input):
            tag_anchors = re.findall(r"<anchor>(.*?)</anchor>", input, flags=re.DOTALL)
            current_anchors.extend(tag_anchors)
        
        # 2. Clean the input string (remove tags)
        cleaned_input = re.sub(r"<anchor>|</anchor>", "", input)

        # 3. Preprocess all collected anchors (unique, non-empty, sorted desc)
        final_anchors = preprocess_anchors(current_anchors)

        # 4. Escape anchors for regex and build pattern (longest first)
        masked_input = cleaned_input # Initialize with cleaned input
        if final_anchors:
            if whole_word_only:
                # Use lookarounds to assert boundaries without consuming them (Fix 1)
                escaped_anchors = [rf"(?<!\w){re.escape(a)}(?!\w)" for a in final_anchors]
            else:
                escaped_anchors = [re.escape(a) for a in final_anchors]
            
            pattern = "|".join(escaped_anchors)
            # 5. Perform anchor replacement in one pass
            masked_input = re.sub(pattern, mask_token, cleaned_input)
        
        # 6. Post-processing: Merge consecutive mask tokens (separated by space)
        if mask_token: # Avoid processing if mask_token is empty
            escaped_mask_token = re.escape(mask_token)
            # Improved merging logic (Fix 2)
            merge_pattern = f"{escaped_mask_token}\s+{escaped_mask_token}"
            while re.search(merge_pattern, masked_input):
                masked_input = re.sub(merge_pattern, mask_token, masked_input)
            # Optional: merge masks without space if needed, e.g., mask_token+mask_token -> mask_token
            # merge_pattern_no_space = f"{escaped_mask_token}{escaped_mask_token}"
            # while re.search(merge_pattern_no_space, masked_input):
            #     masked_input = re.sub(merge_pattern_no_space, mask_token, masked_input)

        return cleaned_input, masked_input

    elif isinstance(input, list):
        cleaned_input_list = []
        masked_input_list = []

        for msg in input:
            msg_copy = msg.copy() # Work on a copy
            content = msg_copy.get("content", "")

            # 1. Collect all anchors for this message
            current_anchors = list(anchors) # Start with global anchors
            if "anchors" in msg_copy:
                dict_anchors = msg_copy.get("anchors", [])
                if isinstance(dict_anchors, list):
                    current_anchors.extend(dict_anchors)
            tag_anchors = []
            if re.search(r"<anchor>", content):
                tag_anchors = re.findall(r"<anchor>(.*?)</anchor>", content, flags=re.DOTALL)
                current_anchors.extend(tag_anchors)

            # 2. Clean the message content (remove tags)
            cleaned_content = re.sub(r"<anchor>|</anchor>", "", content)

            # 3. Preprocess all collected anchors for this message
            final_anchors = preprocess_anchors(current_anchors)

            # 4. Escape anchors, build pattern, and replace in one pass
            masked_content = cleaned_content # Initialize
            if final_anchors:
                if whole_word_only:
                    # Use lookarounds to assert boundaries without consuming them (Fix 1)
                    escaped_anchors = [rf"(?<!\w){re.escape(a)}(?!\w)" for a in final_anchors]
                else:
                    escaped_anchors = [re.escape(a) for a in final_anchors]
                
                pattern = "|".join(escaped_anchors)
                masked_content = re.sub(pattern, mask_token, cleaned_content)
            
            # 5. Post-processing: Merge consecutive mask tokens (separated by space) for this message
            if mask_token:
                escaped_mask_token = re.escape(mask_token)
                # Improved merging logic (Fix 2)
                merge_pattern = f"{escaped_mask_token}\s+{escaped_mask_token}"
                while re.search(merge_pattern, masked_content):
                     masked_content = re.sub(merge_pattern, mask_token, masked_content)
                # Optional: merge masks without space if needed
                # merge_pattern_no_space = f"{escaped_mask_token}{escaped_mask_token}"
                # while re.search(merge_pattern_no_space, masked_content):
                #     masked_content = re.sub(merge_pattern_no_space, mask_token, masked_content)

            # 6. Prepare output dictionaries
            final_cleaned_msg = msg_copy.copy()
            final_cleaned_msg["content"] = cleaned_content
            if "anchors" in final_cleaned_msg:
                del final_cleaned_msg["anchors"]

            final_masked_msg = msg_copy.copy()
            final_masked_msg["content"] = masked_content
            if "anchors" in final_masked_msg:
                del final_masked_msg["anchors"]

            cleaned_input_list.append(final_cleaned_msg)
            masked_input_list.append(final_masked_msg)

        return cleaned_input_list, masked_input_list
    else:
        raise ValueError("Invalid input type. Must be string or list of dictionaries.")


def get_mask_messages(messages, mask_token):
        mask_msg = messages.copy()  # get a copy of the messages
        
        # Debug anchor count
        for msg in mask_msg:
            if "anchors" in msg:
                # Debug pre-replacement content
                original_content = msg["content"]
                
                # Sort anchors by length (descending) to replace longest matches first
                anchors = sorted(msg["anchors"], key=len, reverse=True)
                
                for anchor in anchors:
                    if anchor in msg["content"]:
                        # Replace the anchor with mask token
                        msg["content"] = msg["content"].replace(anchor, mask_token)
                
                # Debug post-replacement content
                if original_content == msg["content"]:
                    print(f"WARNING: No anchors were replaced in message: {original_content[:50]}...")
                    print(f"Anchors: {anchors}")
        
        return mask_msg


def convert_to_tensor_format(inputs, device=None):
    # Case 1: Already a tensor in correct format
    if isinstance(inputs, torch.Tensor) and len(inputs.shape) == 2:
        if device is not None:
            inputs = inputs.to(device)
        return inputs
        
    # Case 2: Object with input_ids attribute
    if hasattr(inputs, 'input_ids'):
        inputs = inputs.input_ids
        
    # Case 3: Dictionary with input_ids key
    elif isinstance(inputs, dict) and 'input_ids' in inputs:
        inputs = inputs['input_ids']
        
    # Case 4: List of token IDs
    elif isinstance(inputs, list):
        inputs = torch.tensor([inputs], device=device)
        
    # Case 5: Single tensor but needs reshaping
    elif isinstance(inputs, torch.Tensor):
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)
            
    # Ensure it's on the correct device
    if isinstance(inputs, torch.Tensor) and device is not None:
        inputs = inputs.to(device)
        
    return inputs

def create_default_attention_mask(input_ids, device=None):
    """
    Creates a default attention mask (all 1s) for the given input_ids tensor.
    
    Args:
        input_ids (torch.Tensor): The input IDs tensor, shape (batch_size, seq_len)
        device: The device to place the attention mask on
        
    Returns:
        torch.Tensor: Attention mask with the same shape as input_ids, all values set to 1
    """
    # Ensure input_ids is on the right device if specified
    if device is not None and input_ids.device != device:
        input_ids = input_ids.to(device)
        
    # Create attention mask filled with 1s (all tokens attend to all positions)
    attention_mask = torch.ones_like(input_ids)
    
    return attention_mask

def spa_tokenize(prompt_with_anchors, global_anchors, tokenizer, device):
    
    # Set pad token if missing
    if tokenizer.pad_token is None:
        print("Setting pad token to EOS token")
        tokenizer.pad_token = tokenizer.eos_token
        # Remove reference to global model variable
        # model.config.pad_token_id = model.config.eos_token_id
    
    if tokenizer.mask_token:
        mask_token = tokenizer.mask_token
    else:
        mask_token = "MASKTOKEN"
    
    
    main_prompt, aux_prompt = format_spa_input(
            input=prompt_with_anchors,
            anchors=global_anchors, 
            mask_token=mask_token, 
            whole_word_only=False
        )
    
    
    # detect if tokenizer has chat_template
    if isinstance(main_prompt, list):
        # Expected for chat models
        # print("--- Message list processed by chat template")
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        
            main_inputs = tokenizer.apply_chat_template(
                main_prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
        
            aux_inputs = tokenizer.apply_chat_template(
                aux_prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            
        else:
            # non-chat models, need to convert to a string prompt
            # print("--- Message list processed by flat prompt")
            flat_prompt_main = ""
            for msg in main_prompt:
                flat_prompt_main += f"{msg['role']}: {msg['content']}\n"
            flat_prompt_main += "Assistant: "  # Add assistant prefix for generation
            
            flat_prompt_aux = ""
            for msg in aux_prompt:
                flat_prompt_aux += f"{msg['role']}: {msg['content']}\n"
            flat_prompt_aux += "Assistant: "  # Add assistant prefix for generation 
            
            # Tokenize the flattened prompts
            main_inputs = tokenizer(flat_prompt_main, return_tensors="pt").to(device)
            aux_inputs = tokenizer(flat_prompt_aux, return_tensors="pt").to(device)
            
    # User provides a string prompt
    elif isinstance(prompt_with_anchors, str):
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            # print("--- String prompt processed by chat template")
            
            # If user only provides a string prompt, we need to convert it to a chat prompt
            main_prompt = [{"role": "user", "content": main_prompt}]
            aux_prompt = [{"role": "user", "content": aux_prompt}]
            
            main_inputs = tokenizer.apply_chat_template(
                main_prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            
            aux_inputs = tokenizer.apply_chat_template(
                aux_prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            
        else:
            # non-chat models, need to convert to a string prompt
            # print("--- String prompt processed by flat prompt")
            main_inputs = tokenizer(main_prompt, return_tensors="pt").to(device)
            aux_inputs = tokenizer(aux_prompt, return_tensors="pt").to(device)
            
    else:
        raise ValueError("Invalid prompt format")
    
    # Make sure the returned input_ids follow the expected format: tensor([[1, 2, 3]], device='x')
    # Handle all possible tokenizer output formats
    
    main_inputs = convert_to_tensor_format(main_inputs, device)
    aux_inputs = convert_to_tensor_format(aux_inputs, device)
    
    return main_inputs, aux_inputs, mask_token


class SPALogitsProcessor(LogitsProcessor):
    """Processor that combines logits from a main and auxiliary model."""
    
    def __init__(self, aux_model, aux_input_ids, mask_token, strength=1.5, modulated_by_prob=True, tokenizer=None, use_attention_mask=True):
        self.aux_model = aux_model  # Same model, used for aux inputs
        self.aux_input_ids = aux_input_ids
        self.aux_past_key_values = None
        self.strength = strength
        self.modulated_by_prob = modulated_by_prob  # Whether to modulate weight by probability
        self.tokenizer = tokenizer  # Optional, for debug printing
        self.mask_token = mask_token  # Store mask_token
        # Store the device of the input_ids to use consistently
        self.device = aux_input_ids.device
        self.use_attention_mask = use_attention_mask
        if self.use_attention_mask:
            self.attention_mask = create_masked_attention(self.aux_input_ids, [mask_token], self.tokenizer)
        else:
            self.attention_mask = None
        
    def __call__(self, input_ids, scores):
        # Get aux model outputs for the current step
        if self.aux_past_key_values is None:
            # First step, run on full aux prompt
            aux_outputs = self.aux_model(
                input_ids=self.aux_input_ids, 
                use_cache=True, 
                return_dict=True,
                attention_mask=self.attention_mask
            )
            self.aux_past_key_values = aux_outputs.past_key_values
            aux_logits = aux_outputs.logits[:, -1, :]
        else:
            # Subsequent steps, run only on new token with past_key_values
            last_token = input_ids[:, -1].unsqueeze(-1).to(self.device)  # Ensure same device
            # For subsequent tokens, we don't need to pass the attention mask
            aux_outputs = self.aux_model(
                input_ids=last_token,
                past_key_values=self.aux_past_key_values,
                use_cache=True,
                return_dict=True
            )
            self.aux_past_key_values = aux_outputs.past_key_values
            aux_logits = aux_outputs.logits[:, -1, :]
        
        # Special case: strength = 1 means use only main logits
        if abs(self.strength - 1.0) < 1e-4:
            return scores
        
        # if strength is 0, return the aux logits
        if abs(self.strength - 0.0) < 1e-4:
            return aux_logits
            
        # Ensure scores and aux_logits are on the same device
        if scores.device != aux_logits.device:
            aux_logits = aux_logits.to(scores.device)
        
        # Check for NaNs in the inputs
        if torch.isnan(scores).any() or torch.isnan(aux_logits).any():
            print("Warning: NaN values detected in input scores or aux_logits")
            scores = torch.nan_to_num(scores, nan=0.0)
            aux_logits = torch.nan_to_num(aux_logits, nan=0.0)
        
        # Calculate the difference between main and aux logits
        diff = scores - aux_logits
        
        # Calculate the base weight
        base_weight = self.strength - 1.0
        
        # Modulate the weight by probability if enabled
        # Only do this when strength > 1 (that's what can cause random behavior. If -1 < strength < 1, it is semantic dimishment, disable this for more precise control)
        if self.modulated_by_prob and (self.strength > 1 or self.strength < -1):
            # Convert logits to probabilities with temperature scaling for stability
            temperature = 1.0
            scaled_logits = scores / temperature
            main_probs = F.softmax(scaled_logits, dim=-1)
            
            # Clamp probabilities to avoid numerical issues
            main_probs = torch.clamp(main_probs, min=1e-6, max=1.0)
            
            # Each token's weight is scaled by its probability
            
            # get the max probability
            max_prob = torch.max(main_probs)
            # normalize the base weight by the max probability
            base_weight = base_weight / max_prob
            # get different weights for each token based on their main probability
            token_weights = base_weight * main_probs
            
            # Apply the weighted adjustment
            adjustment = token_weights * diff
            
            # Clamp the adjustment to avoid extreme values
            adjustment = torch.clamp(adjustment, min=-1e2, max=1e2)
            
            # Compute final scores
            final_scores = scores + adjustment
        else:
            # Safe computation of weighted difference
            weighted_diff = base_weight * diff
            # Check for and handle any NaNs that might have appeared
            weighted_diff = torch.nan_to_num(weighted_diff, nan=0.0)
            # Clamp to avoid extreme values
            weighted_diff = torch.clamp(weighted_diff, min=-1e3, max=1e3)
            final_scores = scores + weighted_diff

        
        # Final stability check
        final_scores = torch.clamp(final_scores, min=-1e3, max=1e3)
        
        return final_scores




    