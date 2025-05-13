

#########################################################
###           Example for model.generate()            ###
#########################################################


from spa import *

if __name__ == "__main__":
    
    # --- Hyperparameters ---
    anchoring_strength = 5 # Weight towards the modified prompt's influence
    modulated_by_prob = True  # If True, the anchoring strength is weighted by token probability. Turn it on for stable results.
    use_attention_mask = True
    
    # Specify the model name
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Specify device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    
    print(f"\nModel name: {model_name}")
    print(f"Device: {device}")
    
    print(f"\nAnchoring_strength: {anchoring_strength}")
    print(f"Modulated by probability: {modulated_by_prob}")
    print(f"Use attention mask: {use_attention_mask}\n")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # configure the Huggingface model here
    model_kwargs = {
        # Configure the Huggingface model here
        "device_map": device
        # set other parameters here
    }
    
    # If you want to use quantization, add quantization (for CUDA devices)
    if device == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model directly using AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Define the prompt
    # We provide 4 input options, you can choose one of them, depending on your needs and the model type (i.e., chat or completion).

    #################### Input option 1 #############

    prompt_with_anchors = "What's the weather today?"
    
    global_anchors = ['today']

    #################### Input option 2 #############

    prompt_with_anchors = "What's the weather <anchor>today</anchor>?"
    
    global_anchors = []  # optional

    #################### Input option 3 #############

    prompt_with_anchors = [
        {
            "role": "system", 
            "content": "You are a helpful assistant.", 
            "anchors": ["assistant"]  
        },
        {
            "role": "user",
            "content": "What's the weather today?", 
            "anchors": ["today"]
        },
    ]
    
    global_anchors = []  # optional
    

    #################### Input option 4 #############

    prompt_with_anchors = [
        {
            "role": "system", 
            "content": "You are a helpful <anchor>assistant</anchor>."
        },
        {
            "role": "user", 
            "content": "What's the weather <anchor>today</anchor>?"
        },
    ]
    
    global_anchors = []  # optional
    
    #########################################################
    
    # Tokenize the input
    main_inputs, aux_inputs, mask_token = spa_tokenize(prompt_with_anchors, global_anchors, tokenizer, device)


    # Create SPA logits processor
    spa_processor = SPALogitsProcessor(
        aux_model=model, 
        aux_input_ids=aux_inputs, 
        strength=anchoring_strength,
        modulated_by_prob=modulated_by_prob,
        use_attention_mask=use_attention_mask,
        mask_token=mask_token,
        tokenizer=tokenizer,  # for debug
    )
    
    start_time = time.time()
    
    # Create attention mask for main_inputs
    default_attention_mask = create_default_attention_mask(main_inputs, device=device)
    
    # Generate the output
    output_sequences = model.generate(
        input_ids=main_inputs,
        attention_mask=default_attention_mask,
        logits_processor=[spa_processor],
        
        # Set standard Huggingface model parameters here, please refer to Huggingface documentation (https://huggingface.co/docs/transformers/en/main_classes/text_generation).
        # Below are some example parameters:
        max_new_tokens=1024,
        min_new_tokens=1,
        num_beams=1,
        do_sample=False,
        # top_k=50,
        # top_p=0.95,
        # temperature=0.0,
    )
    
    # Decode and print
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
    
    end_time = time.time()
    
    print("-" * 40)
    print(generated_text)
    print("-" * 40)

    
    # Calculate the number of new generated tokens
    num_input_tokens = main_inputs.shape[1]
    num_output_tokens = output_sequences.shape[1]
    num_new_tokens = num_output_tokens - num_input_tokens
    tokens_per_second = num_new_tokens / (end_time - start_time)
    
    
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"New tokens generated: {num_new_tokens}")
    print(f"Generation speed: {tokens_per_second:.2f} tokens/second")
    
    
    