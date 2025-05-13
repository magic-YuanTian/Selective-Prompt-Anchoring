

#########################################################
###               Example for pipeline                ###
#########################################################

from spa_pipeline import *


if __name__ == "__main__":
    
    
    # Register the pipeline
    register_spa_pipeline()
    
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Create the pipeline
    spa_pipe = pipeline(
        "selective-prompt-anchoring",
        model=model_name,
        anchoring_strength=3,
        modulated_by_prob=False,
        use_attention_mask=True,
        device_map="auto",
        skip_special_tokens=False
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
    
    
    # Our code supports both streaming and non-streaming generation.
    # You can choose one of them, depending on your needs.
    
    #################### Generation option 1 #############
    print('\n\nNow we are using the non-streaming generation...')
    output = spa_pipe(prompt_with_anchors, anchors=global_anchors, max_new_tokens=100)
    print('-'*100)
    print(output["generated_text"])
    
    
    #################### Generation option 2 #############
    print('\n\nNow we are using the streaming generation...')
    print('-'*100)
    for token in spa_pipe(prompt_with_anchors, anchors=global_anchors, max_new_tokens=1024, stream=True):
        print(token, end="", flush=True)
    print('\n' + '-'*100)
    
    

    
    