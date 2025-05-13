from transformers import Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import pipeline
from transformers import TextIteratorStreamer
import threading

# Import SPA components from the original implementation
from spa import (
    SPALogitsProcessor,
    spa_tokenize,
    preprocess_anchors,
    create_default_attention_mask
)


class SPAPipeline(Pipeline):
    """
    Selective Prompt Anchoring Pipeline for text generation.
    This pipeline applies anchoring to influence the LLM's generation behavior.
    """
    
    def __init__(self, model, tokenizer, **kwargs):
        # Extract SPA-specific parameters from kwargs before initializing parent
        self.anchoring_strength = kwargs.pop("anchoring_strength", 1.4)
        self.modulated_by_prob = kwargs.pop("modulated_by_prob", True)
        self.use_attention_mask = kwargs.pop("use_attention_mask", True)
        # Call parent initializer after setting our attributes
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.device = model.device
        
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        
        # Extract and process parameters
        if "anchors" in kwargs:
            preprocess_kwargs["anchors"] = kwargs.pop("anchors", [])
        
        if "anchoring_strength" in kwargs:
            forward_kwargs["anchoring_strength"] = kwargs.pop("anchoring_strength", self.anchoring_strength)
            
        if "modulated_by_prob" in kwargs:
            forward_kwargs["modulated_by_prob"] = kwargs.pop("modulated_by_prob", self.modulated_by_prob)
            
        if "use_attention_mask" in kwargs:
            forward_kwargs["use_attention_mask"] = kwargs.pop("use_attention_mask", self.use_attention_mask)
            
        # Generation parameters
        if "max_new_tokens" in kwargs:
            forward_kwargs["max_new_tokens"] = kwargs.pop("max_new_tokens", 100)
            
        if "min_new_tokens" in kwargs:
            forward_kwargs["min_new_tokens"] = kwargs.pop("min_new_tokens", 1)
            
        if "do_sample" in kwargs:
            forward_kwargs["do_sample"] = kwargs.pop("do_sample", True)
            
        if "temperature" in kwargs:
            forward_kwargs["temperature"] = kwargs.pop("temperature", 0.7)
            
        if "top_p" in kwargs:
            forward_kwargs["top_p"] = kwargs.pop("top_p", 0.95)
            
        if "top_k" in kwargs:
            forward_kwargs["top_k"] = kwargs.pop("top_k", 50)
            
        # Streaming parameter
        if "stream" in kwargs:
            forward_kwargs["stream"] = kwargs.pop("stream", False)
            
        # Any remaining kwargs go to postprocessing
        postprocess_kwargs.update(kwargs)
        
        
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs
    
    def preprocess(self, inputs, anchors=None):
        """
        Preprocess the inputs and anchors for SPA.
        
        Args:
            inputs: Text or messages to generate from
            anchors: List of anchor strings to influence generation
            
        Returns:
            Dict with main_inputs, aux_inputs, and mask_token
        """

        # Default to empty list if anchors not provided
        if anchors is None:
            anchors = []
        
        # Special handling for chat message format
        # Check if this is a dict with 'role' and 'content' - a single message
        if isinstance(inputs, dict) and 'role' in inputs and 'content' in inputs:
            # It's a single message, convert to a list of messages
            inputs = [inputs]
            
        # Check if this is a list of dictionaries with 'role' and 'content' - a proper chat format
        elif isinstance(inputs, list) and len(inputs) > 0:
            if all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in inputs):
                # It's a proper chat message format, keep it as is
                pass
            # If it's the first element of the list that is a chat message, extract it
            elif len(inputs) == 1 and isinstance(inputs[0], list) and len(inputs[0]) > 0:
                # This handles the case where input is wrapped in an extra list: [[msg1, msg2]]
                if all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in inputs[0]):
                    inputs = inputs[0]
            # Otherwise, take the first item (default pipeline behavior)
            else:
                inputs = inputs[0]
        
        # Preprocess anchors
        anchors = preprocess_anchors(anchors)
        
        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Tokenize with SPA
        main_inputs, aux_inputs, mask_token = spa_tokenize(
            prompt_with_anchors=inputs,
            global_anchors=anchors,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        return {
            "main_inputs": main_inputs,
            "aux_inputs": aux_inputs,
            "mask_token": mask_token
        }
    
    def _forward(self, model_inputs, 
                anchoring_strength=1.4, 
                modulated_by_prob=True, 
                use_attention_mask=True,
                max_new_tokens=100,
                min_new_tokens=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                stream=False):
        """
        Run generation with SPA.
        
        Args:
            model_inputs: Dict containing main_inputs, aux_inputs, and mask_token
            anchoring_strength: How much to weight the anchored version
            modulated_by_prob: Whether to modulate strength by token probability
            use_attention_mask: Whether to use attention masking for anchor tokens
            max_new_tokens: Maximum number of tokens to generate
            min_new_tokens: Minimum number of tokens to generate
            do_sample: Whether to use sampling for generation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stream: Whether to stream the output
            
        Returns:
            Generated output sequences or a TextIteratorStreamer if stream=True
        """
        main_inputs = model_inputs["main_inputs"]
        aux_inputs = model_inputs["aux_inputs"]
        mask_token = model_inputs["mask_token"]
        
        # Store the original input length for later use
        input_length = main_inputs.shape[1]
        
        # Create SPA logits processor
        spa_processor = SPALogitsProcessor(
            aux_model=self.model,
            aux_input_ids=aux_inputs,
            strength=anchoring_strength,
            modulated_by_prob=modulated_by_prob,
            use_attention_mask=use_attention_mask,
            mask_token=mask_token,
            tokenizer=self.tokenizer
        )
        
        # Get attention mask if available
        if hasattr(main_inputs, 'attention_mask'):
            attention_mask = main_inputs.attention_mask
        else:
            # Create a default attention mask (all 1s) using the helper function 
            attention_mask = create_default_attention_mask(main_inputs, device=self.device)
        
        # Set up generation kwargs
        generation_kwargs = {
            "input_ids": main_inputs,
            "attention_mask": attention_mask,
            "logits_processor": [spa_processor],
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        
        if stream:
            # Set up streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_special_tokens=True, 
                skip_prompt=True
            )
            generation_kwargs["streamer"] = streamer
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self.model.generate, 
                kwargs=generation_kwargs
            )
            generation_thread.start()
            
            # Return streamer and input length for postprocessing
            return {"streamer": streamer, "input_length": input_length, "thread": generation_thread}
        else:
            # Normal generation (non-streaming)
            output_sequences = self.model.generate(**generation_kwargs)
            
            # Return the sequences along with the input length
            return {"sequences": output_sequences, "input_length": input_length}
    
    def postprocess(self, model_outputs, **kwargs):
        """
        Process the generated outputs.
        
        Args:
            model_outputs: Dict containing output sequences and input length,
                          or a streamer if streaming was enabled
            
        Returns:
            Dict with generated_text (only new content) and full_output (complete output),
            or a generator that yields tokens as they are generated
        """

        
        # Check if we're in streaming mode
        if "streamer" in model_outputs:
            streamer = model_outputs["streamer"]
            # Return a generator that yields tokens as they are generated
            def token_generator():
                for token in streamer:
                    yield token
            
            return token_generator()
        
        # Normal (non-streaming) mode
        # Determine whether to skip special tokens
        # By default, skip special tokens for generated_text but not for full_output
        skip_special_tokens = kwargs.get("skip_special_tokens", True)
        
        # Extract the sequences and input length
        sequences = model_outputs["sequences"]
        input_length = model_outputs["input_length"]
        
        # Get the full generated output (optionally with special tokens)
        full_output = self.tokenizer.decode(sequences[0], skip_special_tokens=skip_special_tokens)
        
        # Extract only the newly generated tokens (always skip special tokens for clarity)
        new_tokens = sequences[0][input_length:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return {"generated_text": generated_text, "full_output": full_output}

    def __call__(self, inputs, **kwargs):
        """
        Override the Pipeline's __call__ method to properly handle chat message lists
        
        Args:
            inputs: Input text, message dict, or list of message dicts
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text output
        """
        
        # Special handling for chat message lists
        # If we have a list of message dicts (proper chat format), handle it as a single input 
        if isinstance(inputs, list) and len(inputs) > 0:
            if all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in inputs):
                # We need to pass the entire message list as one unit to preprocess
                preprocess_args = {"inputs": inputs}
                # Extract any preprocessing arguments from kwargs
                if "anchors" in kwargs:
                    preprocess_args["anchors"] = kwargs.pop("anchors", None)
                
                # Call preprocess directly with our chat message list
                model_inputs = self.preprocess(**preprocess_args)
                forward_args = {**kwargs}
                model_outputs = self._forward(model_inputs, **forward_args)
                result = self.postprocess(model_outputs, **kwargs)
                return result
        
        # For other input types, use the parent's __call__ method
        return super().__call__(inputs, **kwargs)


# Register the pipeline
def register_spa_pipeline():
    """Register the SPA pipeline with the Hugging Face pipeline factory."""
    from transformers import AutoModelForCausalLM
    
    PIPELINE_REGISTRY.register_pipeline(
        "selective-prompt-anchoring",
        pipeline_class=SPAPipeline,
        pt_model=AutoModelForCausalLM
    )


# Example usage
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
    
    # Example prompts
    prompt = "I am an introverted person. How to describe my personality? Answer in 1 sentence."

    # Test with a regular chat message list
    prompt = [
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user", 
            "content": "What's the weather <anchor>today</anchor>?"
        }
    ]
    
    global_anchors = ['weather']
    
    # Example with one time
    output = spa_pipe(prompt, anchors=global_anchors, max_new_tokens=100)
    print('-'*100)
    print(output["generated_text"])
    
    
    # Example with streaming
    print('-'*100)
    for token in spa_pipe(prompt, anchors=global_anchors, max_new_tokens=1024, stream=True):
        print(token, end="", flush=True)
    print('\n' + '-'*100)
    
    
    
    # # batch generation
    
    # # Define a list of string prompts
    # prompts = ["What's the weather <anchor>today</anchor>?", "What's the weather <anchor>tomorrow</anchor>?"]
    
    # # Or you can use a list of messages
    # prompts = [
    #     [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "What's the weather <anchor>today</anchor>?"}
    #     ],
    #     [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "What's the weather <anchor>tomorrow</anchor>?"}
    #     ]
    # ]
    
    # outputs = spa_pipe(prompts, anchors=global_anchors, max_new_tokens=1024, stream=False)
    # print('-'*100)
    # for output in outputs:
    #     print(output["generated_text"])
    #     print('-'*100)
    
    
    