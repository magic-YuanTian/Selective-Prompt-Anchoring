<h1 align="center">Selective Prompt Anchoring</h1>


<p align="center">
  <a href="https://pypi.org/project/anchoring/"><img src="https://img.shields.io/pypi/v/anchoring.svg" alt="PyPI"></a>
  <a href="https://huggingface.co/DoctorChaos/selective-prompt-anchoring"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-SPA-yellow" alt="Hugging Face"></a>
  <a href="https://arxiv.org/abs/2408.09121"><img src="https://img.shields.io/badge/arXiv-2408.09121-b31b1b.svg" alt="arXiv"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

**Selective Prompt Anchoring (SPA)**  is a model-agnostic algorithm designed for large language models (LLMs) that provides fine-grained control over text generation.

This is the official repository for the 📄 **ICML 2025** paper: [Selective Prompt Anchoring for Code Generation](https://arxiv.org/abs/2408.09121).

## 🤔 Why use SPA?

In human communication, nuanced emphasis and fine-grained implications are often conveyed through variations in volume, tone, or pauses. Conveying such subtleties in text-based communication with AI is challenging with plain text prompts.

✨ SPA enables users to assign importance, emphasis, or weights to specific parts of input text when prompting language models. SPA brings this capability to text-based AI communication by allowing users to "anchor" (the name is inspired by *anchoring effect* in psychology) certain words or phrases in the prompt, causing the model to pay more attention to them during generation. With SPA, users can flexibly steer LLMs' attention through a general, easy-to-use API.

📢 *Note: While we currently work on text-to-text generation and evaluating on code generation in our paper, the underlying idea can be applied to other tasks (e.g., classification) or with other modalities (image).*


## 💡 How SPA Works

SPA creates two parallel processing paths:
1. The original prompt
2. A modified prompt with anchored tokens masked

During token generation, SPA compares logits from both paths and adjusts final probabilities based on the anchoring strength, causing the model to emphasize the anchored concepts while maintaining coherent generation.


## 💻 Installation

### From PyPI (⭐️ Recommended)

Install directly from PyPI using `pip`:

```bash
pip install anchoring
```

For specific versions or more details, see the [official package page](https://pypi.org/project/anchoring/).

### From GitHub

```bash
git clone https://github.com/your-username/selective-prompt-anchoring.git
cd selective-prompt-anchoring

pip install -e .
```

### From HuggingFace

```
pip install huggingface_hub
pip install git+https://github.com/magic-YuanTian/Selective-Prompt-Anchoring.git
```

Check [this page](https://huggingface.co/DoctorChaos/selective-prompt-anchoring) for more details.

## ⚡ Quick Start with Pipeline

Quick start with pipeline API:

```python
from transformers import pipeline
import anchoring

pipe = pipeline(
    "selective-prompt-anchoring",
    model="meta-llama/Llama-3.1-8B-Instruct",
)

output = pipe("How is the weather today?", anchors=['today'])
print(output["generated_text"])
```


A bit more settings:

```python
from transformers import pipeline
import anchoring  # The pipeline is automatically registered on import

# Create pipeline
spa_pipe = pipeline(
    "selective-prompt-anchoring",
    model="meta-llama/Llama-3.1-8B-Instruct",
    anchoring_strength=3.0,
    modulated_by_prob=True,
    use_attention_mask=True,
    device_map="auto"
)

# Simple text prompt with global anchors
prompt = "How is the weather today?"
global_anchors = ['today']

output = spa_pipe(prompt, anchors=global_anchors, max_new_tokens=1024)
print(output["generated_text"])
```

### Or you can stream the output

SPA supports streaming for real-time generation:

```python
# Get streaming output
for token in spa_pipe(prompt, anchors=global_anchors, max_new_tokens=1024, stream=True):
    print(token, end="", flush=True)
print()
```


### 🛠️ Alternative: Direct Usage with `model.generate()`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from anchoring import SPALogitsProcessor, spa_tokenize

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Define anchors and prompt
global_anchors = ['today']
prompt = "How is the weather today?"

# Tokenize with SPA
main_inputs, aux_inputs, mask_token = spa_tokenize(
    prompt_with_anchors=prompt,
    global_anchors=global_anchors,
    tokenizer=tokenizer,
    device=model.device
)

# Create SPA logits processor
spa_processor = SPALogitsProcessor(
    aux_model=model, 
    aux_input_ids=aux_inputs, 
    strength=3.0,
    modulated_by_prob=False,
    use_attention_mask=True,
    mask_token=mask_token,
    tokenizer=tokenizer
)

# Generate text with SPA
output_sequences = model.generate(
    input_ids=main_inputs,
    attention_mask=torch.ones_like(main_inputs),
    logits_processor=[spa_processor],
    max_new_tokens=1024,
    do_sample=False,
)

# Decode and print
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(generated_text)
```

<details>
<summary>Batch Processing Examples</summary>

```python
# Define a list of prompts
prompts = ["What's the weather <anchor>today</anchor>?", "What's the weather <anchor>tomorrow</anchor>?"]

# Or with chat format
prompts = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather <anchor>today</anchor>?"}
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather <anchor>tomorrow</anchor>?"}
    ]
]

# Process all prompts
outputs = spa_pipe(prompts, anchors=['weather'], max_new_tokens=1024)
for output in outputs:
    print(output["generated_text"])
```
</details>

## 📝 Input Formats

Our code supports multiple input formats, allowing developers to conveniently represent anchors in prompts or messages. Developers can use inline paired tags, `<anchor> </anchor>`, or a global anchor list to denote anchored text. 
They can also work with chat messages in a list, following the [OpenAI API standard](https://huggingface.co/docs/text-generation-inference/en/messages_api), or simply use a prompt string.

<details>
<summary>1️⃣ String with Global Anchors</summary>

```python
prompt = "How is the weather today?"
global_anchors = ['today']
```
</details>

<details>
<summary>2️⃣ String with Inline Anchors</summary>

```python
prompt = "What's the weather <anchor>today</anchor>? Think <anchor>step by step</anchor>."
```
</details>

<details>
<summary>3️⃣ Chat Messages with Message-Level Anchors</summary>

```python
prompt = [
    {
        "role": "system", 
        "content": "You are a helpful assistant.", 
        "anchors": ["You", "assistant"]  
    },
    {
        "role": "user",
        "content": "What's the weather today?", 
        "anchors": ["today"]
    },
]
```
</details>

<details>
<summary>4️⃣ Chat Messages with Inline Anchors</summary>

```python
prompt = [
    {
        "role": "system", 
        "content": "You are a helpful assistant."
    },
    {
        "role": "user", 
        "content": "What's the weather <anchor>today</anchor>?"
    },
]
```
</details>



### **➡️ Two example usage scripts are included under the [example folder](https://github.com/magic-YuanTian/Selective-Prompt-Anchoring/tree/main/examples)!** 


## ⚙️ Key Parameters

### SPA-Specific Parameters

- `strength` (default: 1.4): Controls the influence of anchored text.
  - `1.0`: No effect (normal generation)
  - `0.0`: Completely ignore anchored text
  - `>1.0`: Emphasize anchored text (higher values = stronger emphasis)
  - `<0.0`: Reverse the influence of the anchored text (negative values = stronger reversed influence)

- `modulated_by_prob` (default: True): When True, the anchoring strength is modulated by token probability.
  - Enable for more stable results, especially with higher anchoring strengths
  - Disable for more precise control at lower strengths

- `use_attention_mask` (default: True): When True, uses attention masking for anchor tokens, enhancing the effect of anchoring.

### Standard Generation Parameters

SPA supports all standard Hugging Face generation parameters, such as:

- `max_new_tokens`: Maximum number of tokens to generate
- `do_sample`: Whether to use sampling for generation
- `temperature`: Controls randomness (higher = more random)
- `top_p`: Top-p sampling parameter (nucleus sampling)
- `top_k`: Top-k sampling parameter
- `min_new_tokens`: Minimum number of tokens to generate

For more parameters, please check the official [Huggingface Transformers' generation documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation).


## 🧩 Practical Hyperparameter Settings

1. `strength` (Anchoring Strength): 
   - When you want to increase the model's attention/text emphasis
       - If `modulated_by_prob = True`, you can give a relatively high value of **anchoring strength** (e.g., 20).
       - If `modulated_by_prob = False`, we recommend a value less than 2.
       - If you are pursuing an optimal value, you can easily tune this value through grid search on your benchmark. Our experiment demonstrates that this value follows a simple pattern (as value increases, performance first improves, then declines), and it is easy to tune by dozens of examples.
   - For reducing (`0 < anchoring_strength < 1`) or reversing (`anchoring_strength < 0`), please set the value based on your concrete needs.

2. `modulated_by_prob` (Weight influence by token probabilities): We recommend setting `modulated_by_prob=True` for stable results. Set it as False if you aim for precise control or have other development needs.  

3. `use_attention_mask` (whether to use attention mask or just special token masking): Set `True` by default for more reliable performance, unless you detect any performance issue, you can set it as `False`, SPA supports a backup masking strategy by special tokens.
   
## Model Compatibility

SPA is a model-agnostic algorithm. Our implementation inherits the [Huggingface Transformers](https://github.com/huggingface/transformers) generation API. It should work for **any** LLM in [Huggingface model collections](https://huggingface.co/models). Please follow the corresponding model documentation for detailed instructions.


## 📜 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 📚 Citation

If you use SPA in your research, please cite:

```bibtex
@inproceedings{tian2025spa,
  title = {Selective Prompt Anchoring for Code Generation},
  author = {Tian, Yuan and Zhang, Tianyi},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year = {2025},
  url = {https://arxiv.org/abs/2408.09121},
  eprint = {2408.09121},
  archivePrefix = {arXiv}
}
```
