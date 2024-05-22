# Selective Prompt Anchoring (SPA)

This repo includes all the code for **S**elective **P**rompt **A**nchoring (SPA).

**weighted_text_utils.py** includes all the utils and helper functions. Necessary instructions are inside the code file.

**anchor_evaluation.py** serves as the main entry for evaluation. Users can use hyperparameters to specify the evaluation conditions:
1. `checkpoint`: Specify the model path. It can be the HuggingFace checkpoint name.
2. `benchmark`: Specify the benchmark name. For example, *HumanEval* or *MBPP*.
3. `mask_test_case`: If this is true, the test cases will be masked and included in the anchored text.
4. `mask_entire_prompt`: If this is true, the entire prompt will be masked as the anchored text.
5. `mode`: Specify the evaluation mode. Just input "*instruction*" or leave blank for base SPA.
6. `weight`: Specify the weighting value **ω**. **0** means simply using the original model. **<0** means reducing the impact of anchored text, while **>0** means amplifying the impact.
7. `approach`:  Just input "*difference*" or leave blank for base SPA.
8. `note`: Add notes to this experiment. The note text will appear as the prefix of output files (log file and generated data file).
9. `task_id`: If there are multiple GPUs, define the GPU ID to use. 
10. `use_multiple_gpu`: If this is true, evaluation will be conducted on multiple GPUs. You can specify which GPUs to use in [this line](https://github.com/magic-YuanTian/Selective-Prompt-Anchoring/blob/4b637f41e76f5a385aaa51ca4db0ab97859588e0/anchor_evaluation.py#L99).
