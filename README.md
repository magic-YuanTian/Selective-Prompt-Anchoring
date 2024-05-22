# Selective Prompt Anchoring (SPA)

This repo includes all the code for **S**elective **P**rompt **A**nchoring (SPA).

`weighted_text_utils.py` includes all the utils and helper functions. Necessary instructions are inside the code file.

`anchor_evaluation.py` serves as the main entry for evaluation. Users can use hyperparameters to specify the evaluation conditions:
    '--task_id', type=int, default=0, help="Define which GPU to use")
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
    
