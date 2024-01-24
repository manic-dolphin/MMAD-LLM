import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
# import utils
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import LlamaForCausalLM, LlamaTokenizer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

if __name__ == '__main__':
    
    tokenizer = LlamaTokenizer.from_pretrained('./hf_models/llama2/llama2-7b-chat')
    # model = LlamaForCausalLM.from_pretrained('./hf_models/llama2/llama2-7b-chat', device_map='auto')
    # print(model)
    print(tokenizer.pad_token)