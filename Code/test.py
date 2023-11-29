import pandas as pd
import json
import torch
import re
import replicate
from transformers import LlamaForCausalLM, LlamaTokenizer

if __name__ == '__main__':
    model = LlamaForCausalLM.from_pretrained("./output/llama2-7b-chat")
    tokenizer = LlamaTokenizer.from_pretrained("./output/llama2-7b-chat")