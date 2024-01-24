import json
import pandas as pd
import itertools
from typing import List, Iterator
import torch
import numpy as np
import datasets
from datasets import Dataset
from datasets import load_dataset
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PretrainedConfig
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from transformers import PhiPreTrainedModel, PhiModel
import evaluate
import deepspeed
from typing import Optional, Union

def remove_embedding_layer(model):
    
    for _, param in enumerate(model.named_parameters()):
        print(param)
        
class Phi_Without_Embedding(PhiModel):
    
    def __init__(self, config):
        super().__init__(config)
    
    def forward(
    self,
    input_ids: torch.LongTensor,
    hidden_states,
    past_key_values:  None,
    attention_mask: Optional[torch.BoolTensor] = None,
                ) -> torch.FloatTensor:
        # hidden_states = self.embd(input_ids)
        hidden_states = hidden_states

        for layer in self.h:
            hidden_states = layer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )

        return hidden_states


if __name__ == '__main__':
    # bleu = evaluate.load('bleu')
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/hf_models/phi-2')
    model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/hf_models/phi-2', trust_remote_code=True).to("cuda")
    # model = Phi_Without_Embedding.from_pretrained('/data/yanyuliang/Code/got/hf_models/phi-2', trust_remote_code=True).to("cuda")
    model_1 = model
    # tokenizer = AutoTokenizer.from_pretrained('./Mistral-7B-Instruct-v0.1')
    # model = AutoModelForCausalLM.from_pretrained('./Mistral-7B-Instruct-v0.1')
    # tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/output/llama2-70b-chat')
    # model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/output/llama2-70b-chat')
    # ds_engine = deepspeed.init_inference(
    #     model,
    #     mp_size=1,
    #     dtype=torch.float32,
    #     replace_with_kernel_inject=True
    # )
    # model = ds_engine.module
    
    prompt = "hello, do you understand functional analysis?"
    # prompt = """
    # Here is a chemical reaction formula: Reactants are:aryl halide:CCOC(=O)C1=CC2=C(O1)C(=CC=C2)Br;amine:C1CN(CCN1)CCC2=CC=CC=N2, 
    # Reagents are:Base:C(=O)([O-])[O-].[Cs+].[Cs+];Solvent:C1COCCO1;
    # metal and ligand:CC(C)C1=CC(=C(C(=C1)C(C)C)C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C)C;
    # metal and ligand:C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.[Pd].[Pd], 
    # and Products are 0:CCOC(=O)C1=CC2=C(O1)C(=CC=C2)N3CCN(CC3)CCC4=CC=CC=N4, 
    # please give me the reaction condition of this formula.
    # """
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    hidden_states = torch.rand((4, 10, 2560)).to("cuda")
    # output = model.generate(**hidden_states, max_new_tokens=1024, do_sample=True)
    output = model(hidden_states)
    # print(output.shape)
    # text = tokenizer.batch_decode(output)[0]
    print(output.hidden_states.shape)
    print(output.hidden_states)
    
    
    #############################
    # remove_embedding_layer(model)