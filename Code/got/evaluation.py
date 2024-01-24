from datasets import Dataset, load_dataset
import pandas as pd
from tqdm import tqdm
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from evaluate.visualization import radar_plot
import torch
from typing import List
import logging
import os
import numpy as np
import random
from reaction_condition_prediction import get_dataset
import torch
import numpy as np
from get_models import *
from templates import *

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(4096)

def yield_prediction_evaluation():
    
    data_dir = "/data/yanyuliang/Code/got/data/chem_data/Suzuki.npz"
    data = np.load(data_dir, allow_pickle=True)
    dataset = data['data_df']
    dataset = pd.DataFrame(dataset, columns=['reaction', 'yield'])
    dataset = Dataset.from_pandas(dataset)
    print(dataset)
    print(len(dataset))
    print(dataset[0])

def uspto_dataset_split():
    
    # data_dir = '/data/yanyuliang/Code/got/data/chem_data/llama_chem_v1.json'
    # dataset = load_dataset('json', data_files=data_dir, split='train')
    dataset = get_dataset()
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=4096)
    train_set = dataset['train']
    test_set = dataset['test']
    train_set.save_to_disk('/data/yanyuliang/Code/got/data/chem_data/uspto_train')
    test_set.save_to_disk('/data/yanyuliang/Code/got/data/chem_data/uspto_test')

def get_orderly_dataset() -> Dataset:
    
    raw_train_data = pd.read_parquet("./data/chem_data/orderly_condition_train.parquet")
    train_dataset = Dataset.from_pandas(raw_train_data, split='train')
    raw_test_data = pd.read_parquet("./data/chem_data/orderly_condition_test.parquet")
    test_dataset = Dataset.from_pandas(raw_test_data, split='test')
    
    return train_dataset, test_dataset

def process_orderly_dataset(dataset,
                            save_data=False
                            ):
    
    if os.path.exists("./data/chem_data/orderly_train"):
        dataset = Dataset.load_from_disk("./data/chem_data/orderly_train")
        return dataset
    
    raw_data = []
    for i in tqdm(range(len(dataset))):
        data = {}
        example = dataset[i]
        data['reaction'] = "Here is a chemical reaction. Reactants are: " + example['reactant_000'] + ("" if example['reactant_001'] == "NULL" else ", " + example['reactant_001']) + ". Product is: " + example['product_000']
        data['condition'] = "The reaction conditions of this reaction are: " + "Agents: " + example['agent_000'] + ("" if example['agent_001'] == None else ", " + example['agent_001']) + ("" if example['agent_002'] == None else ", " + example['agent_002']) + ". Solvents: " + ("" if example['solvent_000'] == None else example['solvent_000']) + ("" if example['solvent_001'] == None else ", " + example['solvent_001'])
        data['temperature'] = example['temperature'] 
        raw_data.append(data)
        
    raw_data = pd.DataFrame(raw_data, columns=['reaction', 'condition', 'temperature'])
    dataset = Dataset.from_pandas(raw_data, split='train')
    
    if save_data:
        dataset.save_to_disk("./data/chem_data/orderly_train")

    return dataset

def reaction_prediction_evaluation():
    
    pass

if __name__ == '__main__':
    
    # yield_prediction_evaluation()
    # uspto_dataset_split()
    # train_dataset = Dataset.load_from_disk('/data/yanyuliang/Code/got/data/chem_data/uspto_train')
    # test_dataset = Dataset.load_from_disk('/data/yanyuliang/Code/got/data/chem_data/uspto_test')
    # print(train_dataset[0])
    # print(train_dataset[1000])
    
    train_dataset, test_dataset = get_orderly_dataset()
    print(train_dataset[50]['reactant_000'])
    print(train_dataset[50]['reactant_001'])
    print(train_dataset[50]['rxn_str'])
    print(train_dataset[50]['product_000'])
    print(train_dataset[50])
    print(train_dataset.features)
    
    
    # print(list(map(process_orderly_dataset, train_dataset)))
    # print(train_dataset[5])
    data = process_orderly_dataset(train_dataset)
    print(data[5000])
    print(len(data))
    print(data.features)
    
    model = init_llama_model()
    prompt = GENERAL_CONDITION_TEMPLATE + IN_CONTEXT_LEARNING_CONDITION_TEMPLETE.format(data[5230]['reaction'])
    # prompt = "你知道生辰八字吗？请给我解释一下。"
    message = get_message_for_llama(prompt)
    prediction = model.chat_completion(
                    dialogs=message,
                    temperature=0.2,
                    max_gen_len=1048
                )[0]['generation']['content']
    print(prediction)
    print(data[5230]['condition'])