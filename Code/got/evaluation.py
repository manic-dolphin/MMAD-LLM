import os
import re
import torch
import random
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
# from colorama import init, Fore

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import evaluate
from evaluate.visualization import radar_plot

from templates import *

logging.basicConfig(filename='./evaluation_kot_ft_0220.log',
                    filemode='a',
                    format='%(message)s',
                    level=logging.DEBUG
                    )
logger = logging.getLogger(__name__)

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

def get_orderly_dataset() -> Dataset:
    
    raw_train_data = pd.read_parquet("./data/chem_data/orderly_condition_train.parquet")
    train_dataset = Dataset.from_pandas(raw_train_data, split='train')
    raw_test_data = pd.read_parquet("./data/chem_data/orderly_condition_test.parquet")
    test_dataset = Dataset.from_pandas(raw_test_data, split='test')
    
    return train_dataset, test_dataset

def process_orderly_dataset(dataset,
                            save_data=False
                            ):
    
    if os.path.exists("./data/chem_data/orderly_test"):
        dataset = Dataset.load_from_disk("./data/chem_data/orderly_test")
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
        dataset.save_to_disk("./data/chem_data/orderly_test")

    return dataset

def hf_orderly_dataset(test_only=False):
    
    if test_only:
        return Dataset.load_from_disk("./data/chem_data/orderly_test")
    
    return Dataset.load_from_disk("./data/chem_data/orderly_train"), Dataset.load_from_disk("./data/chem_data/orderly_test")

def reaction_prediction_evaluation(setting: str,
                                   model,
                                   tokenizer,
                                   knowledge_model,
                                   sava_per_steps: int=200
                                   ):
    
    SETTINGS = ["4_shot_w/o_ft", "8_shot_w/o_ft", "ft", "kot_ft"]
    assert setting in SETTINGS
    logging.info("Evaluation setting: {}".format(setting))
    logging.info("----------------Starting evaluation-------------------")
    
    test_dataset = hf_orderly_dataset(test_only=True)
    l = len(test_dataset)
    raw_test_data = pd.read_parquet("./data/chem_data/orderly_condition_test.parquet")
    raw_test_dataset = Dataset.from_pandas(raw_test_data)
    assert l == len(raw_test_dataset)
    
    agents_number, agensts_true = 0, 0
    solvents_number, solvents_true = 0, 0
    result = {}
    
    for i in tqdm(range(l)):
        reaction = test_dataset[i]['reaction']
        condition = test_dataset[i]['condition']
        example = raw_test_dataset[i]
        agents = [example[agent] for agent in ['agent_000', 'agent_001', 'agent_002'] if example[agent] != None]
        solvents = [example[solvent] for solvent in ['solvent_000', 'solvent_001'] if example[solvent] != None]
        
        if setting in ["4_shot_w/o_ft", "8_shot_w/o_ft"]:
            
            prompt = GENERAL_CONDITION_TEMPLATE + IN_CONTEXT_LEARNING_CONDITION_TEMPLETE.format(reaction)
            model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
            output_ids = model.generate(model_inputs.input_ids, max_length=1048)
            output = tokenizer.batch_decode(output_ids)[0]
            # logging.info(output)
            
            # start_index = output.rfind("Agents:")
            start_index = output.rfind("Here is a chemical reaction.")
            input_string = output[start_index:]
            agents_match = re.search(r"Agents: (.*?)(?=\s*Solvents:|$)", input_string)
            agents_pred = agents_match.group(1).strip() if agents_match else None
            if agents_pred == None:
                agents_pred = ["null"]
            else:
                agents_pred = agents_pred.replace(' ', '').replace('.', '').split(",")
            
            end_index = input_string.rfind("Please")
            input_string = input_string[:end_index].strip()
            solvents_match = re.search(r"Solvents: (.*)$", input_string)
            # solvents_match = re.search(r"Solvents: (.*?)(?=\bPlease\b)", input_string)
            solvents_pred = solvents_match.group(1).strip() if solvents_match else None
            if solvents_pred == None:
                solvents_pred = ["null"]
            else:
                solvents_pred = solvents_pred.replace(' ', '').replace('.', "").split(",")
            
            # evaluate
            agents_number += len(agents)
            solvents_number += len(solvents)
            
            for agent in agents_pred:
                if agent in agents:
                    agensts_true += 1
                    
            for solvent in solvents_pred:
                if solvent in solvents:
                    solvents_true += 1
            
            logging.info("##############################################################")        
            logging.info("----------------------No {} test data-------------------------".format(i + 1))
            logging.info("Model's output: {}".format(output))
            logging.info("Predict agents: {}. Ground truth agents: {}.".format(agents_pred, agents))
            logging.info("Predict solvents: {}. Ground truth solvents: {}".format(solvents_pred, solvents))
            logging.info("##############################################################") 
            
            if i % sava_per_steps == 0:
                cur_agents_acc = agensts_true / agents_number
                cur_solvents_acc = solvents_true / solvents_number
                logging.info("----------------------Temp Result----------------------")
                logging.info("Current agents predict acc is {}, current solvents predict acc is {}.".format(cur_agents_acc, cur_solvents_acc))
                logging.info("-------------------------------------------------------")
        
        if setting == "ft":
            prompt = reaction
            model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
            output_ids = model.generate(model_inputs.input_ids, max_new_tokens=96) # max_new_tokens=96
            output = tokenizer.batch_decode(output_ids)[0]
            # logging.info(output)
            
            # start_index = output.rfind("Agents:")
            start_index = output.rfind("Here is a chemical reaction.")
            input_string = output[start_index:]
            agents_match = re.search(r"Agents: (.*?)(?=\s*Solvents:|$)", input_string)
            agents_pred = agents_match.group(1).strip() if agents_match else None
            if agents_pred == None:
                agents_pred = ["null"]
            else:
                agents_pred = agents_pred.replace(' ', '').replace('.', '').split(",")
            
            end_index = input_string.find("<")
            input_string = input_string[:end_index].strip()
            solvents_match = re.search(r"Solvents: (.*)$", input_string)
            # solvents_match = re.search(r"Solvents: (.*?)(?=\bPlease\b)", input_string)
            solvents_pred = solvents_match.group(1).strip() if solvents_match else None
            if solvents_pred == None:
                solvents_pred = ["null"]
            else:
                solvents_pred = solvents_pred.replace(' ', '').replace('.', "").split(",")
            
            # evaluate
            agents_number += len(agents)
            solvents_number += len(solvents)
            
            for agent in agents_pred:
                if agent in agents:
                    agensts_true += 1
                    
            for solvent in solvents_pred:
                if solvent in solvents:
                    solvents_true += 1
            
            logging.info("##############################################################")        
            logging.info("----------------------No {} test data-------------------------".format(i + 1))
            logging.info("Model's output: {}".format(output))
            logging.info("Predict agents: {}. Ground truth agents: {}.".format(agents_pred, agents))
            logging.info("Predict solvents: {}. Ground truth solvents: {}".format(solvents_pred, solvents))
            logging.info("##############################################################") 
            
            if i % sava_per_steps == 0:
                cur_agents_acc = agensts_true / agents_number
                cur_solvents_acc = solvents_true / solvents_number
                logging.info("----------------------Temp Result----------------------")
                logging.info("Current agents predict acc is {}, current solvents predict acc is {}.".format(cur_agents_acc, cur_solvents_acc))
                logging.info("-------------------------------------------------------")
        
        if setting == 'kot_ft':
            
            prompt = REACTION_CONDITION_COT_0.format(reaction)
            model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
            output_ids = knowledge_model.generate(model_inputs.input_ids, max_new_tokens=512)
            cot_0 = tokenizer.batch_decode(output_ids)[0]
            
            prompt = REACTION_CONDITION_COT_1.format(reaction)
            model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
            output_ids = knowledge_model.generate(model_inputs.input_ids, max_new_tokens=512)
            cot_1 = tokenizer.batch_decode(output_ids)[0]
            
            prompt = REACTION_CONDITION_COT_2.format(reaction)
            model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
            output_ids = knowledge_model.generate(model_inputs.input_ids, max_new_tokens=512) # max_new_tokens=1024
            cot_2 = tokenizer.batch_decode(output_ids)[0]
            
            cot_prompt = cot_0 + cot_1 + cot_2 + "  " + reaction
            model_inputs = tokenizer(cot_prompt, return_tensors='pt').to("cuda")
            output_ids = model.generate(model_inputs.input_ids, max_new_tokens=96) # max_new_tokens=96
            output = tokenizer.batch_decode(output_ids)[0]
            
            start_index = output.rfind("Here is a chemical reaction.")
            input_string = output[start_index:]
            agents_match = re.search(r"Agents: (.*?)(?=\s*Solvents:|$)", input_string)
            agents_pred = agents_match.group(1).strip() if agents_match else None
            if agents_pred == None:
                agents_pred = ["null"]
            else:
                agents_pred = agents_pred.replace(' ', '').replace('.', '').split(",")
            
            end_index = input_string.find("<")
            input_string = input_string[:end_index].strip()
            solvents_match = re.search(r"Solvents: (.*)$", input_string)
            # solvents_match = re.search(r"Solvents: (.*?)(?=\bPlease\b)", input_string)
            solvents_pred = solvents_match.group(1).strip() if solvents_match else None
            if solvents_pred == None:
                solvents_pred = ["null"]
            else:
                solvents_pred = solvents_pred.replace(' ', '').replace('.', "").split(",")
            
            # evaluate
            agents_number += len(agents)
            solvents_number += len(solvents)
            
            for agent in agents_pred:
                if agent in agents:
                    agensts_true += 1
                    
            for solvent in solvents_pred:
                if solvent in solvents:
                    solvents_true += 1
            
            logging.info("##############################################################")        
            logging.info("----------------------No {} test data-------------------------".format(i + 1))
            logging.info("Model's output: {}".format(output))
            logging.info("Predict agents: {}. Ground truth agents: {}.".format(agents_pred, agents))
            logging.info("Predict solvents: {}. Ground truth solvents: {}".format(solvents_pred, solvents))
            logging.info("##############################################################") 
            
            if i % sava_per_steps == 0:
                cur_agents_acc = agensts_true / agents_number
                cur_solvents_acc = solvents_true / solvents_number
                logging.info("----------------------Temp Result----------------------")
                logging.info("Current agents predict acc is {}, current solvents predict acc is {}.".format(cur_agents_acc, cur_solvents_acc))
                logging.info("-------------------------------------------------------")
            
    
    result['agents_acc'] = agensts_true / agents_number
    result['solvents_acc'] = solvents_true / solvents_number
    logging.info("----------------------Result----------------------")
    logging.info("Total number of agents: {}.".format(agents_number))
    logging.info("Total number of solvents: {}.".format(solvents_number))
    logging.info("Agents predict acc: {}.".format(agensts_true / agents_number))
    logging.info("Solvents predict acc: {}.".format(solvents_true / solvents_number))
    
    return result

if __name__ == '__main__':
    
    train_dataset, test_dataset = hf_orderly_dataset()
    
    # model = init_llama_model()
    # prompt = GENERAL_CONDITION_TEMPLATE + IN_CONTEXT_LEARNING_CONDITION_TEMPLETE.format(data[5230]['reaction'])
    # prompt = "你知道生辰八字吗？请给我解释一下。"
    # message = get_message_for_llama(prompt)
    # prediction = model.chat_completion(
    #                 dialogs=message,
    #                 temperature=0.2,
    #                 max_gen_len=1048
    #             )[0]['generation']['content']
    # print(prediction)
    # data_index = 1560
    # data = test_dataset
    # print(data[data_index]['reaction'] + data[data_index]['condition'])
    
    print("###############################################################################")
    # prompt = data[data_index]['reaction']
    # prompt = GENERAL_CONDITION_TEMPLATE + IN_CONTEXT_LEARNING_CONDITION_TEMPLETE.format(data[data_index]['reaction'])
    model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/output_step1_llama2_7b_lora/', device_map='auto')
    knowledge_model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/output_step1_llama2_7b_lora/')
    # tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/')
    # model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    # output_ids = model.generate(model_inputs.input_ids, max_length=2048)
    # output = tokenizer.batch_decode(output_ids)[0]
    # print(output)
    
    print("###############################################################################")
    reaction_prediction_evaluation("kot_ft",
                                   model=model,
                                   tokenizer=tokenizer,
                                   knowledge_model=knowledge_model
                                   )