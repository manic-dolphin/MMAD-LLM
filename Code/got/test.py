import re
import unittest
import pandas as pd
from colorama import init, Fore, Back, Style

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from templates import *
from evaluation import hf_orderly_dataset

train, test = hf_orderly_dataset()
raw_train_data = pd.read_parquet("./data/chem_data/orderly_condition_train.parquet")
raw_train_dataset = Dataset.from_pandas(raw_train_data)
knowledge_model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/')

class TestModel(unittest.TestCase):
    
    def run_ft_model_test(self):
        
        model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/output_step1_llama2_7b_lora/', device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/output_step1_llama2_7b_lora/')
        
        prompt = "Here is a chemical reaction. Reactants are: Brc1ccccc1Br, CCCC[Sn](Cl)(CCCC)CCCC. Product is: CCCC[Sn](CCCC)(CCCC)c1ccccc1Br"
        model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
        output_ids = model.generate(model_inputs.input_ids, max_new_tokens=96) # max_new_tokens=96
        output = tokenizer.batch_decode(output_ids)[0]
        print(Fore.RED + output)
        
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
        
        print(Fore.GREEN + "Predict agents: {}.".format(agents_pred))
        print(Fore.GREEN + "Predict solvents: {}.".format(solvents_pred))
    
    def run_kot_model_test(self, 
                           reaction,
                           knowledge_model
                           ):
        
        # knowledge_model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/', device_map='auto')
        # tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/')
        # print(tokenizer.bos_token_id)
        # print(tokenizer.eos_token_id)
        # print(tokenizer.pad_token_id)
        # generation_config = GenerationConfig(
        #     max_new_tokens=512,
        #     do_sample=True,
        #     # top_p=0.9,
        #     repetition_penalty=2.0,
        #     pad_token_id=tokenizer.eos_token_id
        # )
        
        # reaction = "Here is a chemical reaction. Reactants are: Brc1ccccc1Br, CCCC[Sn](Cl)(CCCC)CCCC. Product is: CCCC[Sn](CCCC)(CCCC)c1ccccc1Br"
        # prompt = REACTION_CONDITION_COT_0.format(reaction)
        prompt = EXTRACT_KNOWLEDGE.format(reaction)
        model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
        output_ids = knowledge_model.generate(model_inputs.input_ids, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id) # max_new_tokens=96
        cot_0 = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        # cot_0 = [cot.strip() for cot in cot_0.split("*") if cot.strip()]
        # remove prompt
        # cot_0 = cot_0[1:]
        # cot_0 = [cot.split('\n')[0] for cot in cot_0]
        # print(output_ids)
        # print(Fore.YELLOW + cot_0)
        # response_start = cot_0.find("Now, give me your response:")
        # response = cot_0[response_start + len("Now, give me your response:"):].strip()
        print(cot_0)
        
    
    def run_kot_ft_model_test(self, reaction):
        
        model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/output_step1_llama2_7b_lora/', device_map='auto')
        knowledge_model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/', device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/output_step1_llama2_7b_lora/')
        
        # reaction = "Here is a chemical reaction. Reactants are: Brc1ccccc1Br, CCCC[Sn](Cl)(CCCC)CCCC. Product is: CCCC[Sn](CCCC)(CCCC)c1ccccc1Br"
        
        prompt = REACTION_CONDITION_COT_0.format(reaction)
        model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
        output_ids = knowledge_model.generate(model_inputs.input_ids, max_new_tokens=1024) # max_new_tokens=96
        cot_0 = tokenizer.batch_decode(output_ids)[0]
        print(Fore.YELLOW + cot_0)
        
        prompt = REACTION_CONDITION_COT_1.format(reaction)
        model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
        output_ids = knowledge_model.generate(model_inputs.input_ids, max_new_tokens=1024) # max_new_tokens=96
        cot_1 = tokenizer.batch_decode(output_ids)[0]
        print(Fore.YELLOW + cot_1)
        
        prompt = REACTION_CONDITION_COT_2.format(reaction)
        model_inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
        output_ids = knowledge_model.generate(model_inputs.input_ids, max_new_tokens=1024) # max_new_tokens=96
        cot_2 = tokenizer.batch_decode(output_ids)[0]
        print(Fore.YELLOW + cot_2)
        
        # cot_prompt = cot_0.strip() + cot_1.strip() + cot_2.strip() + "  " + reaction
        cot_prompt = cot_0 + cot_1 + cot_2 + "  " + reaction
        model_inputs = tokenizer(cot_prompt, return_tensors='pt').to("cuda")
        output_ids = model.generate(model_inputs.input_ids, max_new_tokens=96) # max_new_tokens=96
        output = tokenizer.batch_decode(output_ids)[0]
        print(Fore.RED + output)
        
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
        
        print(Fore.GREEN + "Predict agents: {}.".format(agents_pred))
        print(Fore.GREEN + "Predict solvents: {}.".format(solvents_pred))

class TestModelGenerate(unittest.TestCase):
    
    def test_generate(self):
        model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/output_step1_llama2_7b_lora/', device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/output_step1_llama2_7b_lora/')
        reaction = "Here is a chemical reaction. Reactants are: Brc1ccccc1Br, CCCC[Sn](Cl)(CCCC)CCCC. Product is: CCCC[Sn](CCCC)(CCCC)c1ccccc1Br"
        
        
class GnnTest(unittest.TestCase):
    
    def test_concept_extract(self):
        model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/')

if __name__ == '__main__':
    init()
    TestModel = TestModel()
    # TestModel.run_ft_model_test()
    # TestModel.run_kot_ft_model_test()
    # for i in range(100):
        # reaction = train[i]['reaction']
        # print(raw_train_data.columns)
        # example = raw_train_dataset[i]
        # print(example)
        # reactants = [example[reactant] for reactant in ['reactant_000', 'reactant_001'] if example[reactant] != None]
        # products = [example[product] for product in ['product_000']]
        # print(reactants)
        # print(products)
        # agents = [example[agent] for agent in ['agent_000', 'agent_001', 'agent_002'] if example[agent] != None]
        # solvents = [example[solvent] for solvent in ['solvent_000', 'solvent_001'] if example[solvent] != None]
        # TestModel.run_kot_model_test(reaction=reaction, knowledge_model=knowledge_model)
    
    # GnnTest = GnnTest()
    # GnnTest.test_embedding()
    data = Dataset.load_from_disk("./data/chem_data/orderly_train")
    # r = "CN(CC(F)c1ccc(F)cc1)S(=O)(=O)c1ccc(Br)s1. Functional group 1: Amine group (–NH2)"