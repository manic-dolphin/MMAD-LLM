from graph import *
from planner import *
from incontext_prompt import SCORE_EXAMPLES_PROMPTS
from models.models import Planner_Model
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
import numpy as np
import logging

if __name__ == '__main__':
    # os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    logging.basicConfig(filename="sample_test_1.log", 
                        filemode='a',
                        format='%(message)s',
                        level=logging.INFO
                        )
    
    # dataset = load_dataset("yuyuc/chem-uspto", cache_dir='/data/yanyuliang/Code/got/data')
    dataset = load_dataset('json', data_files='./data/chem_data/llama_chem_v1.json', split='train')
    dataset = dataset.train_test_split(0.2, shuffle=True)
    # dataloader = DataLoader(dataset, batch_size=10)
    l = len(dataset['train'])
    logging.info("Start to sample, the train set's size is {}.".format(l))
    
    
    pm = Planner_Model('llama2/llama-2-7b-chat/',
                       'llama2/tokenizer.model',
                       max_seq_len=2048
                       )
    model = pm.get_model()
    planner_model = Planner(model=model,
                            max_gen_len=768,
                            subject='chem'
                            )
    average_scores = []
    for i in tqdm(range(10000)):
        # sample numbers
        scores = []
        for j in range(10):
            indice = torch.randint(0, l, (1,))
            problem = dataset['train'][indice]['INSTRUCTION'][0]
            # print(problem)
            
            initial_state = planner_model.get_initial_state(problem)
            initial_thought = Thought(initial_state)
            operations = Operations(
                            model,
                            problem,
                            max_gen_len=64,
                            subject='chem'
                            )
            score = operations.score(initial_thought)
            print(score)
            scores.append(score)
        
        average_scores.append(sum(scores) / len(scores))
        if i % 20 == 0:
            saved_scores = np.array(average_scores)
            np.save("scores_1.npy", saved_scores)
        
    print("average scores: {}.".format(average_scores))
    # saved_scores = np.array(average_scores)
    # np.save("scores.npy", saved_scores)