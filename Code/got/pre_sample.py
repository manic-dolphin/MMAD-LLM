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

if __name__ == '__main__':
    os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
    
    dataset = load_dataset("yuyuc/chem-uspto", cache_dir='/data/yanyuliang/Code/got/data')
    # dataloader = DataLoader(dataset, batch_size=10)
    l = len(dataset)
    
    pm = Planner_Model('llama2/llama-2-7b-chat/',
                       'llama2/tokenizer.model',
                       max_seq_len=4096
                       )
    model = pm.get_model()
    planner_model = Planner(model=model,
                            max_gen_len=1024,
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
                            max_gen_len=1024,
                            subject='chem'
                            )
            score = operations.score(initial_thought)
            scores.append(score)
        
        average_scores.append(sum(scores) / len(scores))
        
    print("average scores: {}.".format(average_scores))
    saved_scores = np.array(average_scores)
    np.save("scores.npy", saved_scores)