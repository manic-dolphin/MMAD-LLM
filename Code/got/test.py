import json
import pandas as pd
import itertools
from typing import List, Iterator
import torch
import numpy as np
import datasets
from datasets import load_dataset
# datasets.config.HF_DATASETS_OFFLINE = True
import os
import transformers

if __name__ == '__main__':
    # ids: Iterator[int] = itertools.count(0)
    # print(type(ids))
    # for i in range(5):
    #     print(next(ids))
    # print(list(set(np.array(torch.randint(1, 5, (5,))))))
    # print(type(str([1, 3, 3, 4])))
    # print("Instruction: Considering the initial solution along with the original problem, rate the current solution on a scale of 0 to 10, returning a single numerical value. The initial problem is {}. Here is the current solution: {}. Your score is:".format(1, 2))
    # x = 'Score: 8/10'
    # print(int(x[7]))
    # a = [1, 2, 3, 4, 5, 6, 7, 8]
    # print(a[-2:])
    # x = None
    # print(type(x))
    # print(x is None)
    # y = [1, 2, 3]
    # print(y is None)
    os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
    # dataset = load_dataset("yuyuc/chem-uspto", cache_dir='/data/yanyuliang/Code/got/data')
    # print(dataset['train'][0]['INSTRUCTION'])
    x = torch.tensor([1, 2, 3]).to('cuda')
    print(x)
    print(torch.__version__)