import os
import ast
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from gnn_untils import *

def load_json_file(data_dir):
    with open(data_dir, 'r') as file:
        data = json.load(file)
        
    return data

# TODO
def parsing_extract_knowledge(data):
    res = []
    for k, v in data.items():
        res.append(v)
    return res

def construct_dataset_for_gnn(raw_dataset, gnn_data):
    """Construct the dataset for trainning LLM whit Gnn.
    Added column include the "graph", in the form of graph{"x": , "edge_index": , "edge_attr": }.

    Args:
        raw_dataset (Dataset): the raw dataset for SFT training.
        gnn_data (Dataframe): the raw dataset extracted from the chemical reactions.
    """
    # raw_dataset = Dataset.load_from_disk('./data/chem_data/orderly_train')
    # gnn_data = pd.read_csv('./extract_data_using_knowledge/data_sample.csv')
    
    indexs_map = {}
    for i in range(len(gnn_data)):
        indexs_map.update({gnn_data.iloc[i]['original_idx'] : gnn_data.iloc[i]['reactions']})
    
    graph = []
    for i in tqdm(range(len(raw_dataset))):
        if i in indexs_map:
            graph.append(ast.literal_eval(indexs_map[i])[0])
        else:
            graph.append(["Failed"])
    
    new_dataset = raw_dataset
    new_dataset = new_dataset.add_column(name='graph_knowledge', column=graph)
    filtered_dataset = new_dataset.filter(lambda example: example['graph_knowledge'] != ['Failed'])
    
    filtered_dataset.save_to_disk("./data/chem_data/orderly_train_with_konwledge")
    
def preprocess_data_for_gnn(raw_dataset,
                            model,
                            tokenizer):
    """Transform the graph knowledge into graph: {x, edge index, edge attr}.

    Args:
        raw_dataset (Dataset): the dataset which has the column('reaction', 'condition', 'temperature', 'graph_knowledge')
    """

    raw_dataset = raw_dataset.select(range(0, 15000))
    graph = []
    valid_index = []
    for i in tqdm(range(len(raw_dataset))):
        res = {}
        example = raw_dataset[i]
        graph_knowledge = example['graph_knowledge']
        x = getEmbeddings(graph_knowledge, model, tokenizer)
        similarity_matrix = getSimilarityMatrix(embeddings=x)
        adjacent_matrix = getAdjacentMatrix(similarity_matrix=similarity_matrix, threshold=0.40)
        edge_index = convertAdjacentMtrix2EdgeIndex(adjacent_matrix=adjacent_matrix)
        # some graph don't have any edges. Just skip this index.
        if len(edge_index) == 0:
            continue
        else:
            valid_index.append(i)
            edge_attr = torch.ones(len(edge_index[0]), 2, dtype=torch.long)
            res['edge_attr'] = edge_attr.cpu().detach().numpy()
            
        res['x'] = x.cpu().detach().numpy()
        res['edge_index'] = edge_index.cpu().detach().numpy()
        # res['edge_attr'] = edge_attr

        graph.append(res)
    
    valid_dataset = raw_dataset.select(valid_index)
    graph_data = Dataset.from_dict({"graph": graph})
    
    dataset_concat = concatenate_datasets([valid_dataset, graph_data], axis=1)
    dataset_concat.save_to_disk("./data/chem_data/orderly_train_with_graph_test")
    

if __name__ == '__main__':
    # dataset = Dataset.load_from_disk("./data/chem_data/orderly_train_with_knowledge")
    # model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/', device_map='auto')
    # tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/')
    # preprocess_data_for_gnn(dataset, model, tokenizer)
    dataset = Dataset.load_from_disk("./data/chem_data/orderly_train_with_graph_test")
    print(dataset[0].keys())
    smiles_for_grover = []
    for i in range(len(dataset)):
        condition = lambda x : ("Reactant" in x or "Product" in x) and "Functional" not in x
        res = list(filter(condition, dataset[i]['graph_knowledge']))
        res = [ss.replace("Reactant: ","").replace("Product: ","").replace("[", "").replace("]","") for ss in res]
        smiles_for_grover.append(res)
    
    smiles_for_grover = Dataset.from_dict({"origin_smiles": smiles_for_grover})    
    dataset = concatenate_datasets([dataset, smiles_for_grover], axis=1)
    # print(dataset[0])
    # # print(dataset[0]['graph']['x'])
    print(dataset[2]['origin_smiles'])
    print(dataset.features)
    #print(torch.tensor(dataset.select(range(0, 10))['graph']['x']).shape)