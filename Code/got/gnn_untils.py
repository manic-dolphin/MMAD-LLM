from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig 
from gnn import *
from gnn_llama import *
from dschat.utils.model.model_utils import print_trainable_parameters
# TODO
def extractKnowledgeNode(text: str,
                         batch_extract=True
                         ):
    """

    Args:
        text (str): batch of text 
        batch_extract (bool, optional): whether to use batch extract
    """
    if batch_extract:
        n = len(text)
        for i in range(n):
            cur_text = text[i]

def getEmbeddings(nodes: List[str],
                 model,
                 tokenizer
                 ):
    """_summary_

    Args:
        nodes (List[str]): extract knowledge nodes list. (n, )
    """
    n = len(nodes)
    res = []
    for i in range(n):
        cur_node = nodes[i]
        model.eval()
        input_ids = tokenizer(cur_node, return_tensors='pt').to("cuda")
        # hidden_states = model(**input_ids).last_hidden_state
        hidden_states = torch.squeeze(model(**input_ids, output_hidden_states=True).hidden_states[0]).sum(dim=0)
        # print(hidden_states.shape)
        res.append(hidden_states)
        
    return torch.stack(res)

def getSimilarityMatrix(embeddings: List[torch.Tensor],
                        ):
    n = len(embeddings)
    similarity_matrix = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        embed_0 = embeddings[i]
        for j in range(n):
            embed_1 = embeddings[j]
            similarity_matrix[i][j] = F.cosine_similarity(embed_0, embed_1, dim=0)
        
    return similarity_matrix # float32 (n, n)

def getAdjacentMatrix(similarity_matrix,
                      threshold=0.45):
    mask = similarity_matrix > threshold
    n = len(similarity_matrix)
    adjacent_matrix = torch.masked_fill(torch.zeros((n, n), dtype=torch.long), mask=mask, value=1)
    
    return adjacent_matrix

def convertAdjacentMtrix2EdgeIndex(adjacent_matrix):
    
    edge_index = []
    n = len(adjacent_matrix)
    for row in range(n):
        for col in range(n):
            if row != col and adjacent_matrix[row][col] == 1:
                edge_index.append([row, col])
    
    return torch.tensor(edge_index, dtype=torch.long).T
                     
if __name__ == '__main__':
#     text = []
#     text1 = """
#     Certainly! Here is the response based on the provided chemical reaction:

#     Reactants:
#     Reactant 1: [BrC(Br)(Br)Br]. Functional Group 1: Aryl halide (Br), Functional Group 2: Alkyl halide (Br).

#     Reactant 2: [OCc1cc2ccccc2cc1I]. Functional Group 1: Ether (-OC-), Functional Group 2: Aryl iodide (I).

#     Products:
#     Product 1: [BrCc1cc2ccccc2cc1I]. Functional Group 1: Aryl halide (Br), Functional Group 2: Aryl iodide (I).

#     Reaction Types:
#     Reaction Type 1: Aryl halide coupling reaction - Formation of an aryl-aryl bond.
#     Reaction Type 2: Halogen exchange reaction - Replacement of an aryl bromide with an aryl iodide.

#     Note: The identification of functional groups and reaction types is based on the chemical structures and known organic chemistry principles.
#     """
#     text.append(text1)
#     text1 = text1.split("\n")
#     text1 = [s.strip() for s in text1]
#     text1 = [s for s in text1 if s != "" and len(s) >= 25]
#     print(text1)
    # extract_knowledge_nodes = [["Reactant: [BrC(Br)(Br)Br]", 
    #                      "Reactant: Functional Group: Aryl halide (Br)",
    #                      "Reactant: Functional Group: Alkyl halide (Br)",
    #                      "Reactant: [OCc1cc2ccccc2cc1I]",
    #                      "Reactant: [OCc1cc2ccccc2cc1I]: Functional Group: Ether (-OC-)",
    #                      "Reactant: Functional Group: Aryl iodide (I)",
    #                      "Product: [BrCc1cc2ccccc2cc1I]",
    #                      "Product: Functional Group: Aryl halide (Br)",
    #                      "Product: Functional Group: Aryl iodide (I)",
    #                      "Reaction Type: Aryl halide coupling reaction - Formation of an aryl-aryl bond",
    #                      "Reaction Type: Halogen exchange reaction - Replacement of an aryl bromide with an aryl iodide."
    #                      ],
    #                     ["Reactant: [BrC(Br)(Br)Br]", 
    #                      "Reactant: Functional Group: Aryl halide (Br)",
    #                      "Reactant: Functional Group: Alkyl halide (Br)",
    #                      "Reactant: [OCc1cc2ccccc2cc1I]",
    #                      "Reactant: [OCc1cc2ccccc2cc1I]: Functional Group: Ether (-OC-)",
    #                      "Reactant: Functional Group: Aryl iodide (I)",
    #                      "Product: [BrCc1cc2ccccc2cc1I]",
    #                      "Product: Functional Group: Aryl halide (Br)",
    #                      "Product: Functional Group: Aryl iodide (I)",
    #                      "Reaction Type: Aryl halide coupling reaction - Formation of an aryl-aryl bond",
    #                      "Reaction Type: Halogen exchange reaction - Replacement of an aryl bromide with an aryl iodide."
    #                      ]]
    model = AutoModelForCausalLM.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/', device_map='auto')
    print_trainable_parameters(model)
    # tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/')
    # x = getEmbeddings(extract_knowledge_nodes, model, tokenizer).to(model.device)
    # print(x.shape)
    # matrix = getSimilarityMatrix(embeddings=x)
    # print(matrix)
    # adjacent_matrix = getAdjacentMatrix(matrix)
    # print(adjacent_matrix)
    # edge_index = convertAdjacentMtrix2EdgeIndex(adjacent_matrix=adjacent_matrix).to(model.device)
    # print(edge_index)
    # edge_attr = torch.ones(2, len(edge_index[0]), 2, dtype=torch.long).to(model.device)
    
    # graph = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
    # gnnLlama = GnnLlamaForCausalLM.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/')