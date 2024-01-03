from datasets import Dataset, load_dataset
import json
import pandas as pd
import tqdm
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import torch
from typing import List
import os
# os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'

def data_processing(data,
                    ):
    
    conditions = {}
    # 根据关键字分割字符串
    result = []
    keywords = ["Reactants are:", "Reagents are:", "and Products are 0:"]
    example = data['INSTRUCTION']
    response = data['RESPONSE']

    # 在每个关键字后面进行分割
    for i in range(len(keywords) - 1):
        start = example.find(keywords[i])
        end = example.find(keywords[i + 1])

        if start != -1 and end != -1:
            # last_comma_index = example.rfind(",")
            # conditions[keywords[i]] = example[start + len(keywords[i]):end].strip()
            example_ = example[start + len(keywords[i]):end].strip()
            # 去除逗号之后的内容
            last_comma_index = example_.rfind(",")
            example_ = example_[:last_comma_index]
            conditions[keywords[i]] = example_

    # 处理最后一个关键字后的内容
    last_key = keywords[-1]
    last_start = example.find(last_key)
    if last_start != -1:
        # conditions[last_key] = example[last_start + len(last_key):].strip()
        example_ = example[last_start + len(last_key):].strip()
        last_comma_index = example_.rfind(",")
        example_ = example_[:last_comma_index]
        conditions[last_key] = example_
        
    for value in conditions.values():
        result.append(value)
    reaction = result[0]
    condition = "".join(result[1: -1])
    product = result[-1]
    example_final = {'reaction': "Here is a chemical reaction formula, reagents are: {" + reaction + "} ,products are: {" + product +"}", 
                     'condition': condition,
                     'response': response
                     }
    
    return example_final

def get_dataset():
    
    dataset = load_dataset('json', data_files='./data/chem_data/llama_chem_v1.json', split='train')
    dataset = dataset.select(range(0, 5000))
    dataset_ = []
    
    for data in dataset:
        dataset_.append(data_processing(data))
    
    dataset_ = pd.DataFrame(dataset_, columns=['reaction', 'condition', 'response'])
    dataset_ = Dataset.from_pandas(dataset_)
    
    return dataset_

def get_knowledge_graph(data,
                        model,
                        tokenizer,
                        model_type: str
                        ):
    
    prompt = """
    Given a chemical reaction and its relevant conditions, 
    construct a knowledge graph based on the following aspects: 1. Functional groups; 2. Types of chemical reactants; 3. Types of chemical reactions. 
    The purpose of building this knowledge graph is to precisely and reasonably deduce the relevant conditions for a given chemical reaction in the future. 
    Therefore, the constructed knowledge graph needs to align with this task. 
    The chemical reaction is: {}. The relevant conditions for the chemical reaction are: {}. 
    Please construct the knowledge graph.
    """.format(data['reaction'], data['condition'])
    
    if model_type != 'llama':
        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        output = model.generate(**model_inputs, max_new_tokens=2048, do_sample=True, temperature=0.2)
        knowledge_graph = tokenizer.batch_decode(output)[0]
        
        return knowledge_graph
    
    else:
        pass

def extract_answer(knowledge_graph):
    
    answer_index = knowledge_graph.find("## Answer:")
    knowledge_graph_ = knowledge_graph[answer_index + len("## Answer:"): ]
    knowledge_graph_ = knowledge_graph_.strip()
    
    return knowledge_graph_

def init_model(model_dir,
               mp_size
               ):
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    ds_engine = deepspeed.init_inference(
        model,
        mp_size=mp_size,
        dtype=torch.float32,
        replace_with_kernel_inject=True
    )
    
    model = ds_engine.module
    
    return tokenizer, model

def construct_vector_base(texts: List[str]):
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                   model_kwargs={'device':"cuda"})
    vs_path = './vector-store'
    texts = [Document(page_content=text) for text in texts]
    texts_ = [text.page_content for text in texts]
    docs = embeddings.embed_documents(texts_)
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(vs_path)
    
    return docs

if __name__ == '__main__':
    
    dataset = get_dataset()
    print(dataset[0])
    
    # tokenizer, model = init_model('./hf_models/Mistral-7B-Instruct-v0.1',
    #                               mp_size=1
    #                               )
    # knowledge_graph = get_knowledge_graph(dataset[0],
    #                                       model=model,
    #                                       tokenizer=tokenizer,
    #                                       model_type='wizard-lm'
    #                                       )
    # # print(knowledge_graph)
    # print(extract_answer(knowledge_graph))
    vs_path = './vector-store'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                   model_kwargs={'device':"cuda"})
    texts = [dataset[0]['reaction'], dataset[0]['response'], dataset[1]['reaction']]
    construct_vector_base(texts=texts)
    vector_store = FAISS.load_local(vs_path, embeddings)
    related_docs_with_score = vector_store.similarity_search_with_score(query="天道酬勤", k=1)
    print(related_docs_with_score)