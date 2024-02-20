from datasets import Dataset, load_dataset
import json
import pandas as pd
import tqdm
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from evaluate.visualization import radar_plot
import deepspeed
import torch
from typing import List
from llama2.llama import Llama
from collections import deque
import logging
import os
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'

# logging.basicConfig(filename='construct_knowledge_0112.log',
#                     filemode='a',
#                     format='%(message)s',
#                     level=logging.DEBUG
#                     )
# logger = logging.getLogger(__name__)

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
    # dataset = dataset.select(range(0, 5000))
    dataset_ = []
    
    for data in dataset:
        dataset_.append(data_processing(data))
    
    dataset_ = pd.DataFrame(dataset_, columns=['reaction', 'condition', 'response'])
    dataset_ = Dataset.from_pandas(dataset_, split='train')
    
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

class knowledge_node:
    
    def __init__(self, state):
        self.parents = []
        self.childs = []
        self.state = state
    
    def get_parents(self):
        return self.parents
    
    def get_childs(self):
        return self.childs
    
    def get_state(self):
        return self.state

def init_knowledge_model():
    
    model = Llama.build(
            ckpt_dir='llama2/llama-2-13b-chat/',
            tokenizer_path='llama2/tokenizer.model',
            max_seq_len=4096,
            max_batch_size=4
        )
    
    return model

def get_message_for_llama(prompt):
    
    return [[{"role": "user", "content": prompt}]]

def construct_knowledge(
                        data,
                        max_depth=3,
                        max_breadth=2
                        ) -> List[knowledge_node]:
    
    model = init_knowledge_model()
    
    reaction, condition = data['reaction'], data['condition']
    # initial_prompt = """
    # Given a chemical reaction and its conditions, extract the formation-related chemical knowledge or concepts, as well as make reasoned inferences, 
    # from three aspects: 1. Functional groups; 2. Reactants involved in the chemical reaction; 3. Chemical reactions, etc. 
    # These generated contents must be accurate and systematic. 
    # Note that the purpose of this knowledge is, in the future, to deduce the required reaction conditions solely based on the given chemical reaction. 
    # The chemical reaction is {}. The reaction conditions are {}.
    # """.format(reaction, condition)
    # initial_prompt_0 = """
    # Given a chemical reaction and its conditions, extract the formation-related chemical knowledge or concepts, as well as make reasoned inferences, 
    # from the perspective of: Functional groups. 
    # These generated contents must be accurate and systematic. 
    # Note that the purpose of this knowledge is, in the future, to deduce the required reaction conditions solely based on the given chemical reaction. 
    # The chemical reaction is {}. The reaction conditions are {}.
    # """.format(reaction, condition)
    initial_prompt_0 = """
    Generate a detailed functional group analysis based on the following specific chemical reaction equation. 
    The analysis should include a thorough understanding of the types and positions of functional groups in the reactants and products. 
    Provide detailed information about the functional groups, covering their structures, chemical properties, and potential transformations during the reaction. 
    Ensure that your analysis reflects a rich understanding of chemical knowledge and use scientific and chemical terminology for detailed explanations.
    The chemical reaction is: {}.
    Please consider the following:
    1.Identify and describe the functional groups present in the reactants and products.
    2.Analyze potential changes in functional groups during the reaction, including possible additions, eliminations, or transformations.
    3.Describe the structures and properties of the functional groups, as well as their potential roles in the reaction.
    4.Consider any potential catalytic or participatory roles of functional groups in the reaction.
    5.Use professional scientific terminology to ensure that the generated analysis reflects a profound understanding of functional groups.
    """.format(reaction)
    # initial_prompt_1 = """
    # Given a chemical reaction and its conditions, extract the formation-related chemical knowledge or concepts, as well as make reasoned inferences, 
    # from the perspective of: Reactants involved in the chemical reaction. 
    # These generated contents must be accurate and systematic. 
    # Note that the purpose of this knowledge is, in the future, to deduce the required reaction conditions solely based on the given chemical reaction. 
    # The chemical reaction is {}. The reaction conditions are {}.
    # """.format(reaction, condition)
    initial_prompt_1 = """
    Based on the specific chemical reaction equation provided below, provide a detailed analysis to gain a profound understanding of the essence of the reaction. 
    Focus on the reaction type, reaction conditions, and potential involvement of novel catalysts.
    The chemical reaction is: {}.
    Please elaborate on the following aspects, incorporating chemical knowledge for explanation:
    1.Reaction Type: Determine and elaborate on the specific type of the reaction, such as nucleophilic substitution, addition reaction, redox reaction, etc. Explain how the substrate structure and reaction mechanism influence the categorization.
    2.Reaction Conditions: Analyze the applicable reaction conditions, including temperature, pressure, solvent selection, etc. Provide detailed explanations of how these conditions impact the reaction rate and selectivity, considering principles from reaction kinetics and thermodynamics.
    3.Novel Catalysts: Consider whether novel catalysts are involved in the reaction. If so, describe their structure, potential catalytic mechanisms, and advantages compared to traditional catalysts, incorporating principles of catalyst design.
    4.Please use professional scientific and chemical terminology to ensure that the generated analysis fully reflects a profound understanding of the chemical reaction.
    """.format(reaction)
    # initial_prompt_2 = """
    # Given a chemical reaction and its conditions, extract the formation-related chemical knowledge or concepts, as well as make reasoned inferences, 
    # from the perspective of: Chemical reactions. 
    # These generated contents must be accurate and systematic. 
    # Note that the purpose of this knowledge is, in the future, to deduce the required reaction conditions solely based on the given chemical reaction. 
    # The chemical reaction is {}. The reaction conditions are {}.
    # """.format(reaction, condition)
    initial_prompt_2 = """
    Generate a detailed reaction mechanism based on the following specific chemical reaction equation. 
    Consider each step of the reaction, providing structural information for intermediates, depicting transition states, and indicating any potential involvement of catalysts and reaction conditions. 
    Explain key aspects of the reaction mechanism, including the formation and cleavage of bonds in each step.
    The chemical reaction is: {}.
    Please note the following:
    1.If possible, provide structures for intermediates and transition states.
    2.Describe possible reaction pathways, including steps involving the formation and cleavage of bonds.
    3.Consider whether catalysts are involved in the reaction and specify any critical reaction conditions.
    4.If there is uncertainty or multiple possibilities for any reaction step, offer relevant discussions or comments.
    5.Express the explanation in scientific and chemical terminology to ensure accuracy and professionalism in the generated reaction mechanism.
    """.format(reaction)
    
    prompts = [initial_prompt_0, initial_prompt_1, initial_prompt_2]
    messages = [get_message_for_llama(prompt) for prompt in prompts]
    # initial_state = model.chat_completion(
    #             dialogs=messages,
    #             temperature=0.0,
    #             max_gen_len=1024
    # )[0]['generation']['content']
    initial_states = [model.chat_completion(
        dialogs=message,
        temperature=0.0,
        max_gen_len=1024
    )[0]['generation']['content'] for message in messages]
    initial_nodes = [knowledge_node(initial_state) for initial_state in initial_states]
    # TODO
    # explore_prompt = """
    # From the perspective of a chemistry expert and leveraging relevant chemical knowledge, 
    # explore and expand upon the existing knowledge in breadth and depth. 
    # The expansion must adhere to the following conditions: 
    # 1. The expanded content must be accurate and reasonable; 
    # 2. The purpose of expanding the knowledge is to better accomplish a task, wherein, given a chemical reaction, inferring the reaction conditions is required; 
    # 3. The expanded content should form a cohesive paragraph, maintaining a length roughly consistent with the original content. The existing knowledge is: .
    # """
    explore_prompt_0 = """
    Building upon the previously generated knowledge about functional groups, the model has provided some reasoning or additional information. 
    Please, on the basis of this generated content, further extend the understanding of functional groups, ensuring that the expansion remains within the domain of functional group-related knowledge and does not encompass other chemical areas.
    The previsously generated knowledge is {}.
    Please elaborate on the following:
    1.Summary of Reasoning or Knowledge: Summarize the reasoning or knowledge generated by the model regarding functional groups, including their structures, properties, and reaction characteristics.
    2.In-Depth Expansion: Building on the existing knowledge, delve deeper into the properties of functional groups, their reaction mechanisms, and potential application areas.
    3.Broadening the Scope: Extend the knowledge about functional groups to encompass new categories or derivative structures. Consider different types of functional groups and their roles in various compounds.
    Use concise scientific terminology, ensuring that the generated content represents a purposeful extension of knowledge about functional groups in terms of both depth and breadth.
    """
    explore_prompt_1 = """
    Building upon the previously provided specific chemical reaction equations, the model has conducted in-depth analyses regarding reaction type, reaction conditions, and potential involvement of novel catalysts. 
    Please, based on these analyses, further extend the knowledge about the chemical reactions themselves, ensuring that the expansion remains within the domain of chemical reactions and does not encompass other chemical areas.
    The previsously generated knowledge is {}.
    Please provide detailed explanations for the following points, incorporating chemical knowledge:
    1.Reaction Mechanism: Elaborate on the mechanism of the chemical reaction, including potential intermediates, transition states, and changes in bonding. Explain the influence of substrate structures and reaction conditions on the reaction mechanism.
    2.Reaction Kinetics: Analyze the principles of reaction rates and selectivity, considering factors such as concentration, temperature, etc. Delve into the chemical principles underlying reaction kinetics.
    3.Reaction Applications: Extend the application of the chemical reaction to new scenarios, considering different substrate structures and reaction conditions. Provide detailed insights into potential uses of the reaction in practical chemical synthesis or other fields.
    Use professional scientific and chemical terminology to ensure that the generated content represents a purposeful extension of knowledge about chemical reactions themselves in terms of both depth and breadth.
    """
    explore_prompt_2 = """
    Building upon the previously generated knowledge about reaction mechanisms, the model has provided some reasoning or additional information. 
    Please, on the basis of this generated content, further extend the understanding of reaction mechanisms, ensuring that the expansion remains within the domain of reaction mechanism-related knowledge and does not encompass other chemical areas.
    The previsously generated knowledge is {}.
    Please elaborate on the following:
    1.Summary of Reasoning or Knowledge: Summarize the reasoning or knowledge generated by the model regarding reaction mechanisms, including potential intermediates, transition states, and related aspects.
    2.In-Depth Expansion: Building on the existing knowledge, delve deeper into the details of reaction mechanisms, including potential reaction pathways, catalytic mechanisms, and other relevant aspects.
    3.Broadening the Scope: Extend the knowledge about reaction mechanisms to encompass new reaction types or categories. Consider the mechanisms of different reaction types and their applications in organic synthesis.
    Use concise scientific terminology, ensuring that the generated content represents a purposeful extension of knowledge about reaction mechanisms in terms of both depth and breadth.
    """
    explore_prompts = [explore_prompt_0, explore_prompt_1, explore_prompt_2]
    q = deque()
    for initial_node in initial_nodes:
        q.append(initial_node)
        
    count = 0
    
    for _ in range(max_depth):
        
        l = len(q)
        
        for i in range(l):
             
            cur = q.popleft()
            cur_state = cur.state
            # BUG
            # content = explore_prompt + cur_state
            content = explore_prompts[i].format(cur_state)
            message = get_message_for_llama(content)
            
            for j in range(max_breadth):
                
                t = float(torch.rand(1)[0])
                new_state = model.chat_completion(
                    dialogs=message,
                    temperature=t,
                    max_gen_len=2048
                )[0]['generation']['content']
                logging.info("{} nodes has been created.".format(count + 1))
                count += 1
                new_knowledge_node = knowledge_node(new_state)
                q.append(new_knowledge_node)
                cur.childs.append(new_knowledge_node)
                new_knowledge_node.parents.append(cur)
                
    return initial_nodes

def bfs(nodes):
    
    level = 0
    logging.info("start...")
    
    q = deque()
    # q.append(node)
    for node in  nodes:
        q.append(node)
    while len(q) != 0:
        l = len(q)
        logging.info("level {}".format(level))
        logging.info("#####################################")
        level += 1
        
        for i in range(l):
            
            cur = q.popleft()
            logging.info("no {}: {}".format(i + 1, cur.state))
            childs = cur.childs
            for child in childs:
                q.append(child)
        
        logging.info("#####################################")
    
    logging.info("complete")
            

if __name__ == '__main__':
    
    dataset = get_dataset()
    print(dataset[100])
    data = dataset[100]
    
    initial_nodes = construct_knowledge(data, max_depth=5, max_breadth=1)
    bfs(initial_nodes)
    
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
    # vs_path = './vector-store'
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
    #                                model_kwargs={'device':"cuda"})
    # texts = [dataset[0]['reaction'], dataset[0]['response'], dataset[1]['reaction']]
    # construct_vector_base(texts=texts)
    # vector_store = FAISS.load_local(vs_path, embeddings)
    # related_docs_with_score = vector_store.similarity_search_with_score(query="天道酬勤", k=1)
    # print(related_docs_with_score)
    
    # bleu = evaluate.load('bleu')
    # rouge = evaluate.load('rouge')
    # predictions = ["hello there general kenobi", "foo bar foobar"]
    # references = ["hello there general bert", "foo bar barfoo"]
    # results = bleu.compute(predictions=predictions, references=references)
    # print(results)
    # results = rouge.compute(predictions=predictions, references=references)
    # print(results)
    # data = [results]
    # model_names = ['mistral-7b']
    # plot = radar_plot(data=data, model_names=model_names)
    # plot.show()
    # plot.savefig('radar.png')