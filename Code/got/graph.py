from typing import List, Optional, Iterator
from collections import deque
import torch
import json
import numpy as np
import re

class Thought:
    """
    One thought represents a node in the graph, it contains the reasoning content of the model.
    """
    def __init__(self,
                 state: str,
                 value: int=0
                 ):
        self.predecessors: List[Thought] = []
        self.successors: List[Thought] = []
        self.state = state
        self.value = value
        self.refine_tag = False
        
class Operations:
    
    def __init__(self,
                 model,
                 initial_problem: str,
                 max_gen_len: int=32,
                 prompt_config_dir='./prompts/prompts.json'
                 ):
        
        self.model = model
        self.initial_problem = initial_problem
        self.max_gen_len = max_gen_len
        self.prompt_config = json.load(open(prompt_config_dir))
        self.generate_prompt = self.prompt_config['generate_prompt']
        self.refine_prompt = self.prompt_config['refine_prompt']
        self.aggregate_prompt = self.prompt_config['aggregate_prompt']
        self.score_prompt = self.prompt_config['score_prompt']
    
    def aggregate(self,
                  thoughts: List[Thought]):
        l = len(thoughts)
        # decide how many nodes should be aggreated
        size = torch.randint(2, l, (1,))
        # sample indices to aggrearted
        index = list(set(np.array(torch.randint(0, l, (size,)))))
        states = [thoughts[i].state for i in index]
        prompt = self.aggregate_prompt + str(states)
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ]]
        new_state = self.model.chat_completion(
                    dialogs=messages,
                    temperature=0.0,
                    max_gen_len=self.max_gen_len
                    )
        new_thought = Thought(new_state[0]['generation']['content'])
        # update the graph
        for i in index:
            thoughts[i].successors.append(new_thought)
            new_thought.predecessors.append(thoughts[i])
        
        return new_thought
    
    def generate(self,
                 thought: Thought) -> Thought:
        
        prompt = self.generate_prompt + thought.state
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ]]
        
        new_state = self.model.chat_completion(
                    dialogs=messages,
                    temperature=0.0,
                    max_gen_len=self.max_gen_len
                    )
        new_thought = Thought(new_state[0]['generation']['content'])
        
        return new_thought
    
    def refine(self,
               thought: Thought) -> Thought:
        
        prompt = self.refine_prompt + thought.state
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ]]
        
        new_state = self.model.chat_completion(
                    dialogs=messages,
                    temperature=0.0,
                    max_gen_len=self.max_gen_len
                    )
        new_thought = Thought(new_state[0]['generation']['content'])
        new_thought.refine_tag = True
        
        return new_thought
    
    def score(self,
              thought: Thought) -> int:
        
        current_state = thought.state
        prompt = self.score_prompt.format(self.initial_problem, current_state)
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ]]
        score: int= -1
        while score < 0 or score > 10:
            output = self.model.chat_completion(
                        dialogs=messages,
                        temperature=0.0,
                        max_gen_len=self.max_gen_len
                        )
            value = output[0]['generation']['content']
            print(value)
            value = re.findall(r'Score:\s*\d+/\d+', value)[0]
            score = int(value[7])
        
        thought.value = score
        
        return score
        
class GoT:
    """
    Implement the reasoning algorithm.
    """
    def __init__(self,
                 initial_thought: Thought,
                 operations: Operations,
                 max_breadth: int=3,
                 aggregate_prob: float=0.5,
                 refine_prob: float=0.2,
                 ):
        
        self.previous_thoughts, self.current_thoughts = [], []
        self.previous_thoughts.append(initial_thought)
        self.current_thoughts.append(initial_thought)
        
        self.operations = operations
        self.max_breadth = max_breadth
        self.aggregate_prob = aggregate_prob
        self.refine_prob = refine_prob
        
    def find_n_max_score(self, scores: List[Optional]):
        
        if len(scores) <= self.max_breadth:
            return None
        indices = np.argsort(scores)[-self.max_breadth:]
        
        return indices
    
    # TODO
    def plot_graph(self):
        pass
    
    def reason(self,
               max_breadth: int,
               max_depth: int,
               max_generate_nodes_number_per_node: int=2):
        
        assert max_breadth > 0 and max_depth > 0
        
        depth_flag = 0
        
        while depth_flag < max_depth:
            depth_flag += 1
            # number of nodes in the current graph level
            cur_node_number = len(self.current_thoughts)
            new_thoughts: List[Thought] = []
            
            for i in range(cur_node_number):
                # generate new thoughts
                for j in range(max_generate_nodes_number_per_node):
                    cur_thought = self.current_thoughts[i]
                    new_thought = self.operations.generate(cur_thought)
                    cur_thought.successors.append(new_thought)
                    new_thought.predecessors.append(cur_thought)
                    new_thoughts.append(new_thought)

                # self refine
                r_prob = torch.rand(1)
                if r_prob <= self.refine_prob:
                    new_thought = self.operations.refine(cur_thought)
                    new_thoughts.append(new_thought)
                    
            # aggregate the thoughts
            if cur_node_number > 1:
                for i in range(int(2 ** depth_flag)):
                    a_prob = torch.rand(1)
                    if a_prob > 0.5:
                        new_thought = self.operations.aggregate(self.current_thoughts)
                        new_thoughts.append(new_thought)
                    else: continue
            
            # rate the generated new thoughts, and update the graph
            scores = [self.operations.score(thought) for thought in new_thoughts]
            indices = self.find_n_max_score(scores)
            if indices == None:
                self.current_thoughts = new_thoughts
                for thought in self.current_thoughts:
                    self.previous_thoughts.append(thought)
            else:
                self.current_thoughts = [new_thoughts[i] for i in indices]
                for thought in self.current_thoughts:
                    self.previous_thoughts.append(thought)