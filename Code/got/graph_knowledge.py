from typing import List, Optional, Iterator
from collections import deque
import torch
import json
import numpy as np
import re
import logging
from incontext_prompt import SCORE_EXAMPLES_PROMPTS
from collections import deque

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
        self.parents = []
        
class Operations:
    
    def __init__(self,
                 model,
                 initial_problem: str,
                 max_gen_len: int=128,
                 max_temperature: float=0.5,
                 subject: str='math',
                 prompt_config_dir='./prompts/prompts.json'
                 ):
        
        self.model = model
        self.initial_problem = initial_problem
        self.max_gen_len = max_gen_len
        self.max_temperature = max_temperature
        self.subject = subject
        if self.subject == 'math':
            self.prompt_config = json.load(open('./prompts/math_prompts.json'))
        else:
            self.prompt_config = json.load(open(prompt_config_dir))
            
        self.generate_prompt = self.prompt_config['generate_prompt']
        self.refine_prompt = self.prompt_config['refine_prompt']
        self.aggregate_prompt = self.prompt_config['aggregate_prompt']
        self.score_prompt = self.prompt_config['score_prompt']
        self.aggregate_gen_length = 128
    
    def aggregate(self,
                  thoughts: List[Thought]):
        l = len(thoughts)
        # decide how many nodes should be aggreated
        size = torch.randint(2, l, (1,))
        # sample indices to aggrearted
        index = list(set(np.array(torch.randint(0, l, (size,)).cpu())))
        states = [thoughts[i].state for i in index]
        s = ""
        for i in range(len(states)):
            s += "state {}: ".format(i + 1) + states[i]
        # prompt = self.aggregate_prompt + str(states)
        if self.subject == 'math':
            prompt = self.aggregate_prompt.format(s)
        else:
            prompt = self.aggregate_prompt + s
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
        
        if self.subject == 'math':
            prompt = self.generate_prompt.format(self.initial_problem, thought.state)
        else:
            prompt = self.generate_prompt + thought.state
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ]]
        m = torch.distributions.Uniform(0.0, self.max_temperature)
        temperature = m.sample()
        new_state = self.model.chat_completion(
                    dialogs=messages,
                    temperature=float(temperature),
                    max_gen_len=self.max_gen_len
                    )
        new_thought = Thought(new_state[0]['generation']['content'])
        
        return new_thought
    
    def refine(self,
               thought: Thought) -> Thought:
        
        if self.subject == 'math':
            prompt = self.refine_prompt.format(thought.state)
        else:
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
        # prompt = self.score_prompt.format(SCORE_EXAMPLES_PROMPTS, self.initial_problem, current_state)
        prompt = SCORE_EXAMPLES_PROMPTS.format(self.initial_problem, current_state)
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ]]
        score: int= 0
        output = self.model.chat_completion(
                        dialogs=messages,
                        temperature=0.0,
                        max_gen_len=self.max_gen_len
                    )
        value = output[0]['generation']['content']
        try: 
            score = int(re.findall(r"Score: \[(\d+)\]", value)[0])
        except IndexError as e:
            # TODO
            # score = score(self, thought)
            pass
            
        return score,value
        
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
        
    def find_n_max_score(self, scores: List[int]):
        
        if len(scores) <= self.max_breadth:
            return None
        indices = np.argsort(scores)[-self.max_breadth:]
        
        return indices
    
    # TODO
    def plot_graph(self):
        pass
    
    def consturct_graph(self, history_thoughts: List[Thought]):
        root = history_thoughts[0]
        q = deque()
        q.append(root)
        graph = []
        graph.append(root)
        visited = {}
        for i in range(len(history_thoughts)):
            visited[history_thoughts[i]] = False
        while len(q) != 0:
            cur = q.popleft()
            successors = cur.successors
            successors_new = []
            for thought in successors:
                if thought in history_thoughts and visited[thought] == False:
                    visited[thought] = True
                    successors_new.append(thought)
                    thought.parents.append(cur)
                    q.append(thought)
                    graph.append(thought)
                    
            cur.successors = successors_new
        
        return graph
                       
    def reason(self,
            #    max_breadth: int,
               max_depth: int,
               max_generate_nodes_number_per_node: int=2):
        
        assert max_depth > 0
        
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
            if cur_node_number > 2:
                for i in range(int(2 ** depth_flag)):
                    a_prob = torch.rand(1)
                    if a_prob > 1 - self.aggregate_prob:
                        new_thought = self.operations.aggregate(self.current_thoughts)
                        new_thoughts.append(new_thought)
                    else: continue
            
            # rate the generated new thoughts, and update the graph
            scores = [self.operations.score(thought) for thought in new_thoughts]
            indices = self.find_n_max_score(scores)
            if indices is None:
                self.current_thoughts = new_thoughts
                for thought in self.current_thoughts:
                    self.previous_thoughts.append(thought)
            else:
                self.current_thoughts = [new_thoughts[i] for i in indices]
                for thought in self.current_thoughts:
                    self.previous_thoughts.append(thought)