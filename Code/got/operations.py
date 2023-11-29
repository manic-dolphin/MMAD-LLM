from graph import Thought
import json

class Operations:
    
    def __init__(self,
                 model,
                 max_gen_len: int=32,
                 prompt_config_dir='./prompts/prompts.json'):
        self.model = model
        self.max_gen_len = max_gen_len
        self.prompt_config = json.load(open(prompt_config_dir))
        self.generate_prompt = self.prompt_config['generate_prompt']
    
    def aggregate(self,
                  thoughts: List[Thought]):
        pass
    
    def generate(self,
                 thought: Thought):
        
        prompt = self.generate_prompt + thought.state
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ]]
        
        new_thought = self.model.chat_completion(
                    dialogs=messages,
                    temperature=0.0,
                    max_gen_len=self.max_gen_len
                    )
    
    def refine(self,
               thought: Thought):
        pass
    
    def score(self,
              thought: Thought):
        pass