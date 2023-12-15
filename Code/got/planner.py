import json

class Planner:
    """
    model: the model to serve as a planner, such as a LLM.
    max_gen_len: the max length of the initial plan that the model generated.
    prompt_config_dir: the dir of the prompt config file.
    """
    def __init__(self, 
                 model,
                 subject:str="math",
                 max_gen_len=512,
                 prompt_config_dir='./prompts/prompts.json'):
        
        self.model = model
        self.max_gen_len = max_gen_len
        self.subject = subject
        if subject == 'math':
            prompt_config = json.load(open('./prompts/math_prompts.json'))
        else:
            prompt_config = json.load(open(prompt_config_dir))
            
        self.plan_prompt = prompt_config['planner_prompt']
        
    
    def get_initial_state(self,
                          problem: str
                          ):
        plan_prompt = problem + self.plan_prompt
        messages = [[
            {
                "role": "user",
                "content": plan_prompt
            }
        ]]
        
        state_0 = self.model.chat_completion(
                    dialogs=messages,
                    temperature=0.0,
                    max_gen_len=self.max_gen_len
                    )
        
        return state_0[0]['generation']['content']