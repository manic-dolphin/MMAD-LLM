from llama2.llama import Llama
from planner import Planner
from models.models import Planner_Model
from graph import Thought, GoT, Operations

if __name__ == '__main__':
    pm = Planner_Model('llama2/llama-2-13b-chat/',
                       'llama2/tokenizer.model'
                       )
    # model = Llama.build(
    #         ckpt_dir='llama2/llama-2-13b-chat/',
    #         tokenizer_path='llama2/tokenizer.model',
    #         max_seq_len=512,
    #         max_batch_size=4
    #     )
    model = pm.get_model()
    planner_model = Planner(model=model,
                            max_gen_len=128)
    # print(planner_model.get_initial_state("please compute 10!, here is the answer: "))
    initial_state = planner_model.get_initial_state("please compute 10!, here is the answer: ")
    operations = Operations(model,
                            "please compute 10!, here is the answer: ")
    initial_thought = Thought(initial_state)
    got = GoT(initial_thought=initial_thought,
              operations=operations,
              )
    print(operations.generate(initial_thought).state)
    print("------------------")
    print(operations.score(initial_thought))
    got.reason(5)
    print(got.previous_thoughts)